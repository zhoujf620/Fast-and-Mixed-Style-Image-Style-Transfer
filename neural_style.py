import os
from os.path import basename, splitext
import sys
import re
import time
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg import Vgg16
from option import Options


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    vis = utils.Visualizer(env='style_transfer_zhoujf')

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    content_transform = transforms.Compose([
        # the shorter side is resize to match image_size
        transforms.Resize(args.content_size),
        transforms.CenterCrop(args.content_size),
        transforms.ToTensor(),  # to tensor [0,1]
        transforms.Lambda(lambda x: x.mul(255))  # convert back to [0, 255]
    ])
    content_dataset = datasets.ImageFolder(args.content_dataset, content_transform)
    # to provide a batch loader
    content_loader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        # transforms.Resize(args.style_size), 
        # transforms.CenterCrop(args.style_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))  # mul 255 and then normalize
    ])

    style_image = [f for f in os.listdir(args.style_dir)]
    style_num = len(style_image)
    print("There is {} style images.".format(style_num))

    style_list = []
    feature_s_list = []
    gram_s_list = []
    for i in range(style_num):
        style = utils.load_image(os.path.join(args.style_dir, style_image[i]), size=args.style_size) # style_size None 先试一下
        style = style_transform(style)
        style = style.repeat(args.batch_size, 1, 1, 1).to(device)
        style_list.append(style)

    # for only one style image
    # style_image = utils.load_image(args.style_image, size=args.style_size)
    # style_image = style_transform(style_image)
    # style_image = style_image.repeat(args.batch_size, 1, 1, 1).to(device)

        feature_s = vgg(utils.normalize_batch(style)) # [0, 1]
        feature_s_list.append(feature_s)
        gram_s = [utils.gram_matrix(y) for y in feature_s]
        gram_s_list.append(gram_s)

    for epoch in range(args.epochs):
        since = time.time()

        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0

        for batch_id, (x, _) in enumerate(content_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()  # initialize with zero gradients

            x = x.to(device)
            y = transformer(x)  # transformer_net input [0, 255]

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)  # vgg input [0, 1]
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss_list = []
            for i in range(style_num):
                style_loss = 0.
                for ft_y, gm_s in zip(features_y, gram_s_list[i]):
                    gm_y = utils.gram_matrix(ft_y)
                    style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
                style_loss_list.append(style_loss)
            style_loss = 0.
            style_loss_sum = sum(style_loss_list)
            for loss in style_loss_list:
                style_loss = style_loss + loss ** 2
            style_loss = style_loss / style_loss_sum * args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                ave_content_loss = agg_content_loss / args.log_interval
                ave_style_loss = agg_style_loss / args.log_interval

                time_elapsed = time.time() - since
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}\tcost time: {:.0f}m {:.0f}s".format(
                    time.ctime(), epoch + 1, count, len(content_dataset),
                    ave_content_loss, ave_style_loss, ave_content_loss + ave_style_loss,
                    time_elapsed // 60, time_elapsed % 60)
                print(mesg)

                # use visdom
                win_name = str(basename(args.style_dir)) +\
                    "_" + "{:.0e}".format(args.content_weight) + \
                    "_" + "{:.0e}".format(args.style_weight) + \
                    "_" + "{:.0e}".format(args.lr)
                vis.plot_many_stack({
                    'content loss': ave_content_loss,
                    'style loss': ave_style_loss,
                    'total loss': ave_content_loss + ave_style_loss},
                    win_name=win_name,
                    xlabel='log interval({} batch)'.format(args.log_interval),
                    ylabel='loss')

                since = time.time()
                agg_content_loss = 0.
                agg_style_loss = 0.
        
        # save checkpoint every epoch
        if args.checkpoint_model_dir is not None: # and (batch_id + 1) % args.checkpoint_interval == 0:
            transformer.eval().cpu()
            ckpt_model_filename = str(basename(args.style_dir)) +\
                            "_" + "ckpt_epoch_" + str(epoch+1) + \
                            "_" + "{:.0e}".format(args.content_weight) + \
                            "_" + "{:.0e}".format(args.style_weight) + \
                            "_" + "{:.0e}".format(args.lr)+ ".pth"
            ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
            torch.save(transformer.state_dict(), ckpt_model_path)
            transformer.to(device).train()
    
    # save model
    transformer.eval().cpu()
    save_model_filename = str(basename(args.style_dir)) + \
                            "_" + "epoch_" + str(args.epochs) + \
                            "_" + "{:.0e}".format(args.content_weight) + \
                            "_" + "{:.0e}".format(args.style_weight) + \
                            "_" + "{:.0e}".format(args.lr)+ ".pth"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    content_image = utils.load_image(args.content_image, size=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x.mul(255))         
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        since = time.time()
        output = style_model(content_image).cpu()
        time_elapsed = time.time() - since 
        print('stylize cost time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# 5s
    save_transfered_image_path = args.output_dir + \
                    'Content_' + str(splitext(basename(args.content_image))[0]) + \
                    '_Model_' + str(splitext(basename(args.model))[0]) + '.jpg'
    utils.save_image(save_transfered_image_path, output[0])
    print("\nDone, transfered image saved at", save_transfered_image_path)


if __name__ == "__main__":
    # figure out the experiments type
    args = Options().parse()
    print('\n', args, '\n')

    # args.content_dataset = './images/'
    # train(args)

    if args.subcommand is None:
        raise ValueError("ERROR: specify either train or eval")
        # sys.exit(1) # 1 for all other types of error besides syntax
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try to run on CPU")
        # sys.exit(1)
    elif not args.cuda and torch.cuda.is_available():
        print("Cuda is available, try to run on GPU")

    if args.subcommand == 'train':
        check_paths(args)
        train(args)
    elif args.subcommand == 'eval':
        stylize(args)
    else:
        raise ValueError('Unknow experiment type')


# train: CUDA_VISIBLE_DEVICES=0 python neural_style.py train 
# mosaic 