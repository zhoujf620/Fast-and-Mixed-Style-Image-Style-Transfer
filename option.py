import argparse


class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='parser for image style transfer')
        subparsers = self.parser.add_subparsers(
            title='subcommands', dest='subcommand')

        # training args
        train_arg = subparsers.add_parser(
            'train', help='parser for training arguments')
        # train_arg = argparse.ArgumentParser()
        train_arg.add_argument("--epochs", type=int, default=1, # 1个epoch够了
                                      help="number of training epochs, default is 1")
        train_arg.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
        train_arg.add_argument("--content-dataset", type=str,
                                      default='../../data/mscoco/train2014/',
                                      help="path to training dataset, the path should point to a folder "
                                      "containing another folder with all the training images")
        # train_arg_parser.add_argument("--style-image", type=str, default="images/21styles/mosaic.jpg",
        #                               help="path to style-image")
        train_arg.add_argument("--style-dir", type=str, default="images/star-wave",
                               help="path to style-image")
        train_arg.add_argument("--save-model-dir", type=str, default='saved_model/',
                                      help="path to folder where trained model will be saved.")
        train_arg.add_argument("--checkpoint-model-dir", type=str, default=None,
                                      help="path to folder where checkpoints of trained models will be saved")
        train_arg.add_argument("--content-size", type=int, default=256,
                                      help="size of training images, default is 256 X 256")
        train_arg.add_argument("--style-size", type=int, default=None,
                                      help="size of style-image, default is the original size of style image")
        train_arg.add_argument("--cuda", type=int, default=1,
                                      help="set it to 1 for running on GPU, 0 for CPU")
        train_arg.add_argument("--seed", type=int, default=42,
                                      help="random seed for training")
        train_arg.add_argument("--content-weight", type=float, default=1e5,
                                      help="weight for content-loss, default is 1e5")
        train_arg.add_argument("--style-weight", type=float, default=1e10,
                                      help="weight for style-loss, default is 1e10")
        train_arg.add_argument("--lr", type=float, default=1e-4,
                                      help="learning rate, default is 1e-3")
        train_arg.add_argument("--log-interval", type=int, default=500,
                                      help="number of images after which the training loss is logged, default is 500")

        # evaluation args
        eval_arg = subparsers.add_parser(
            "eval", help="parser for evaluation/stylizing arguments")
        eval_arg.add_argument("--content-image", type=str, required=True,
                                     help="path to content image you want to stylize")
        eval_arg.add_argument("--content-scale", type=float, default=None,
                                     help="factor for scaling down the content image")
        eval_arg.add_argument("--output-dir", type=str, default='images/output-images/',
                                     help="path for saving the output image")
        eval_arg.add_argument("--model", type=str, required=True,
                                     help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
        eval_arg.add_argument("--cuda", type=int, required=True,
                                     help="set it to 1 for running on GPU, 0 for CPU")

        # self.parser = train_arg
    def parse(self):
        return self.parser.parse_args()
