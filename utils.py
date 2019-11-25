import numpy as np 
from PIL import Image

import torch 
import visdom 


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2) # swapped ch and w*h, transpose share storage with original
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std 
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) # new_tensor for same type  of tensor
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0) # back to tensor within [0, 1]
    return (batch - mean) / std



class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}   

    def plot_many_stack(self, d, win_name, xlabel, ylabel):

        name = list(d.keys())
        x = self.index.get(win_name, 0)
        val=list(d.values())
        if len(val)==1:
            y=np.array(val)
        else:
            y=np.array(val).reshape(-1,len(val))

        opts = dict(legend=name, title=win_name, xlabel=xlabel, ylabel=ylabel)
        self.vis.line(Y=y, X=np.ones(y.shape)*x,
                    win=str(win_name),
                    opts=opts,
                    update=None if x == 0 else 'append'
                    )
        self.index[win_name] = x + 1


if __name__ == "__main__":
    vis = Visualizer(env='fast_neural_style')

    ave_content_loss = [100, 50, 30, 25, 23, 22, 21, 20]
    ave_style_loss = [80, 50, 40, 35, 32, 31, 20.5, 20.3]
    for i in range(len(ave_content_loss)):
        vis.plot_many_stack({
                        'content loss': ave_content_loss[i],
                        'style loss': ave_style_loss[i]},
                        win_name='train loss',
                        xlabel='log interval',
                        ylabel='loss')