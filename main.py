import torch
import numpy as np
from decode import pre_model
from module import SAGE_Predictor
from warnings import simplefilter
import parse
import torch.nn as nn
simplefilter(action='ignore', category= Warning)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import os

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)
def calcultion(x,mean):

    x_mean = [(x[_] - mean)**2 for _ in range(len(x))]
    return  (sum(x_mean)/len(x))**0.5


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

pt1_path = r'./pt/pcm.pt'
pt2_path = r'./pt/model.pt'

net1 = pre_model(infeat=20,outfeat=1)
net2 = SAGE_Predictor(amio_feature=20,in_feats=320,hidden_feats1=160,hidden_feats2=96,out_feats=63,device='cpu')

net1.load_state_dict(torch.load(pt1_path))
net1.eval()
net2.load_state_dict(torch.load(pt2_path,map_location='cpu'),strict=False)
net2.eval()

def get_contact(seq,represtation,attention_map):

    with torch.no_grad():
        attention1,out1 = net1(attention_map)
        topk_matric,_,out2 = net2(seq,represtation,attention_map,out1)
        out1 = symmetrize(out1)
        out2 = symmetrize(out2)/2
        out = out2 + out1
    return out

if __name__ == '__main__':

    seq = torch.as_tensor(parse.parse_a3m(r'./example/1HR7.fasta'),dtype=torch.int64).squeeze(0)
    represtation = torch.from_numpy(np.load(r'./example/1HR7_representations.npy')).squeeze(0)[1:-1, :]
    attention_map = torch.from_numpy(np.load(r'./example/1HR7_attetnion.npy'))[:,1:-1,1:-1]

    attention_map = apc(symmetrize(attention_map)).permute(1, 2, 0)

    out = get_contact(seq,represtation,attention_map)

    L = out.shape[0]

    level = ticker.MaxNLocator(nbins=100).tick_values(0, 1)  # 用于再合适的位置选择出不超过N个间隔

    cmap = plt.get_cmap('terrain')
    norm_distance = colors.BoundaryNorm(level, ncolors=cmap.N, clip=True)  #

    xx, yy = torch.arange(L), torch.arange(L)

    xx = xx.unsqueeze(0).repeat(L, 1)
    yy = yy.unsqueeze(1).repeat(1, L)

    fig, ax = plt.subplots(nrows=1)

    at = ax.pcolormesh(xx, yy, out, cmap=cmap, norm=norm_distance)
    fig.colorbar(at, ax=ax)

    fig.tight_layout()

    plt.savefig(os.path.join(r'./example/', f'1HR7'))
    plt.close()