import os
import string
import torch
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from sklearn.metrics import roc_auc_score


to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

def print_msa(seq):
    letter=[to1letter[data] for _,data in enumerate(to1letter)]

    seq_=''
    for i in seq:
        seq_+=letter[i]

    return seq_


# read A3M and convert letters into
# integers in the 0..20 range,

def parse_a3m(filename):

    msa = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename,"r"):

        # skip labels
        if line[0] == '>':
            continue

        # remove right whitespaces
        line = line.rstrip()

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa


def parse_pdb(pdb_path,mission='contact'):
    xyz=[]
    atmo_index = [0]
    line_data=[]
    with open(pdb_path) as f:
        for line in f:
            if line[0:4]=='ATOM':
                if int(line[22:26])>atmo_index[-1]:
                    atmo_index.append(int(line[22:26]))
                if line[17:20]=='GLY':
                    if line[13:15]=='CA':
                        xyz.append([float(line[30:38]), float(line[39:46]), float(line[47:55])])
                        #print(xyz[-1])
                elif line[13:15]=='CB':
                    xyz.append([float(line[30:38]),float(line[39:46]),float(line[47:55])])
                    #print(xyz[-1])

    xyz=torch.tensor(xyz)
    #print(xyz.shape)
    dist=torch.cdist(xyz[None,:,:],xyz[:,None,:],p=2)
    dist=dist.squeeze()
    #dist+=torch.eye(dist.shape[0])*999.9
    #print(dist.shape)
    if mission=='contact':
        isContact=torch.zeros(size=(dist.shape[0],dist.shape[1]))
        index=torch.where(dist<=8 )
        isContact[index]+=1
        #print(isContact)
        return isContact
    else:
        distance=torch.zeros(size=(dist.shape[0],dist.shape[1]))
        bins=np.linspace(2.5,20,37)


        distance[dist<=2.5]=0
        distance[dist>20]=36
        #print(dist)
        for i in range(1,len(bins)):
            t,k=torch.where(dist>bins[i])
            distance[t,k]=torch.max(distance[t,k],torch.as_tensor(i))

        return distance


def parse_hhr(hhr_paht):
    hhr_data=[]
    is_append=False
    with open(hhr_paht) as f:
        for line in f:

            if line[0]=='>':
                is_append=True
            elif is_append is False:
                is_append=False
            else:
                data=line.split('  ')
                data=[(d.split('=')[1]) for d in data]
                tt=[]
                for d in data:
                    if d[-1]!='%':
                        tt.append(float(d))
                    else:
                        tt.append(float(d.split('%')[0])/100)
                hhr_data.append(tt)
                is_append=False
    #print(hhr_data)
    hhr_data=np.array(hhr_data)
    #print(hhr_data.shape)
    hhr_mean_data=hhr_data.mean(axis=0)
    #print(hhr_mean_data)
    hhr_data=hhr_data/hhr_mean_data
    #print(hhr_data)
    return hhr_data

def parse_atab(atab_path,idx):
    datas=open(atab_path).readlines()
    count=-1
    atab_data=[]
    tt=[]
    for data in datas:
        if data[0]=='>':


            count+=1
        elif 'i' in data or 'j' in data:
            continue
        else:
            atab_data.append([count,int(data[0:5]),float(data[12:18]),float(data[19:26]),float(data[27:35])])
    #print(atab_data[0:140])

    atab_init=np.zeros(shape=(count+1,idx,3))
    for data in atab_data:
        i,j,d=data[0],data[1],data[2:]
        atab_init[i][j]=d

    return atab_init


def get_data(msa_path,hhr_path,atab_path):
    msa=parse_a3m(msa_path)
    hhr=parse_hhr(hhr_path)
    idx=len(msa[0])
    atab=parse_atab(atab_path,idx)
    print(f'msa.shape={msa.shape},atab.shape={atab.shape},hhr.shape={hhr.shape}')
    return msa,hhr,atab


def get_topk(data,k):

    values,key = torch.topk(input=(data.reshape(-1)),k=k)
    keys = []
    L = data.shape[0]
    for data in key:
        keys.append([int(data/L),data%L])

    return values,torch.as_tensor(keys)


def get_topks(datas,k):
    values_list,keys_list = [] , []

    for data in datas:
        values,key = get_topk(data,k)
        values_list.append(values)
        keys_list.append(key)

    return values_list,keys_list


def new_parse_pdb_1(path):
    letter=[data for _,data in enumerate(to1letter)]
    #print(letter)
    f = open(path,'r')
    seq = []

    xyz = []
    idx = []
    for lines in f.readlines():
        lines = lines[:-1]
        if lines[0:4] == 'ATOM':
            if lines[17:20] != 'GLY':
                if lines[13:15] == 'CB':
                    #print(lines)
                    if lines[17:20] == 'UNK':
                        continue
                    seq.append(letter.index(lines[17:20]))
                    xyz.append([float(lines[30:38]), float(lines[39:46]), float(lines[47:55])])
                    idx.append(int(lines[22:26]))
                    #print(seq[-1],xyz[-1],idx[-1])
            else:
                if lines[13:15] == 'CA':
                    #print(lines)
                    if lines[17:20] == 'UNK':
                        continue
                    seq.append(letter.index(lines[17:20]))
                    xyz.append([float(lines[30:38]), float(lines[39:46]), float(lines[47:55])])
                    idx.append(int(lines[22:26]))
                    #print(seq[-1],xyz[-1],idx[-1])


    return seq,xyz,idx


def parse_pdb_2(path):
    f = open(path,'r')
    line = []
    for lines in f.readlines():
        if lines[:3] == 'TER':
            break
        if lines[:4] == 'ATOM':
            if lines[17:20] == 'GLY':
                if lines[13:15] == 'CA':
                    line.append(lines[:-1])
            else:
                if lines[13:15] == 'CB':
                    line.append(lines[:-1])

    xyz = []
    seq = []
    for d in line:
        xyz.append([float(d[30:39]),float(d[39:47]),float(d[47:55])])
        #print(d)
        #print([(d[30:39]),(d[39:47]),(d[47:55])])
        seq.append(d[17:20])
    xyz_all = torch.as_tensor(xyz)

    L = len(seq)

    if len(seq)==0 or len(seq)!=xyz_all.shape[0]:
        return [],[],[]
    dist = torch.cdist(xyz_all[None, :, :], xyz_all[:, None, :], p=2).reshape(L,L)


    letter = [to1letter[data] for _, data in enumerate(to1letter)]

    right = [True for data in seq if data in to1letter ]

    if len(right) != L:
        return [],[],[]
    seq = [to1letter[data] for data in seq ]
    #print(seq)
    seq_index = [letter.index(data) for data in seq]

    return seq,seq_index,dist

