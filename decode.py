import torch
import numpy as np
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch
import dgl.nn.pytorch.conv
import math
import torch.nn as nn


class Block_ResNet_pre(torch.nn.Module):
    def __init__(self, d_module):
        super(Block_ResNet_pre, self).__init__()
        self.elu1 = torch.nn.LeakyReLU()
        self.batchnorm1 = torch.nn.InstanceNorm2d(num_features=d_module)

        self.conv1 = torch.nn.Conv2d(in_channels=d_module, out_channels=d_module, kernel_size=3,
                                     padding=1, stride=1)

        self.elu2 = torch.nn.LeakyReLU()
        self.batchnorm2 = torch.nn.InstanceNorm2d(num_features=d_module)

        self.conv2 = torch.nn.Conv2d(in_channels=d_module, out_channels=d_module, kernel_size=3
                                     , padding=1, stride=1)
        self.activation = torch.nn.LeakyReLU()
    def forward(self, pair):
        pair1 = pair.unsqueeze(0).permute(0, 3, 1, 2)
        pair1 = self.batchnorm1(pair1)  # (1,K,L,L)
        pair1 = self.elu1(pair1.squeeze(0).permute(1, 2, 0))

        pair1 = pair1.unsqueeze(0).permute(0, 3, 1, 2)
        pair_1 = self.conv1(pair1).squeeze(0).permute(1, 2, 0)  # (L,L,K)

        pair_1 = pair_1.unsqueeze(0).permute(0, 3, 1, 2)
        pair2 = self.batchnorm2(pair_1)

        pair2 = self.elu1(pair2.squeeze(0).permute(1, 2, 0))

        pair2 = pair2.unsqueeze(0).permute(0, 3, 1, 2)
        pair_2 = self.conv2(pair2).squeeze(0).permute(1, 2, 0)  # (L,L,K)

        return self.activation(pair_2 + pair)

class Block_ResNet(torch.nn.Module):
    def __init__(self, d_module,add_conv):
        super(Block_ResNet, self).__init__()
        self.elu1 = torch.nn.LeakyReLU()
        self.batchnorm1 = torch.nn.InstanceNorm2d(num_features=d_module)
        self.add_conv = add_conv
        self.conv1 = nn.Conv2d(in_channels=d_module, out_channels=d_module, kernel_size=3,
                                     padding=1, stride=1)

        self.elu2 = torch.nn.LeakyReLU()
        self.batchnorm2 = torch.nn.InstanceNorm2d(num_features=d_module)

        self.conv2 = nn.Conv2d(in_channels=d_module, out_channels=d_module, kernel_size=3
                                     , padding=1, stride=1)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, pair):
        pair1 = pair.unsqueeze(0).permute(0, 3, 1, 2)
        pair1 = self.batchnorm1(pair1)  # (1,K,L,L)
        pair1 = self.elu1(pair1.squeeze(0).permute(1, 2, 0))

        pair1 = pair1.unsqueeze(0).permute(0, 3, 1, 2)
        pair_1 = self.conv1(pair1).squeeze(0).permute(1, 2, 0)  # (L,L,K)

        pair_1 = pair_1.unsqueeze(0).permute(0, 3, 1, 2)
        pair2 = self.batchnorm2(pair_1)

        pair2 = self.elu1(pair2.squeeze(0).permute(1, 2, 0))

        pair2 = pair2.unsqueeze(0).permute(0, 3, 1, 2)
        pair_2 = self.conv2(pair2).squeeze(0).permute(1, 2, 0)  # (L,L,K)
        if self.add_conv==True:
            return self.activation(pair_2 + pair)
        else:
            return self.activation(pair_2)



class Block_ResNet_1(torch.nn.Module):
    def __init__(self, d_module):
        super(Block_ResNet_1, self).__init__()
        self.elu1 = torch.nn.LeakyReLU()
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=d_module)

        self.conv1 = torch.nn.Conv2d(in_channels=d_module, out_channels=d_module, kernel_size=3,
                                     padding=1, stride=1)

        self.elu2 = torch.nn.LeakyReLU()
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=d_module)

        self.conv2 = torch.nn.Conv2d(in_channels=d_module, out_channels=d_module, kernel_size=3
                                     , padding=1, stride=1)
        #self.activation = torch.nn.LeakyReLU()

    def forward(self, pair):
        pair1 = pair.unsqueeze(0).permute(0,3,1,2)
        pair1 = self.conv1(pair1)
        pair1 = self.elu1(self.batchnorm1(pair1).squeeze(0).permute(1,2,0))

        pair2 = pair1.unsqueeze(0).permute(0,3,1,2)
        pair2 = self.elu2(self.batchnorm2(self.conv2(pair2)).squeeze(0).permute(1,2,0))


        return pair2


class Block_ResNet_Module(torch.nn.Module):
    def __init__(self, d_module, num_layers,add_conv):
        super(Block_ResNet_Module, self).__init__()
        module_lists = []
        #module_lists = nn.ModuleList()
        self.num = num_layers
        for _ in range(num_layers):
            module_lists.append(Block_ResNet(d_module,add_conv))
        self.module_list = torch.nn.Sequential(*module_lists)

    def forward(self, data):
        for _ in range(self.num):
            data = self.module_list[_](data)
        return data

class Block_ResNet_Module_pre(torch.nn.Module):
    def __init__(self,d_module,num_layers):
        super(Block_ResNet_Module_pre, self).__init__()
        module_lists = []
        self.num = num_layers
        for _ in range(num_layers):
            module_lists.append(Block_ResNet_pre(d_module))
        self.module_list = torch.nn.Sequential(*module_lists)
    def forward(self,data):
        for _ in range(self.num):
            data = self.module_list[_](data)
        return data



class pre_model(torch.nn.Module):
    def __init__(self,infeat,outfeat):
        super(pre_model,self).__init__()
        self.conv = Block_ResNet_Module_pre(d_module=infeat,num_layers=10)
        self.bn1 = torch.nn.BatchNorm2d(num_features=infeat)
        self.w1 = torch.nn.Conv2d(in_channels=infeat, out_channels=outfeat,kernel_size=1, stride=1)
        self.activation = torch.nn.Softmax()
    def forward(self,attention):
        attention1 = self.conv(attention)
        attention2 = attention1.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
        attention2 = self.w1(self.bn1(attention2)).squeeze(0).permute(1, 2, 0).squeeze(2)
        return attention1,self.activation(attention2)
