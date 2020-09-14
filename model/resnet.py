#-*- coding:utf-8 -*-
import torch.nn as nn
import math
import sys
import time

import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

sys.path.append('../')
from config import cfg
from IPython import embed

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(5, stride=1)
        # self.fc = nn.Linear(10752 * block.expansion, num_classes)        
        self.avgpool = nn.AvgPool2d((5,25), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #---前面是32倍下采样，如果输入是224，则这里是224/32=36
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNetTemplet(nn.Module):

    def __init__(self, block, layers, input_channel):
        self.inplanes = 64
        super(ResNetTemplet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1000)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

def LoadPretrainedModel(model, pretrained_state_dict):
    model_dict = model.state_dict()
    union_dict = {k : v for k,v in pretrained_state_dict.items() if k in model_dict}
    model_dict.update(union_dict)
    return model_dict


def resnet18(pretrained=False, num_classes=88):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(torch.load(cfg.pretrained_path))
    num_features=model.fc.in_features
    model.fc=nn.Linear(num_features,num_classes)
    return model



def Resnet18Templet(pretrained=False,input_channel=3, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetTemplet(BasicBlock, [2, 2, 2, 2], input_channel, **kwargs)
    if pretrained:
        model_dict = LoadPretrainedModel(model,torch.load(cfg.pretrained_path))
        model.load_state_dict(model_dict)
    return model



class ResidualBlock(nn.Module):
    def __init__(self,inchannels,outchannels,stride = 1,need_shortcut = False):
        super(ResidualBlock,self).__init__()
        self.right = nn.Sequential(
            nn.Conv2d(inchannels,outchannels,kernel_size = 3,stride = stride,padding = 1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(True),
            nn.Conv2d(outchannels,outchannels,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(outchannels)
         )
        if need_shortcut:
            self.short_cut = nn.Sequential(
                nn.Conv2d(inchannels,outchannels,kernel_size = 1,stride = stride),
                nn.BatchNorm2d(outchannels)
            )
        else:
            self.short_cut = nn.Sequential()
    
    def forward(self,x):
        out = self.right(x)
        out += self.short_cut(x)
        return F.relu(out)

#---用以训练二分类的键盘图像(输入112x32),因为其输入图像尺寸较小
class ResNet_112_32(nn.Module):
    def __init__(self, Data_type, input_channel=3, base_channel=8, num_classes=2):
        super(ResNet_112_32,self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(input_channel, base_channel, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        
        self.make_layers = nn.Sequential(
            ResidualBlock(base_channel,base_channel,stride=1,need_shortcut=True),
            ResidualBlock(base_channel,base_channel,stride=1,need_shortcut=False),
            ResidualBlock(base_channel,base_channel*2,stride=2,need_shortcut=True),
            ResidualBlock(base_channel*2,base_channel*2,stride=1,need_shortcut=False),
            ResidualBlock(base_channel*2,base_channel*4,stride=2,need_shortcut=True),
            ResidualBlock(base_channel*4,base_channel*4,stride=1,need_shortcut=False),
            ResidualBlock(base_channel*4,base_channel*8,stride=2,need_shortcut=True),
            ResidualBlock(base_channel*8,base_channel*8,stride=1,need_shortcut=False)
        )
        self.fc = nn.Linear(base_channel * 8 * 4, num_classes)
        #---用以进行区域位置的分类，3分类
        self.Data_type = Data_type
        if not self.Data_type is None:
            self.pos_fc = nn.Linear(base_channel * 8 * 4, 3)
        self.num_classes = num_classes


    def forward(self,x):
        out = self.pre_layer(x)
        out = self.make_layers(out)
        # out = F.dropout(out, p=0.5)
        out = out.view(-1, self.num_flatters(out))
        if self.Data_type is None:
            return self.fc(out)
        else:
            return self.fc(out), self.pos_fc(out)

    def num_flatters(self,x):
        sizes = x.size()[1:]
        result = 1
        for size in sizes:
            result *= size
        return result 

def Block_Net(input_channel=3, base_channel=8):
    Block = nn.Sequential(
        nn.Conv2d(input_channel, base_channel, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(base_channel),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        ResidualBlock(base_channel,base_channel,stride=1,need_shortcut=True),
        ResidualBlock(base_channel, base_channel, stride=1, need_shortcut=False),
        ResidualBlock(base_channel,base_channel*2,stride=2,need_shortcut=True)             
    )
    return Block

#---用以训练连续几帧图像的模型
class ResNet_Time(nn.Module):
    def __init__(self, k=5, Data_type=None, input_channel=3, base_channel=8, num_classes=2):
        super(ResNet_Time, self).__init__()
        self.k = k
        self.Block1_1 = Block_Net()
        self.Block1_2 = Block_Net()
        self.Block1_3 = Block_Net()
        if self.k == 5:
            self.Block1_4 = Block_Net()
            self.Block1_5 = Block_Net()
            self.Conv3d = nn.Conv3d(base_channel*2, base_channel*2, kernel_size=(5, 3, 3), stride=1, padding=(0, 1, 1))
        else:
            self.Conv3d = nn.Conv3d(base_channel*2, base_channel*2, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1))
        
        self.make_layers = nn.Sequential(
            ResidualBlock(base_channel*2,base_channel*2,stride=1,need_shortcut=False),
            ResidualBlock(base_channel*2,base_channel*4,stride=2,need_shortcut=True),
            ResidualBlock(base_channel*4,base_channel*4,stride=1,need_shortcut=False),
            ResidualBlock(base_channel*4,base_channel*8,stride=2,need_shortcut=True),
            ResidualBlock(base_channel* 8, base_channel* 8, stride=1, need_shortcut=False)
            # nn.Dropout()
        )
        self.fc = nn.Linear(base_channel * 8 * 4, num_classes)
        #---用以进行区域位置的分类，3分类
        self.Data_type = Data_type
        if not self.Data_type is None:
            self.pos_fc = nn.Linear(base_channel * 8 * 4, 3)
        self.num_classes = num_classes

    def forward(self, img_lists):
        #out->(8,27,7)  4倍下采样
        out_lists = []
        # for i, img in enumerate(img_lists):
        block1_out = self.Block1_1(img_lists[0])
        block2_out = self.Block1_2(img_lists[1])
        block3_out = self.Block1_3(img_lists[2])
        out_lists.append(block1_out)
        out_lists.append(block2_out)
        out_lists.append(block3_out)
        if self.k == 5:
            block4_out = self.Block1_4(img_lists[3])
            block5_out = self.Block1_5(img_lists[4])
            out_lists.append(block4_out)
            out_lists.append(block5_out)
        #---batchxcxkx14x4
        stack_input = torch.stack(out_lists, dim=2)
        #---batchxcx1x14x4  #---k即时间序列通道，经过卷积后变成1了0.0
        #---squeeze()函数最好指定维度的位置，不然容易在测试时把第一维的batch=1也忽略掉
        aggregation_out = (self.Conv3d(stack_input)).squeeze(2)
        out = self.make_layers(aggregation_out)
        # out = F.dropout(out, p=0.5)
        out = out.view(-1, self.num_flatters(out))
        if self.Data_type is None:
            return self.fc(out)
        else:
            return self.fc(out), self.pos_fc(out)

    def num_flatters(self,x):
        sizes = x.size()[1:]
        result = 1
        for size in sizes:
            result *= size
        return result 



if __name__=='__main__':
    # model=resnet18(pretrained=True)
    # input=torch.randn(1,3,145,800)
    # y=model(input)
    # print(y.size())

    input1 = torch.randn(12, 3, 112, 32)
    input2 = torch.randn(12, 3, 112, 32)
    inputs = [input1, input2, input1, input1, input1]
    # net = ResNet_112_32()
    net = ResNet_Time(k=5)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    stime = time.time()
    output = net(inputs)
    if isinstance(output, tuple):
        print(output[0].shape)
        print(output[1].shape)
    else:
        print(output.size())
    print('cost {}'.format(time.time()-stime))
