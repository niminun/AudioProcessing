import math
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
import torch
import torch.nn as nn
import torch.nn.functional as functional

SCALE_FACTOR = 100


class LeakyBottleneck(Bottleneck):
    """ Bottleneck block with leaky relu"""
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(LeakyBottleneck, self).__init__(inplanes, planes, stride, downsample)
        self.relu = nn.LeakyReLU(inplace=True)  # overriding parent's relu.


class LeakyBasicBlock(BasicBlock):
    """ Basic block with leaky relu"""
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(LeakyBasicBlock, self).__init__(inplanes, planes, stride, downsample)
        self.relu = nn.LeakyReLU(inplace=True)  # overriding parent's relu.


class NonPaddedBasicBlock(BasicBlock):
    """
    BasicBlock variation, see:
    https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # cropping height and width to right size after convs
        h_diff = identity.shape[2] - out.shape[2]
        w_diff = identity.shape[3] - out.shape[3]
        out += identity[:, :,
                        int(h_diff/2.0): -int(h_diff/2.0 + 0.5),
                        int(w_diff/2.0): -int(w_diff/2.0 + 0.5)]

        return out


class ResNetDiscriminator(ResNet):
    """
    A discriminator based on ResNet.
    """
    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, LeakyBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetGenerator(nn.Module):
    """
    A ResNet-based generator network. Based on:
        https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf
    """

    def __init__(self, block, layers, zero_init_residual=False):

        super(ResNetGenerator, self).__init__()
        self.inplanes = 128

        self.pad1 = nn.ReflectionPad2d(sum(layers)*4)  # padding here to overcome un-padded convs in residual layers.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=9, padding=4, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 128, layers[2])
        self.layer4 = self._make_layer(block, 128, layers[3])

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU(inplace=True)
        self.deconv3 = nn.Conv2d(32, 1, kernel_size=9, padding=4, stride=1)
        self.bn6 = nn.BatchNorm2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, LeakyBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):

        # down-sampling layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # up-sampling layers
        x = self.deconv1(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.deconv2(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.deconv3(x)
        x = self.bn6(x)
        x = x.tanh() * SCALE_FACTOR

        return x
