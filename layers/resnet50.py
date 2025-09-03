import torchvision.models as models
import torch.nn as nn

from typing import Sequence, List, Dict, Tuple

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained = pretrained)
        # Stem
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Stages
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for p in self.layer1.parameters():
            p.requires_grad = False

    def forward(self, x) -> Sequence[torch.Tensor]: 
        """
        dfjdsoifjsidof
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return (c5,)
    
class SingleScaleFPN(nn.Module):
    def __init__(self,
                 in_channels=2048,   # backbone 마지막 레벨 (C5)
                 out_channels=256,   # 원하는 출력 채널
                 num_outs=1,         # 출력 feature 개수 (기본 1개, P5)
                 add_extra_convs=False,  # 'on_output' 사용 가능
                 relu_before_extra_convs=False):
        super(SingleScaleFPN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs

        # C5 → out_channels 로 맞추는 conv
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: Tensor (N, 2048, H, W), backbone C5 출력
        return: Tuple of feature maps (P5, [P6, P7, ...])
        """
        # 1. C5 → lateral conv
        x=x[0]
        out = self.lateral_conv(x)
        out = self.fpn_conv(out)

        return (out,)
