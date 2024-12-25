import torch
import torch.nn as nn
from torchvision import models

class Depth_Model(nn.Module):
    def __init__(self, in_ch=3, out_ch=1024):
        super(Depth_Model, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        tempelate_model = models.resnet101(pretrained=True)
        self.stage1 = torch.nn.Sequential(*list(tempelate_model.children())[0:-4])
        self.stage2 = torch.nn.Sequential(*list(tempelate_model.children())[-4:-3])
        self.stage3 = torch.nn.Sequential(*list(tempelate_model.children())[-3:-2])

    def forward(self,x):
        # import ipdb;ipdb.set_trace()
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return [x1,x2,x3]
