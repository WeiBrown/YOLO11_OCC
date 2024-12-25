import torch.nn as nn
import torchvision.models as models
import torch

class SPNet(nn.Module):
    def __init__(self,):
        super(SPNet,self).__init__()
        tempelate_model = models.regnet_y_3_2gf(pretrained=True)
        tempelate_model = torch.nn.Sequential(*list(tempelate_model.children())[0:-2])
        self.stem_in = torch.nn.Sequential(*list(tempelate_model.children())[0])
        self.block1 = torch.nn.Sequential(*list(tempelate_model.children())[1][0:-2])
        self.block2 = torch.nn.Sequential(*list(tempelate_model.children())[1][-2:-1])
        self.block3 = torch.nn.Sequential(*list(tempelate_model.children())[1][-1])
    
    def forward(self,x):
        x = self.stem_in(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        return [x1,x2,x3]
        