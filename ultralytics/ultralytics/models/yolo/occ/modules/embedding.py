import torch.nn as nn
import torch

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FPN(nn.Module):
    def __init__(self, in_dims=[],hidden_dim=256, out_dim=72):
        super(FPN, self).__init__()
        self.lateral1 = nn.Conv2d(in_dims[0], hidden_dim, kernel_size=1)
        self.lateral2 = nn.Conv2d(in_dims[1], hidden_dim, kernel_size=1)
        self.lateral3 = nn.Conv2d(in_dims[2], hidden_dim, kernel_size=1)

        self.fusion = nn.Conv2d(hidden_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        c1,c2,c3 = x
        p3 = self.lateral3(c3)
        p2 = self.lateral2(c2) + self._upsample(p3)
        p1 = self.lateral1(c1) + self._upsample(p2)
        
        p1 = self.fusion(p1)
        p2 = self.fusion(p2)
        p3 = self.fusion(p3)

        return [p1, p2, p3]

    def _upsample(self, x):
        return nn.functional.interpolate(x, mode='nearest', scale_factor=2)
    
class Embedding_Module(nn.Module):
    def __init__(self, in_im=[], in_de=[], in_sp=[], in_dsp=[], hidden_ch=256, out_ch=72):
        super(Embedding_Module, self).__init__()
        sas = []
        cas = []
        fusions = []
        self.out_ch = out_ch
        self.fpn_d = FPN(in_de,hidden_ch,out_ch)
        self.fpn_dsp = FPN(in_dsp,hidden_ch,out_ch)
        self.fpn_sp = FPN(in_sp,hidden_ch,out_ch)
        
        for imi,di,spi,dspi in zip(in_im,in_de,in_sp,in_dsp):
            sas.append(SpatialAttention())
            # TODO: @VE 
            cas.append(ChannelAttention(out_ch*4))

        self.cas = nn.ModuleList(cas)
        self.sas = nn.ModuleList(sas)
        self.fusion = nn.Conv2d(out_ch*4,out_ch,kernel_size=1)
    
    def forward(self,x,pcdfs,sfs,dsfs,dfs):
        out = []
        sfs = self.fpn_sp(sfs)
        dsfs = self.fpn_dsp(dsfs)
        dfs = self.fpn_d(dfs)

        for imf,sf,dsf,df,ca,sa in zip(x,sfs,dsfs,dfs,self.cas,self.sas):
            inputf = torch.cat([imf,sf,dsf,df],dim=1)
            f = ca(inputf)*inputf
            f = sa(f)*f
            # import ipdb;ipdb.set_trace()
            f = self.fusion(f)
            out.append(f)
        
        return out