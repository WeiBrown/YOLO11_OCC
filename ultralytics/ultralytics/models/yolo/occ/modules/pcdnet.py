import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, in_ch=3, out_ch=128, hidden_ch=256):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_ch,hidden_ch)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_ch,out_ch)
    
    def forward(self,x):
        return self.linear2(self.relu(self.linear1(x)))

class PcdNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=128, hidden_ch=256):
        super(PcdNet, self).__init__()
        self.pcdfe = MLP(in_ch,out_ch,hidden_ch)
        self.apply(self.init_weights)

    def init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        
    def forward(self, pcd, spixel=None, dspixel=None):
        pcdf = self.pcdfe(pcd)
        return pcdf