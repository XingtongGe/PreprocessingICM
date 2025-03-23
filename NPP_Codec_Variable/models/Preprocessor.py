import torch 
import torch.nn as nn
from .UNet import UNet
from .Scaling_net import Scaling_Net
class Preprocessor(nn.Module):

    def __init__(self):
        super().__init__()
        # 上层分支
        self.conv1 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.Scaling_Net1 = Scaling_Net(16, 3)
        self.Scaling_Net2 = Scaling_Net(16, 3)
        # 下层分支
        self.UNet = UNet()
    
    def forward(self, x, scaling_lambda):
        # print(scaling_lambda)
        # print(scaling_lambda.shape)
        m1 = self.Scaling_Net1(scaling_lambda)
        m2 = self.Scaling_Net2(scaling_lambda)
        x1 = self.relu1(self.conv1(x) * m1)
        x2 = self.relu2(self.conv2(x1) * m2)
        x3 = self.UNet(x, scaling_lambda)
        return x2+x3

def build_model():
    pro = Preprocessor()
    x = torch.randn(4,3,256,256)
    out = pro(x)

if __name__ == '__main__':
    build_model()