import torch 
import torch.nn as nn
from .UNet import UNet

class Preprocessor(nn.Module):

    def __init__(self):
        super().__init__()
        # 上层分支
        self.conv1 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # 下层分支
        self.UNet = UNet()
    
    def forward(self,x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))
        x3 = self.UNet(x)
        return x2 + x3

def build_model():
    pro = Preprocessor()
    # print("build success!")
    x = torch.randn(4,3,16,16)
    out = pro(x)
    print(out.shape)

if __name__ == '__main__':
    build_model()