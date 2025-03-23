# from .basics import *
import torch
import torch.nn as nn
out_channel_mv = 128
class Scaling_Net(nn.Module):
    def __init__(self, middle_units = 64, hidden_units=4*out_channel_mv):
        super(Scaling_Net,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, middle_units),
            nn.ReLU(),
            nn.Linear(middle_units, hidden_units),
            nn.ReLU()
        )
        # initialize_weights(self.layer,1)

    def forward(self,x):
        x = self.layer(x).view(1,-1,1,1)
        # print(x.shape)
        return x

def build_model():
    scaling_Net = Scaling_Net(hidden_units=128)
    input_lambda = torch.Tensor([128])
    output = scaling_Net(input_lambda).clamp(0,100)
    


if __name__ == "__main__":
   build_model()