import torch
import torch.nn as nn
from models.Preprocessor import Preprocessor
from ImageCompression.model import ImageCompressor



class Towards(nn.Module):
    def __init__(self) -> None:
        super(Towards, self).__init__()
        self.preprocessor = Preprocessor()
        self.compressor = ImageCompressor(192, 192)
    def forward(self, input_image):
        image_feature = self.preprocessor(input_image)
        clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = self.compressor(image_feature)
        # loss函数里面最好再加上深度学习编码器编解码前后的mse loss
        

        return clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp 
