import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from .models import *


def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


#  返回模型的索引  加载模型参数信息
def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        # 更新部分模型参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=192):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net_nips(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net_nips(out_channel_N=out_channel_N)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.context_prediction = Context_prediction_net(out_channel_M=out_channel_M)
        self.entropy_parameters = Entropy_parameter_net(out_channel_M=out_channel_M)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64, input_image.size(3) // 64).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        # print("input_image: ",input_image.size())

        feature = self.Encoder(input_image)
        # print("feature: ",feature.size())
        #  通过feature的size()获取batch_size大小
        batch_size = feature.size()[0]

        # 超先验部分
        z = self.priorEncoder(feature)
        # 设置了net.train()
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        # print("compressed_z: ",compressed_z.size())
        # 获得估计部分的sigma参数
        recon_sigma = self.priorDecoder(compressed_z)
        # print("recon_sigma: ",recon_sigma.size())
        feature_renorm = feature
        if self.training: 
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        predict_context = self.context_prediction(compressed_feature_renorm)
        entropy_params = self.entropy_parameters(torch.cat((recon_sigma, predict_context), 1))
        recon_image = self.Decoder(compressed_feature_renorm)
        # entropy_params = recon_sigma
        mu = entropy_params[:, 0: self.out_channel_M, :, :]
        sigma = entropy_params[:, self.out_channel_M: self.out_channel_M * 2, :, :]
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        def feature_probs_based_sigma_nips(feature, mu, sigma):
            sigma = sigma.pow(2)
            sigma = sigma.clamp(1e-5, 1e5)
            gaussian = torch.distributions.normal.Normal(mu, sigma)
            # 计算概率probs 
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
            return total_bits, prob
        # print("feature")
        # print(compressed_feature_renorm)
        total_bits_feature, _ = feature_probs_based_sigma_nips(compressed_feature_renorm, mu, sigma)
        # print(total_bits_feature)
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z

        return clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp
