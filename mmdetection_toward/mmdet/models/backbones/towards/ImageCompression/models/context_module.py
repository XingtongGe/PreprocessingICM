import math
import torch.nn as nn
import torch
import numpy as np


def same_padding(kernel_size):
    # assumming stride 1
    if isinstance(kernel_size, int):
        return kernel_size // 2
    else:
        return kernel_size[0] // 2, kernel_size[1] // 2


class MaskedConvolution2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 *args, mask='A', vertical=False, mask_mode="noblind", **kwargs):
        if "padding" not in kwargs:
            assert "stride" not in kwargs
            kwargs["padding"] = same_padding(kernel_size)
        remove = {"conditional_features", "conditional_image_channels"}
        for feature in remove:
            if feature in kwargs:
                del kwargs[feature]
        super(MaskedConvolution2D, self).__init__(in_channels,
                                                  out_channels, kernel_size, *args, **kwargs)
        Cout, Cin, kh, kw = self.weight.size()
        pre_mask = np.array(np.ones_like(self.weight.data.cpu().numpy())).astype(np.float32)
        yc, xc = kh // 2, kw // 2

        assert mask_mode in {"noblind", "turukin", "fig1-van-den-oord"}
        if mask_mode == "noblind":
            # context masking - subsequent pixels won't hav access
            # to next pixels (spatial dim)
            if vertical:
                if mask == 'A':
                    # In the first layer, can ONLY access pixels above it
                    pre_mask[:, :, yc:, :] = 0.0
                else:
                    # In the second layer, can access pixels above or even with it.
                    # Reason being that the pixels to the right or left of the current pixel
                    #  only have a receptive field of the layer above the current layer and up.
                    pre_mask[:, :, yc + 1:, :] = 0.0
            else:
                # All rows after center must be zero
                pre_mask[:, :, yc + 1:, :] = 0.0
                ### All rows before center must be zero # XXX: not actually necessary
                ##pre_mask[:, :, :yc, :] = 0.0
                # All columns after center in center row must be zero
                pre_mask[:, :, yc, xc + 1:] = 0.0

            if mask == 'A':
                # Center must be zero in first layer
                pre_mask[:, :, yc, xc] = 0.0
            # same pixel masking - pixel won't access next color (conv filter dim)
            # def bmask(i_out, i_in):
            #    cout_idx = np.expand_dims(np.arange(Cout) % 3 == i_out, 1)
            #    cin_idx = np.expand_dims(np.arange(Cin) % 3 == i_in, 0)
            #    a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
            #    return a1 * a2

            # for j in range(3):
            #    pre_mask[bmask(j, j), yc, xc] = 0.0 if mask == 'A' else 1.0

            # pre_mask[bmask(0, 1), yc, xc] = 0.0
            # pre_mask[bmask(0, 2), yc, xc] = 0.0
            # pre_mask[bmask(1, 2), yc, xc] = 0.0
        elif mask_mode == "fig1-van-den-oord":
            if vertical:
                pre_mask[:, :, yc:, :] = 0.0
            else:
                # All rows after center must be zero
                pre_mask[:, :, yc + 1:, :] = 0.0
                ### All rows before center must be zero # XXX: not actually necessary
                ##pre_mask[:, :, :yc, :] = 0.0
                # All columns after center in center row must be zero
                pre_mask[:, :, yc, xc + 1:] = 0.0

            if mask == 'A':
                # Center must be zero in first layer
                pre_mask[:, :, yc, xc] = 0.0
        elif mask_mode == "turukin":
            pre_mask[:, :, yc + 1:, :] = 0.0
            pre_mask[:, :, yc, xc + 1:] = 0.0
            if mask == 'A':
                pre_mask[:, :, yc, xc] = 0.0

        print("%s %s MASKED CONV: %d x %d. Mask:" % (mask, "VERTICAL" if vertical else "HORIZONTAL", kh, kw))
        print(pre_mask[0, 0, :, :])

        self.register_buffer("mask", torch.from_numpy(pre_mask))

    def __call__(self, x):
        self.weight.data = self.weight.data * self.mask
        return super(MaskedConvolution2D, self).forward(x)


class Context_prediction_net(nn.Module):
    '''
    Compress residual prior
    '''

    def __init__(self, out_channel_M=192):
        super(Context_prediction_net, self).__init__()
        self.conv1 = MaskedConvolution2D(out_channel_M, out_channel_M*2, 5, stride=1, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data,
                                     (math.sqrt(2 * (3 * out_channel_M) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)

    def forward(self, x):
        x = self.conv1(x)
        return x


class Entropy_parameter_net(nn.Module):
    '''
    Compress residual prior
    '''

    def __init__(self, out_channel_M=192):
        super(Entropy_parameter_net, self).__init__()
        self.conv1 = nn.Conv2d(768, 640, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (768 + 640) / (768 + 768))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(640, 512, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2 * (512 + 640) / (640 + 640))))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(512, 2*out_channel_M, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, (math.sqrt(2 * (512 + 384) / (512 + 512))))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)
