# Preprocessing Enhanced Image Compression for Machine Vision
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

[[paper](https://ieeexplore.ieee.org/abstract/document/10632166)] [[code](https://github.com/XingtongGe/PreprocessingICM)]

[Guo Lu](https://guolusjtu.github.io/guoluhomepage/), [Xingtong Ge](https://xingtongge.github.io/), [Tianxiong Zhong](https://inkosizhong.github.io/), [Qiang Hu](https://qianghu-huber.github.io/qianghuhomepage/), [Jing Geng](https://pure.bit.edu.cn/en/persons/jing-geng)

IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2024

## Abstract

Recently, more and more images are compressed and sent to the back-end devices for machine analysis tasks (e.g., object detection) instead of being purely watched by humans. However, most traditional or learned image codecs are designed to minimize the distortion of the human visual system without considering the increased demand from machine vision systems. In this work, we propose a preprocessing enhanced image compression method for machine vision tasks to address this challenge. Instead of relying on the learned image codecs for end-to-end optimization, our framework is built upon the traditional non-differential codecs, which means it is standard compatible and can be easily deployed in practical applications. Specifically, we propose a neural preprocessing module before the encoder to maintain the useful semantic information for the downstream tasks and suppress the irrelevant information for bitrate saving. Furthermore, our neural preprocessing module is quantization adaptive and can be used in different compression ratios. More importantly, to jointly optimize the preprocessing module with the downstream machine vision tasks, we introduce the proxy network for the traditional non-differential codecs in the back-propagation stage. We provide extensive experiments by evaluating our compression method for several representative downstream tasks with different backbone networks. Experimental results show our method achieves a better trade-off between the coding bitrate and the performance of the downstream machine vision tasks by saving about 20% bitrate.

## News

* **2024/8**: 🔥 Our paper has been accepted by **IEEE TCSVT** !

* **2022/7**: 🎉 I won the 2022 **Outstanding Graduation Thesis Award** of Beijing Institute of Technology(BIT) for this work !

## Quick Started

### Requirements

[libbpg](https://bellard.org/bpg/)

mmcv-1.7.0, mmdetection-2.19.1, mmclassification, etc.

## Folders

### mmdetection_toward

This is the complete model training and testing framework **without** the QA module.

To set up the environment, follow the official installation guide from MMDetection v2.19.1:  
[Prerequisites — MMDetection 2.19.1 documentation](https://mmdetection.readthedocs.io/en/v2.19.1/get_started.html#installation)

After installing `mmcv` and installing `mmdetection_toward` locally, one modification is required in the `mmcv` source code:

In `/mmcv/image/photometric.py`, update the `imnormalize_` function by replacing the image normalization logic with simple normalization (scaling to [0, 1]). Modify the function as follows:

```python
def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std."""
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    # cv2.subtract(img, mean, img)
    # cv2.multiply(img, stdinv, img)
    img /= 255.0
    return img
```

Once this change is made, training and testing can proceed as described below.

### Training and Testing

Training and evaluation are based on the MMDetection framework.

Using FCOS as an example, we start with the config file:  
`configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py`

This config file should be customized for NPP training, including:
- Loading a pretrained model
- Setting the number of training steps
- Choosing optimizer and learning rate
- Adjusting the loss weights (inherited from `./fcos_r50_caffe_fpn_gn-head_1x_coco.py`)

The total loss function is defined as:  
**Loss = Bpp + 0.5 * Loss_pre + λ * L_FCOS**  
You should set the λ value in the config file before training.

#### Training

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py \
  -p proxy_network_path \
  -pre npp_pretrained_path(Optional) \
  --work-dir work_dir_path 
```

The trained `.pth` model will be saved in the specified working directory.

#### Testing

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py \
  pretrained_model_path \
  --eval bbox  # Evaluated using COCO mAP metrics
```

### Core Source Code Overview

```text
mmdet
├── core
├── datasets
├── models
│   ├── backbones
│   │   ├── towards
│   │   │   ├── ImageCompression (Proxy Network)
│   │   │   ├── models (NPP)
│   │   ├── resnet
```

The NPP_Codec is integrated into the MMDetection framework.  
Key logic is implemented in the `ResNet` class of `mmdetection_toward/mmdet/models/backbones/resnet.py`.

In the `__init__` method, both the proxy network and NPP are initialized. The gradients of the proxy network are frozen:

```python
self.pre_processor = Preprocessor()
self.image_compressor = ImageCompressor(192, 192)
for param in self.image_compressor.parameters():
    param.requires_grad = False
```

The forward pass includes preprocessing, encoding/decoding using the proxy network and BPG.


```python
    def forward(self, x):
        # Training
        # Preprocessing and Proxy Network
        img_num = x.size(0) # 得到batch_size
        ori_imgs = x
        x = self.pre_processor(x)
        x = torch.clamp(x, 0., 1.)
        learned_img = x
        nips_img, mse_loss, bpp_feature, bpp_z, bpp = self.image_compressor(learned_img)
        nips_bpp = bpp
        new_imgs = []
        bpg_bpps = []
        for img_i in range(img_num):
            now_img = learned_img[img_i]
            img_name = str(img_i)
            jpg_path = '/Path/to/mmdetection_toward/testimg/temppng/'+img_name+'_oldtrain.png'
            torchvision.utils.save_image(now_img,jpg_path)
            # BPG encoding/decoding
            bpg_path = '/Path/to/mmdetection_toward/testimg/tempbpg/'+img_name+'train.bpg'
            convert_jpg_bpg ='bpgenc -q 34 -o ' +bpg_path +' '+jpg_path
            os.system(convert_jpg_bpg)
            new_jpg_path = '/Path/to/mmdetection_toward/testimg/temppng/'+img_name+'_newtrain.png'
            convert_bpg_jpg = 'bpgdec -o '+new_jpg_path +" "+ bpg_path
            os.system(convert_bpg_jpg)
            new_img =  torch.from_numpy(np.asarray(Image.open(new_jpg_path))).cuda(x.device).permute(2,0,1)
            cmd = 'cat ' + bpg_path + ' |wc -c'
            byte = os.popen(cmd).read()
            byte = int(byte)
            bpp = (float)(byte*8)/(float)(new_img.size(1)*new_img.size(2))
            # print(bpp)
            bpg_bpps.append(bpp)
            new_imgs += [new_img]
            del new_img
        new_imgs = torch.stack(new_imgs,dim=0).float()
        bpg_bpps = torch.tensor(bpg_bpps).cuda()
        nips_img *= 255.0
        nips_img.data = new_imgs
        nips_bpp.data = bpg_bpps
        x = transform(nips_img)
        

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        # forward the learned_img, ori_imgs and bpp
        outs.append(learned_img)
        outs.append(ori_imgs)
        outs.append(nips_bpp)
        return tuple(outs)
```

### mmdetection_variable

This version builds on the previous setup by incorporating the QA module into the NPP.  
Key training logic is also updated to support QA-aware learning.

```text
mmdet
├── core
├── datasets
├── models
│   ├── backbones
│   │   ├── towards
│   │   │   ├── ImageCompression (Proxy Network)
│   │   │   ├── models (NPP) with Scaling_Net
│   │   ├── resnet
```

The training process remains the same, but there is **no need to manually set the loss weight** for λ.


### NPP_Codec

This directory contains the standalone NPP module (without the QA module) along with test code for image processing using the NPP module.

### NPP_Codec_Variable

This version includes the QA (Quantization-Aware) module integrated into the NPP source code.

## Acknowledgments

Our code was developed based on [mmcv](https://mmcv.readthedocs.io/en/latest/). This is a concise and easily extensible Computer Vision library.

## Citation

If you find our work useful or relevant to your research, please kindly cite our paper:

```
@ARTICLE{10632166,
  author={Lu, Guo and Ge, Xingtong and Zhong, Tianxiong and Hu, Qiang and Geng, Jing},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Preprocessing Enhanced Image Compression for Machine Vision}, 
  year={2024},
  volume={34},
  number={12},
  pages={13556-13568},
  keywords={Image coding;Task analysis;Codecs;Machine vision;Bit rate;Optimization;Neural networks;Image compression;machine vision;preprocessing;deep learning},
  doi={10.1109/TCSVT.2024.3441049}}

```