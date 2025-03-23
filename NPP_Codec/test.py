from models.Preprocessor import Preprocessor
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from glob import glob
import torchvision
import cv2
import mmcv
import numpy as np
from torchvision.transforms import functional as F

# LoadImageFromFile可用可不用
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, filename):

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        return img

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str



class PreprocessDataSet(Dataset):
    def __init__(self, img_path_list) -> None:
        super().__init__()
        self.img_path_list = img_path_list
        self.load = LoadImageFromFile()
        self.transform = transforms.Compose([
            # transforms.Resize((512,512)),
            transforms.ToTensor(),])

    def __getitem__(self, index):
        path = self.img_path_list[index]
        # im = cv2.imread(path)
        im = self.load(path)
        cv2.cvtColor(im, cv2.COLOR_BGR2RGB, im)
        # im = torch.from_numpy(np.asarray(Image.open(path))).permute(2,0,1).float()
        im = im / 255.0
        im = torch.Tensor(im).permute(2,0,1)
        # im = Image.open(path).convert('RGB')
        # im = self.transform(im)
        return im
    
    def __len__(self):
        return len(self.img_path_list)

def load_preprocessor(model,f):
    '''
    读入以fcos为backbone的预训练模型, 提取其中的pre_processor权重赋给当前model
    '''
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        pre_processor_dict = {k[23:]: v for k, v in pretrained_dict['state_dict'].items() if 'pre_processor' in k}
        model_dict = model.state_dict()
        model_dict.update(pre_processor_dict)
        model.load_state_dict(model_dict)
    return 0

if __name__ == "__main__":
    model = Preprocessor()
    path = '/home/luguo/gxt/checkpoints/detection/stage2/iter_100000.pth'
    image_list = sorted(glob(os.path.join('/DATA/luguo/COCO2017/val2017', "*")))
    load_preprocessor(model, path)
    model = model.cuda()
    test_dataset = PreprocessDataSet(image_list)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = 1, # 使用BPG编码时，batch设置了1，串行编码
        shuffle = False,
        num_workers = 4,
        pin_memory = False
    )
    model.eval()
    count = 0
    bpps = []
    with torch.no_grad():
        for idx, inputs in enumerate(test_dataloader):
            img = inputs[0]
            # print(img.shape)
            ori_h = img.shape[1]
            ori_w = img.shape[2]
            pad_h = int(np.ceil(img.shape[1] / 16)) * 16
            pad_w = int(np.ceil(img.shape[2] / 16)) * 16
            height = max(pad_h - img.shape[1], 0)
            width = max(pad_w - img.shape[2], 0)
            img = F.pad(img, (0, 0, width, height))
            img = img.unsqueeze(0)
            img = img.cuda()
            output = model(img)
            output = torch.clamp(output, 0., 1.)
            # print(output.shape)
            out = F.crop(output[0], 0, 0, ori_h, ori_w)
            image_name = image_list[idx].split('/')[-1].split('.')[0]
            jpg_path = '/home/luguo/gxt/towards/41/learned/'+image_name+'.png'
            torchvision.utils.save_image(out, jpg_path)
            bpg_path = '/home/luguo/gxt/towards/41/bpg/'+image_name+'.bpg'
            convert_jpg_bpg ='bpgenc -q 41 -o ' +bpg_path +' '+jpg_path
            os.system(convert_jpg_bpg)
            new_jpg_path = '/home/luguo/gxt/towards/41/recon/'+image_name+'.png'
            convert_bpg_jpg = 'bpgdec -o '+new_jpg_path +" "+ bpg_path
            os.system(convert_bpg_jpg)
            cmd = 'cat ' + bpg_path + ' |wc -c'
            byte = os.popen(cmd).read()
            byte = int(byte)
            bpp = (float)(byte*8)/(float)(ori_h*ori_w)
            bpps.append(bpp)
            # print(bpp)
            count += 1
            if count % 20 == 0:
                print(count)
                print(bpp)
    print('average bpp: ',np.mean(bpps))