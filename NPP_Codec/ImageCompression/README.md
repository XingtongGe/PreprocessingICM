# Pytorch Reimplementation for Variational image compression with a scale hyperprior 
这个代码是NIPS18去掉context的版本（因为我提供的pretrain模型是基于这个版本的）。context的实现在context_module.py里也是有的，在ImageCompressor里把context_prediction和entropy_parameters相关的注释打开就可以了，不过打开的话需要重新训练或finetune一下。
## Training Script
```
CUDA_VISIBLE_DEVICES=0 /workspace/T/luguo/wangyancheng/ImageCompression/checkpoints/baseline/iter_450000.pth.tar
```

## Testing Script
```
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config.json -n baseline --test -p 模型路径
```

