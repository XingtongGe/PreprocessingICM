# 训练
CUDA_VISIBLE_DEVICES=1 nohup python tools/train.py configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py \
--work-dir /home/wyc/gxt/mmdetection_variable/run/codec_variable2 >codec_variable_continue2.out 2>&1
-p /home/wyc/gxt/checkpoints/codec2048.pth.tar \

CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py \
-p /home/wyc/gxt/checkpoints/codec4096.pth.tar \
--work-dir /home/wyc/gxt/mmdetection-master/run/4096/ori_stage2

CUDA_VISIBLE_DEVICES=0 nohup python tools/train.py configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py \
    -p /home/wyc/gxt/ImageCompressContext/checkpoints/baseline/256_bpg41_256/iter_40000.pth.tar \
    --work-dir /home/wyc/gxt/mmdetection-master/run/256/0_5_res2 >codec256_0_5_mse2.out 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --config examples/example/config.json \
    -n baseline/1024_1024L \
    -p /home/wyc/gxt/ImageCompressContext/checkpoints/baseline/iter_2450000.pth.tar \
    --finetune \
    >codec1024_1024L.out 2>&1

# 测试

CUDA_VISIBLE_DEVICES=0 nohup python tools/test.py \
    configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py \
    /home/wyc/gxt/mmdetection_variable/run/codec_variable2/iter_160000.pth \
    --eval bbox >test_variable_160000_choice_bpg40_2.log 2>&1

CUDA_VISIBLE_DEVICES=0 nohup python tools/test.py \
    configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py \
    /home/wyc/gxt/mmdetection-master/run/1024/ori_stage2/iter_20000.pth \
    --eval bbox >test_stage2_1024_ori_20000.log 2>&1

/home/wyc/gxt/ImageCompressContext/checkpoints/baseline/256_bpg41_256/iter_20000.pth.tar

CUDA_VISIBLE_DEVICES=0 nohup python tools/test.py \
    configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py \
    /home/wyc/gxt/mmdetection-master/run/2048/lambda4/epoch_1.pth \
    --eval bbox >test_nips2048_lambda4_epoch1.log 2>&1

CUDA_VISIBLE_DEVICES=0 nohup python train.py --config examples/example/config.json \
    -n baseline/256_new_256 --test \
    -p  /home/wyc/gxt/ImageCompressContext/checkpoints/baseline/256_bpg41_256/iter_30000.pth.tar --finetune \
    >test_codec256_new_256.out 2>&1
# 测试其他backbone
# Faster-RCNN
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py \
    -p /home/wyc/gxt/checkpoints/codec128.pth.tar \
    /home/wyc/gxt/checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth \
    --eval bbox

CUDA_VISIBLE_DEVICES=0 nohup python tools/test.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py \
    -p /home/wyc/gxt/checkpoints/codec256.pth.tar \
    -prep /home/wyc/gxt/checkpoints/256/lambda_0_5_epoch_2.pth \
    /home/wyc/gxt/checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth \
    --eval bbox >test_fastercnn_nips256_with_pre.log 2>&1

CUDA_VISIBLE_DEVICES=0 nohup python tools/test.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py \
    -prep /home/wyc/gxt/checkpoints/stage2/4096_iter_60000.pth \
    /home/wyc/gxt/checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth \
    --eval bbox >test_fastercnn_bpg28.log 2>&1



# VPF cmake 
cmake .. \
  -DFFMPEG_DIR:PATH="/home/wyc/gxt/VPF/FFmpeg/build_x64_release_shared" \
  -DVIDEO_CODEC_SDK_DIR:PATH="/home/wyc/gxt/VPF/Video_Codec_SDK_10.0.26" \
  -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
  -DGENERATE_PYTORCH_EXTENSION:BOOL="1" \
  -DCMAKE_CUDA_COMPILER:PATH="/usr/local/cuda/bin/nvcc" \
  -DCMAKE_INSTALL_PREFIX:PATH="/home/wyc/gxt/VPF/VideoProcessingFramework/install" \
  -DPYTHON_INCLUDE_DIR=/home/wyc/anaconda3/include/python3.9 \
  -DPYTHON_LIBRARY=/home/wyc/anaconda3/lib/libpython3.9.so \
  -DPYTHON_EXECUTABLE=/home/wyc/anaconda3/bin/python \
  -DAVCODEC_INCLUDE_DIR="/home/wyc/gxt/VPF/FFmpeg/build_x64_release_shared/include" \
  -DAVCODEC_LIBRARY="/home/wyc/gxt/VPF/FFmpeg/build_x64_release_shared/lib/libavcodec.so" \
  -DAVFORMAT_INCLUDE_DIR="/home/wyc/gxt/VPF/FFmpeg/build_x64_release_shared/include" \
  -DAVUTIL_LIBRARY="/home/wyc/gxt/VPF/FFmpeg/build_x64_release_shared/lib/libavutil.so" \
  -DAVUTIL_INCLUDE_DIR="/home/wyc/gxt/VPF/FFmpeg/build_x64_release_shared/include" \
  -DAVFORMAT_LIBRARY="/home/wyc/gxt/VPF/FFmpeg/build_x64_release_shared/lib/libavformat.so" \
  -DSWRESAMPLE_LIBRARY="/home/wyc/gxt/VPF/FFmpeg/build_x64_release_shared/lib/libswresample.so"
