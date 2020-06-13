
mkdir -p ../pretrained_pytorch


for VER in lite0 lite1 lite2 lite3 lite4; do

  PYTHONPATH=../.. python load_tf_weights.py \
       --model_name efficientnet-$VER \
       --tf_checkpoint ../pretrained_tensorflow/efficientnet-$VER \
       --output_file ../pretrained_pytorch/efficientnet-$VER.pth

  mv ../pretrained_pytorch/efficientnet-$VER.pth ../pretrained_pytorch/efficientnet-$VER-$(sha256sum ../pretrained_pytorch/efficientnet-$VER.pth | head -c 8).pth

done
