#!/usr/bin/env bash

mkdir original_tf
cd original_tf
touch __init__.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/efficientnet_builder.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/efficientnet_model.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/eval_ckpt_main.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/utils.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/preprocessing.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/model_builder_factory.py

mkdir condconv
touch condconv/__init__.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/condconv/condconv_layers.py -O condconv/condconv_layers.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/condconv/efficientnet_condconv_builder.py -O condconv/efficientnet_condconv_builder.py


sed -i "s/^from tensorflow\.python\.tpu/\#from tensorflow\.python\.tpu/" utils.py
sed -i "s/^from condconv import efficientnet_condconv_builder/\#from condconv import efficientnet_condconv_builder/" model_builder_factory.py
sed -i "s/^from edgetpu import efficientnet_edgetpu_builder/\#from edgetpu import efficientnet_edgetpu_builder/" model_builder_factory.py
sed -i "s/^from tpu import efficientnet_tpu_builder/\#from tpu import efficientnet_tpu_builder/" model_builder_factory.py

mkdir lite
touch lite/__init__.py
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/lite/efficientnet_lite_builder.py -O lite/efficientnet_lite_builder.py

cd ..
mkdir -p tmp
