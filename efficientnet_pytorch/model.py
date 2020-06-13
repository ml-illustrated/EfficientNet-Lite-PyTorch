"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import torch
from torch import nn
from torch.nn import functional as F
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1,1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        if global_params.relu_fn == 'relu6':
            self._swish = nn.ReLU6()
        else:
            self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.
    
    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        >>> import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._fix_head_stem = global_params.fix_head_stem

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        self.input_image_size = global_params.image_size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params, self._fix_head_stem)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for i, block_args in enumerate( self._blocks_args ):

            # Update block input and output filters based on depth multiplier.
            input_filters=round_filters(block_args.input_filters, self._global_params)
            output_filters=round_filters(block_args.output_filters, self._global_params)

            kernel_size = block_args.kernel_size
            if self._fix_head_stem and (i == 0 or i == len(self._blocks_args) - 1):
                # print( 'self._fix_head_stem: ', block_args.num_repeat, self._fix_head_stem, i, len(self._blocks_args ) )
                repeats = block_args.num_repeat
                if self._fix_head_stem and i == 0:
                    input_filters = 32 # override when head stem is fixed??
            else:
                repeats = round_repeats(block_args.num_repeat, self._global_params)
            
            block_args = block_args._replace(
                input_filters = input_filters,
                output_filters = output_filters,
                num_repeat = repeats,
            )

            # print( 'block %d args: %s' % ( i, block_args ) )
            
            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1: # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

            # print( 'img_size: ', image_size )

        self.feature_output_image_size = image_size
        # print( 'feature_output_image_size: ', self.feature_output_image_size )

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params, self._fix_head_stem)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        if global_params.relu_fn == 'relu6':
            self._swish = nn.ReLU6()
        else:
            self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.

        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution 
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
        
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        bs = inputs.size(0)

        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        if self._global_params.local_pooling:
            # shape = x.size() # lite1 is (1, 7, 7, 1280)
            # original
            # kernel_size = [1, shape[2], shape[3], 1]
            # outputs = tf.nn.avg_pool(outputs, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
            # -> (?, 1, 1, 1280)
            # outputs = tf.squeeze(outputs, self._spatial_dims)
            # -> (?, 1280)
            
            kernel_size = self.feature_output_image_size
            print( 'local_pooling: ', kernel_size, x.shape )
            
            # x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=0)
            x = x.view(bs, 1280, -1)
            print( 'view: ', x.shape )
            x = x.mean(-1)
            print( 'mean: ', x.shape )

            x = self._dropout(x)
            x = x.view(bs, -1)
            # print( 'reshaped: ', x.shape )
            x = self._fc(x)
        else:
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            x = self._dropout(x)
            x = self._fc(x)

        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params): 
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False, 
                        in_channels=3, num_classes=1000, **override_params):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str): 
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool): 
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int): 
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params): 
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes = num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name. 

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]

        # Support the construction of 'efficientnet-l2' without pretrained weights
        valid_models += ['efficientnet-l2']

        valid_models += ['efficientnet-lite'+str(i) for i in range(5)]
        
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
    
    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

    def gen_torch_output( self, sample_input ):
        self.eval()
        with torch.no_grad():
            torch_output = self( sample_input )
            torch_output = torch_output.cpu().detach().numpy()
        return torch_output

    def convert_to_onnx( self, filename_onnx, sample_input ):

        input_names = [ self.input_name ]
        output_names = [ self.output_name ]
        
        torch.onnx.export(
            self,
            sample_input,
            filename_onnx,
            input_names=input_names,
            output_names=output_names,
            # operator_export_type=OperatorExportTypes.ONNX
        )

    def convert_to_coreml( self, fn_mlmodel, sample_input, plot_specs=True ):
        import onnx
        import onnx_coreml

        self.input_name = 'input.1' # hack
        self.output_name = 'classes' # hack
        
        
        torch_output = self.gen_torch_output( sample_input )
        print( 'torch_output: shape %s\nsample %s ' % ( torch_output.shape, torch_output[0,:3] ) )

        # first convert to ONNX
        filename_onnx = '/tmp/efficientnet_model.onnx'
        self.convert_to_onnx( filename_onnx, sample_input )

        # set up for Core ML export
        convert_params = dict(
            predicted_feature_name = [],
            minimum_ios_deployment_target='13',
        )

        mlmodel = onnx_coreml.convert(
            model=filename_onnx,
            **convert_params, 
        )

        '''
        output = spec.description.output[0]
        import coremltools.proto.FeatureTypes_pb2 as ft
        output.type.imageType.colorSpace = ft.ImageFeatureType.GRAYSCALE
        output.type.imageType.height = 300
        output.type.imageType.width = 150
        '''

        assert mlmodel != None, 'CoreML Conversion failed'

        mlmodel.save( fn_mlmodel )

        model_inputs = {
            self.input_name : sample_input
        }
        # do forward pass
        mlmodel_output = mlmodel.predict(model_inputs, useCPUOnly=True)
        
        print( 'mlmodel_output: shape %s\nsample %s ' % ( mlmodel_output.shape, mlmodel_output[:3] ) )

        return torch_output, mlmodel_output

'''
import torch 
from PIL import Image
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-lite1')

image_size = model.input_image_size

# Preprocess image
tfms = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
sample_input = tfms(Image.open('./examples/simple/img.jpg')).unsqueeze(0)

import json
outputs = model.gen_torch_output( sample_input )

#fn_mlmodel = '/tmp/efficientnet-lite0.mlmodel'
#torch_output, mlmodel_output = model.convert_to_coreml( fn_mlmodel, sample_input )


labels_map = json.load(open('./examples/simple/labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

outputs = torch.from_numpy( outputs )
print('-----')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
'''


'''
# bash demo.cmd efficientnet-lite0
  -> top_0 (96.81%): giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca  
  -> top_1 (0.16%): Staffordshire bullterrier, Staffordshire bull terrier  
  -> top_2 (0.09%): Samoyed, Samoyede  
  -> top_3 (0.08%): Arctic fox, white fox, Alopex lagopus  
  -> top_4 (0.07%): ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus  

# bash demo.cmd efficientnet-lite1
  -> top_0 (44.93%): giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca  
  -> top_1 (1.26%): ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus  
  -> top_2 (0.63%): American black bear, black bear, Ursus americanus, Euarctos americanus  
  -> top_3 (0.61%): albatross, mollymawk  
  -> top_4 (0.58%): white wolf, Arctic wolf, Canis lupus tundrarum  

# bash demo.cmd efficientnet-lite2
  -> top_0 (62.82%): giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca  
  -> top_1 (1.04%): ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus  
  -> top_2 (0.39%): Arctic fox, white fox, Alopex lagopus  
  -> top_3 (0.31%): Samoyed, Samoyede  
  -> top_4 (0.31%): lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens  

# bash demo.cmd efficientnet-lite3
  -> top_0 (84.02%): giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca  
  -> top_1 (1.27%): ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus  
  -> top_2 (0.27%): brown bear, bruin, Ursus arctos  
  -> top_3 (0.23%): American black bear, black bear, Ursus americanus, Euarctos americanus  
  -> top_4 (0.19%): lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens  

# bash demo.cmd efficientnet-lite4
  -> top_0 (85.58%): giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca  
  -> top_1 (0.26%): brown bear, bruin, Ursus arctos  
  -> top_2 (0.24%): American black bear, black bear, Ursus americanus, Euarctos americanus  
  -> top_3 (0.16%): lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens  
  -> top_4 (0.15%): ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus  

'''
