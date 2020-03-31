# import required packages
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
import tensorflow_addons as tfa
import pickle
from collections import OrderedDict


class BottleNeck(L.Layer):
    expansion = 4

    def __init__(self, output_channels, stride=1, downsample=None, norm_layer=BatchNormalization,
                 dilation_rate=1, use_dcn=False):
        super(BottleNeck, self).__init__()
        self.conv1 = L.Conv2D(output_channels, (1, 1), use_bias=False, dilation_rate=dilation_rate)
        self.bn1 = norm_layer(axis=-1)

        self.conv2 = L.Conv2D(output_channels, (3, 3), strides=stride, padding='valid', use_bias=False,
                              dilation_rate=dilation_rate)
        self.bn2 = norm_layer(axis=-1)
        self.conv3 = L.Conv2D(4 * output_channels, (3, 3), strides=stride,
                              padding='valid', use_bias=False, dilation_rate=dilation_rate)
        self.bn3 = norm_layer(axis=-1)
        self.activation = Activation('relu')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)

        return out


class ResNetBackbone(L.Layer):
    def __init__(self, layers, dcn_layers=None, dcn_interval=1, atrous_layers=None, block=BottleNeck,
                 norm_layer=BatchNormalization):
        super(ResNetBackbone, self).__init__()

        if atrous_layers is None:
            atrous_layers = []
        if dcn_layers is None:
            dcn_layers = [0, 0, 0, 0]
        self.num_base_layers = len(layers)
        self.layers = []
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers

        self.input_channels = 64
        self.conv1 = L.Conv2D(64, (7, 7), strides=2, padding=3, use_bias=False)
        self.bn1 = BatchNormalization(axis=-1)
        self.activation = Activation('relu')
        self.maxpool = L.MaxPooling2D((3, 3), strides=2, padding='valid')

        self._make_layer(block, 64, layers[0], dcn_layers=dcn_layers[0], dcn_interval=dcn_interval)
        self._make_layer(block, 128, layers[1], dcn_layers=dcn_layers[1], dcn_interval=dcn_interval)
        self._make_layer(block, 256, layers[2], dcn_layers=dcn_layers[2], dcn_interval=dcn_interval)
        self._make_layer(block, 512, layers[3], dcn_layers=dcn_layers[3], dcn_interval=dcn_interval)
        self.backbone_modules = [m for m in self.submodules if isinstance(m, L.Conv2D)]

    def _make_layer(self, block, out_channels, blocks, stride, dcn_layers=0, dcn_interval=1):
        downsample = None
        if stride != 1 or self.input_channels != out_channels * block.expansion:
            if len(self.layer) in self.atrous_layers:
                self.dilation += 1
                stride = 1
            downsample = tf.keras.models.Sequential(
                L.Conv2D(out_channels * block.expansion,
                         (1, 1), strides=stride, use_bias=False,
                         dilation_rate=self.dilation),
                self.norm_layer(axis=-1),
            )
            layers = []
            use_dcn = (dcn_layers >= blocks)
            layers.append(block(out_channels, stride, downsample, self.norm_layer, self.dilation, use_dcn))
            self.input_channels = out_channels * block.expansion
            for i in range(1, blocks):
                use_dcn = ((i + dcn_layers) >= blocks) and (i % dcn_interval == 0)
                layers.append(block(out_channels, norm_layer=self.norm_layer, use_dcn=use_dcn))
            layer = tf.keras.models.Sequential(*layers)
            self.channels.append(out_channels * block.expansion)
            self.layers.append(layer)

            return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def init_backbone(self, path):
        """
        initializes the backbone weights for training
        :param path: filepath to weights
        :return: state_dict
        """
        state_dict = tf.keras.models.load_model(path)
        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx - 1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        self.get_config(state_dict)

    def add_layer(self, conv_channels=1024, downsample=None, depth=1, block=BottleNeck):
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)


class ResNetBackboneGN(ResNetBackbone):
    def __init__(self, layers, num_groups=32):
        super().__init__(layers, norm_layer=lambda x: tfa.layers.GroupNormalization(num_groups, x))

    def init_backbone(self, path):
        with open(path, 'rb') as f:
            state_dict = pickle.load(f, encoding='latin1')
            state_dict = state_dict['blobs']

        our_state_dict_keys = list(self.state_dict().keys())
        new_state_dict = {}

        gn_trans = lambda x: ('gn_s' if x == 'weight' else 'gn_b')
        layeridx2res = lambda x: 'res' + str(int(x) + 2)
        block2branch = lambda x: 'branch2' + ('a', 'b', 'c')[int(x[-1:]) - 1]

        # transcribe each Detectron weights name to a YOLACT weights name
        for key in our_state_dict_keys:
            parts = key.split('.')
            transcribed_key = ''

            if parts[0] == 'conv1':
                transcribed_key = 'conv1_w'
            elif parts[0] == 'bn1':
                transcribed_key = 'conv1_' + gn_trans(parts[1])
            elif parts[0] == 'layers':
                if int(parts[1]) >= self.num_base_layers:
                    continue

                transcribed_key = layeridx2res(parts[1])
                transcribed_key += '_' + parts[2] + '_'

                if parts[3] == 'downsample':
                    transcribed_key += 'branch1"'

                    if parts[4] == '0':
                        transcribed_key += 'w'
                    else:
                        transcribed_key += gn_trans(parts[5])
                else:
                    transcribed_key += block2branch(parts[3]) + '_'

                    if 'conv' in parts[3]:
                        transcribed_key += 'w'
                    else:
                        transcribed_key += gn_trans(parts[4])
            new_state_dict[key] = tf.convert_to_tensor(state_dict[transcribed_key])
        self.get_config(new_state_dict)


def DarknetConvLayer(out_channels, *args, **kwargs):
    return tf.keras.models.Sequential(tf.keras.layers.Conv2D(out_channels, *args, **kwargs, use_bias=False),
                                      tf.keras.layers.BatchNormalization(axis=-1),
                                      tf.keras.layers.LeakyReLU(0.1))


class DarknetBlock(L.Layer):
    expansion = 2

    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = DarknetConvLayer(out_channels, (1, 1))
        self.conv2 = DarknetConvLayer(out_channels * self.expansion, (3, 3), padding='valid')

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


class DarkNetBackbone(L.Layer):
    def __init__(self, layers=[1, 2, 8, 8, 4], block=DarknetBlock):
        super().__init__()
        self.num_base_layers = len(layers)
        self.layers = []
        self.channels = []
        self._preconv = DarknetConvLayer(32, (3, 3), padding='valid')
        self.input_channels = 32
        self._make_layer(block, 32, layers[0])
        self._make_layer(block, 64, layers[1])
        self._make_layer(block, 128, layers[2])
        self._make_layer(block, 256, layers[3])
        self._make_layer(block, 512, layers[4])
        self.backbone_modules = [m for m in self.layers if isinstance(m, L.Conv2D)]

    def _make_layer(self, block, out_channels, num_blocks, stride=2):
        layer_list = [DarknetConvLayer(out_channels * block.expansion,
                                       (3, 3), padding='valid', stride=stride)]
        self.input_channels = out_channels * block.expansion
        layer_list += [block(out_channels) for _ in range(num_blocks)]

        self.channels.append(self.input_channels)
        self.layers.append(tf.keras.Sequential(*layer_list))

    def forward(self, x):
        x = self._preconv(x)
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return tuple(outs)

    def add_layer(self, conv_channels=1024, stride=2, depth=1, block=DarknetBlock):
        self._make_layer(block, conv_channels // block.expansion, num_blocks=depth, stride=stride)

    def init_backbone(self, path):
        self.get_config(tf.keras.models.load_model(path))


class VGGBackbone(L.Layer):
    """
    :argument:
    -cfg: A list of layers given as lists. Can be either 'M' signifying a max pooling layer,
    a number signifying that many feature maps in a conv layer, or a tuple of 'M' or a number of
    kwargs dict to pass into the func that creates the layer to pass into add_layer.
    - extra_args: A list of lists of arguments to pass into add_layer
    - norm_layers : Layers indices that need to pass through an 12 norm_layer

    """
    def __init__(self, cfg, extra_args=[], norm_layers=[]):
        super().__init__()

        self.channels = []
        self.layers = []
        self.input_channels = 3
        self.extra_args = list(reversed(extra_args))
        self.total_layer_count = 0
        self.state_dict_lookup = {}

        for idx, layer_config in enumerate(cfg):
            self._make_layer(layer_config)

        self.norms =
