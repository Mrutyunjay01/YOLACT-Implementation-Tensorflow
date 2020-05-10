import tensorflow as tf

from layers.featured_pyramid_network import FeaturePyramidNeck
from layers.prediction_head import PredictionHead
from layers.protonet import ProtoNet
from layers.yolov3 import make_yolov3_model, WeightReader
from utils.create_prior import make_priors

assert tf.__version__.startswith('2')


class YolactYolo(tf.keras.Model):
    """
        Creating the YOLCAT Architecture
        Arguments:
    """

    def __init__(self, input_size, fpn_channels, feature_map_size, num_class, num_mask, aspect_ratio, scales):
        super(YolactYolo, self).__init__()
        base_model = make_yolov3_model()
        weight_reader = WeightReader('/content/yolov3.weights')  # path to yolov3 weights
        weight_reader.load_weights(base_model)

        self.backbone_yolo = tf.python.keras.Model(input=base_model.input,
                                                   output=base_model.output)

        self.backbone_fpn = FeaturePyramidNeck(fpn_channels)
        self.protonet = ProtoNet(num_mask)

        # semantic segmentation branch to boost feature richness
        self.semantic_segmentation = tf.keras.layers.Conv2D(num_class, (1, 1), 1, padding="same",
                                                            kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.num_anchor, self.priors = make_priors(input_size, feature_map_size, aspect_ratio, scales)
        print("prior shape:", self.priors.shape)
        print("num anchor per feature map: ", self.num_anchor)

        # shared prediction head
        self.predictionHead = PredictionHead(256, len(aspect_ratio), num_class, num_mask)

    def set_bn(self, mode='train'):
        if mode == 'train':
            for layer in self.backbone_yolo.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
        else:
            for layer in self.backbone_yolo.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True

    def call(self, inputs, training=None, mask=None):
        # backbone(yolo + FPN)
        c3, c4, c5 = self.backbone_yolo(inputs)
        # print("c3: ", c3.shape)
        # print("c4: ", c4.shape)
        # print("c5: ", c5.shape)
        fpn_out = self.backbone_fpn(c3, c4, c5)

        # Protonet branch
        p3 = fpn_out[0]
        protonet_out = self.protonet(p3)
        # print("protonet: ", protonet_out.shape)

        # semantic segmentation branch
        seg = self.semantic_segmentation(p3)

        # Prediction Head branch
        pred_cls = []
        pred_offset = []
        pred_mask_coef = []

        # all output from FPN use same prediction head
        for f_map in fpn_out:
            cls, offset, coef = self.predictionHead(f_map)
            pred_cls.append(cls)
            pred_offset.append(offset)
            pred_mask_coef.append(coef)

        pred_cls = tf.concat(pred_cls, axis=1)
        pred_offset = tf.concat(pred_offset, axis=1)
        pred_mask_coef = tf.concat(pred_mask_coef, axis=1)

        pred = {
            'pred_cls': pred_cls,
            'pred_offset': pred_offset,
            'pred_mask_coef': pred_mask_coef,
            'proto_out': protonet_out,
            'seg': seg
        }

        return pred
