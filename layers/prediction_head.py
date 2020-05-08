# import required packages
import tensorflow as tf


class PredictionHead(tf.keras.layers.Layer):
    """
    :argument:
    1. out_channels :
    2. num_anchors :
    3. num_classes :
    4. num_masks :
    """

    def __init__(self, out_channels, num_anchors, num_classes, num_masks, **kwargs):
        super(PredictionHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_masks = num_masks
        self.output_channels = out_channels

        # construct our first conv layer : W * H * 256
        self.Conv1 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding='same',
                                            kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.Conv2 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding='same',
                                            kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.classConv = tf.keras.layers.Conv2D(self.num_classes * self.num_anchors, (3, 3), 1, padding='same',
                                                kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.boxesConv = tf.keras.layers.Conv2D(4 * self.num_anchors, (3, 3), 1, padding='same',
                                                kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.maskConv = tf.keras.layers.Conv2D(self.num_masks * self.num_anchors, (3, 3), 1, padding='same',
                                               kernel_initializer=tf.keras.initializers.glorot_uniform())

    def call(self, P):
        """
        (W * H * 256) * 2
        :param P: layer from our FPN layers P3 to P7
        :return: pred_classes, pred_boxes, pred_masks
        """
        P = self.Conv1(P)
        P = self.Conv2(P)

        pred_class = self.classConv(P)
        pred_box = self.boxesConv(P)
        pred_mask = self.maskConv(P)

        # reshape the prediction head result for calculation of
        # classification loss, box regression loss, Mask loss(Pixel-wise binary cross-entropy loss)
        # between assembled mask and the ground truth masks
        pred_class = tf.reshape(pred_class, [pred_class.shape[0], -1, self.num_classes])
        pred_box = tf.reshape(pred_box, [pred_box.shape[0], -1, 4])
        pred_mask = tf.reshape(pred_mask, [pred_mask.shape[0], -1, self.num_masks])

        return pred_class, pred_box, pred_mask