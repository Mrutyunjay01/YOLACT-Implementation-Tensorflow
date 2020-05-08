# import required packages
import tensorflow as tf


def _crop_and_add(x1, x2):
    """
    for parameters to concatenate with matched shape
    :param x1: an numpy array
    :param x2: another numpy array
    :return: Added after being matched shape
    """
    x1_shape = x1.shape
    x2_shape = x2.shape
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    x1_crop = tf.slice(x1, offsets, size=x2.shape)
    return tf.add(x1_crop, x2)


class FeaturePyramidNeck(tf.keras.layers.Layer):
    """
    creating backbone components for feature pyramid network
    :argument
    num_fpn_filters
    """

    def __init__(self, num_fpn_filters):
        super(FeaturePyramidNeck, self).__init__()
        self.upSample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        # No relu activation for down-sampled layer
        self.downSample1 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3),
                                                  2, padding='same',
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.downSample2 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 2,
                                                  padding='same',
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralConv1 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding='same',
                                                   kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralConv2 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding='same',
                                                   kernel_initializer=tf.keras.initializers.glorot_uniform())
        self.lateralConv3 = tf.keras.layers.Conv2D(num_fpn_filters, (1, 1), 1, padding='same',
                                                   kernel_initializer=tf.keras.initializers.glorot_uniform())
        # predict layer for FPN
        # conv layer followed by a relu activation layer
        self.predictP5 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding='same',
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                activation='relu')
        self.predictP4 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding='same',
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                activation='relu')
        self.predictP3 = tf.keras.layers.Conv2D(num_fpn_filters, (3, 3), 1, padding='same',
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                activation='relu')

    # Now call the call function and construct the Whole FPN Layer

    def call(self, C3, C4, C5):
        """

        :param C3: 3rd conv layer from backbone network
        :param C4: 4th conv layer from backbone network
        :param C5: 5th conv layer from backbone network
        :return: returns FPN layer with layeres : P3 to P7
        """
        # lateral conv for C3 to C5
        P5 = self.lateralConv1(C5)
        P4 = _crop_and_add(self.upSample(P5), self.lateralConv2(C4))
        P3 = _crop_and_add(self.upSample(P4), self.lateralConv3(C3))

        # smooth prediction layer for P3 to P5
        P3 = self.predictP3(P3)
        P4 = self.predictP4(P4)
        P5 = self.predictP5(P5)

        # downsample conv to get P6 and P7
        P6 = self.downSample1(P5)
        P7 = self.downSample2(P6)

        return [P3, P4, P5, P6, P7]
# FPN layer done and dusted. @15-03-2020 12:38 AM