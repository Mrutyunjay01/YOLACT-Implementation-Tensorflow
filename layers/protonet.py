# import required packages
import tensorflow as tf


class ProtoNet(tf.keras.layers.Layer):
    """
    creating the component of Protonet
    :argument: num_prototype
    """
    def __init__(self, num_prototype):
        super(ProtoNet, self).__init__()
        self.Conv1 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding='same',
                                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                            activation='relu')
        self.Conv2 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding='same',
                                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                            activation='relu')
        self.Conv3 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding='same',
                                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                            activation='relu')
        self.UpSampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.Conv4 = tf.keras.layers.Conv2D(256, (3, 3), 1, padding='same',
                                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                            activation='relu')
        self.Final = tf.keras.layers.Conv2D(num_prototype, (1, 1), 1, padding='same',
                                            kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                            activation='relu')

    def call(self, P3):
        # 3 * (3, 3) 256 kernels followed by relu activation
        proto = self.Conv1(P3)
        proto = self.Conv2(proto)
        proto = self.Conv3(proto)

        # upsampling + convolution
        proto = self.UpSampling(proto)
        proto = self.Conv4(proto)

        # final convolution
        proto = self.Final(proto)

        return proto
# Protonet done and dusted!
