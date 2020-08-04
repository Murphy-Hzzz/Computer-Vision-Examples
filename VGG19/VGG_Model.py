
import tensorflow as tf
import numpy as np

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

###   Convolution Layer   ###
def ConvLayer(x, kernel_height, kernel_width, strideX, strideY, kernel_num, name, padding = "SAME"):
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[kernel_height, kernel_width, channel, kernel_num])
        bias = tf.get_variable('bias', shape=[kernel_num])
        featureMap = tf.nn.conv2d(x, weights, strides= [1, strideY, strideX, 1], padding=padding)
        out = tf.nn.bias_add(featureMap, bias)

        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name = scope.name)

###    MaxPoolingLayer   ###
def MaxPoolLayer(x, kernel_height, kernel_width, strideX, strideY, name, padding = 'SAME'):
        return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1],
                          strides=[1, strideX, strideY, 1],
                          padding=padding, name=name)

###   Dropout  prob = 0.5 ###
def Dropout(x, keep_prob, name = None):
    return tf.nn.dropout(x, keep_prob, name = name)
###   FcLayer   ###
def Fclayer(x, input_fc, output_fc, name, relu = True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable("weights", shape=[input_fc, output_fc], dtype="float")
        bias = tf.get_variable("bias", [output_fc], dtype = "float")
        output_fc = tf.nn.xw_plus_b(x, weights, bias, name = scope.name)
        if relu:
            relu = tf.nn.relu(output_fc)
            return relu
        else:
            return output_fc

###   Define Module   ###
class VGG19 (object):
    def __init__(self, x, keep_prob, classNum, skip, model_path = 'DEFAULT'):
        self.X = x
        self.KEEPPRO = keep_prob
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = model_path
        if model_path == 'DEFAULT':
            self.MODELPATH = 'vgg19.npy'
        else:
            self.MODELPATH = model_path
        self.buildCNN()

    def buildCNN(self):
    ###   InputSize 224*224   ###
        ###   1st: 2 Conv + 1 MaxPool; Conv3-64   ###
        conv1_1 = ConvLayer(self.X, 3, 3, 1, 1, 64, name='conv1_1')
        conv1_2 = ConvLayer(conv1_1, 3, 3, 1, 1, 64, name='conv1_2')
        pool1   = MaxPoolLayer(conv1_2, 2, 2, 2, 2, name='pool1')

        ###   2nd: 2 Conv + 1 MaxPool; Conv3-128   ###
        conv2_1 = ConvLayer(pool1, 3, 3, 1, 1, 128, name='conv2_1')
        conv2_2 = ConvLayer(conv2_1, 3, 3, 1, 1, 128, name='conv2_2')
        pool2   = MaxPoolLayer(conv2_2, 2, 2, 2, 2, name='pool2')

        ###   3rd: 4 ConV + 1 MaxPool; Conv3-256   ###
        conv3_1 = ConvLayer(pool2, 3, 3, 1, 1, 256, name='conv3_1')
        conv3_2 = ConvLayer(conv3_1, 3, 3, 1, 1, 256, name='conv3_2')
        conv3_3 = ConvLayer(conv3_2, 3, 3, 1, 1, 256, name='conv3_3')
        conv3_4 = ConvLayer(conv3_3, 3, 3, 1, 1, 256, name='conv3_4')
        pool3   = MaxPoolLayer(conv3_4, 2, 2, 2, 2, name='pool3')

        ###   4th: 4 ConV + 1 MaxPool; Conv3-512   ###
    ###   the num of channel is not increase any more   ###
        conv4_1 = ConvLayer(pool3, 3, 3, 1, 1, 512, name='conv4_1')
        conv4_2 = ConvLayer(conv4_1, 3, 3, 1, 1, 512, name='conv4_2')
        conv4_3 = ConvLayer(conv4_2, 3, 3, 1, 1, 512, name='conv4_3')
        conv4_4 = ConvLayer(conv4_3, 3, 3, 1, 1, 512, name='conv4_4')
        pool4   = MaxPoolLayer(conv4_4, 2, 2, 2, 2, name='pool4')

        ###   5th: 4 ConV + 1 MaxPool; Conv3-512   ###
        conv5_1 = ConvLayer(pool4, 3, 3, 1, 1, 512, name='conv5_1')
        conv5_2 = ConvLayer(conv5_1, 3, 3, 1, 1, 512, name='conv5_2')
        conv5_3 = ConvLayer(conv5_2, 3, 3, 1, 1, 512, name='conv5_3')
        conv5_4 = ConvLayer(conv5_3, 3, 3, 1, 1, 512, name='conv5_4')
        pool5   = MaxPoolLayer(conv5_4, 2, 2, 2, 2, name='pool5')

        ###   Fc-4096 7*7, Fc-4096 1*1, Fc-1000 1*1, Softmax  ###
        flattened = tf.reshape(pool5, [-1, 7*7*512])
        fc6       = Fclayer(flattened, 7*7*512, 4096, relu=True, name='fc6')
        dropout1  = Dropout(fc6, self.KEEPPRO)

        fc7       = Fclayer(dropout1, 4096, 4096, relu=True, name='fc7')
        dropout2  = Dropout(fc7, self.KEEPPRO)

        self.fc8  = Fclayer(dropout2, 4096, self.CLASSNUM, relu=True, name='fc8')

    ###   Load ModelFile   ###
    def LoadModel(self, sess):
        weights_dict = np.load(self.MODELPATH, encoding="bytes").item()
        for name in weights_dict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse=True):
                    for data in weights_dict[name]:
                        if len(data.shape) == 1:
                            #bias
                            sess.run(tf.get_variable('bias', trainable=False).assign(data))
                        else:

                            #weights
                            sess.run(tf.get_variable('weights', trainable=False).assign(data))






