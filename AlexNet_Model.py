
import numpy as np
# import cv2 as cv
import tensorflow as tf


# InputPath = "D:\Software\Anaconda(Python3.6)\test1\TestImage\"
# ImgName = "000018.jpg"
# ImgInput = InputPath + ImgName

# hello = tf.constant('hello,tf')
# sess = tf.compat.v1.Session()
# print(sess.run(hello))

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

###   Convolution   group split in channel, groups = 2##
def ConvLayer(x, kernel_height, kernel_width, strideX, strideY, kernel_num, name, padding = "SAME", groups = 1):
    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1],
                                     padding = padding)
    ###   define weights and bias in split layer   ###
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable("weights", shape= [kernel_height, kernel_width, channel/groups, kernel_num])
        bias = tf.get_variable("bias", shape=[kernel_num])
        xNew = tf.split(value= x, num_or_size_splits= groups, axis = 3)
        wNew = tf.split(value= weights, num_or_size_splits= groups, axis = 3)

        NewFeatureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        MergeFeatureMap = tf.concat(axis = 3, values = NewFeatureMap)
        Output_addbias = tf.nn.bias_add(MergeFeatureMap, bias)
        relu = tf.nn.relu(tf.reshape(Output_addbias, MergeFeatureMap.get_shape().as_list()), name = scope.name)

        return relu

###    MaxPoolingLayer   ###
def MaxPoolLayer(x, kernel_height, kernel_width, strideX, strideY, padding, name):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1],
                          strides=[1, strideX, strideY, 1],
                          padding=padding, name=name)

###   Dropout  prob = 0.5 ###
def Dropout(x, keep_prob, name = None):
    return tf.nn.dropout(x, keep_prob, name = name)

###   LRNLayer   ###
def LRN(x, radius, alpha, beta, name, bias = 1.0):
    return tf.nn.local_response_normalization(x, depth_radius= radius,
                                              alpha = alpha, beta = beta,
                                              bias = bias, name = name)

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
class alexnet(object):
    def __init__(self, x, keep_prob, classNum, skip, model_path = 'DEFAULT'):
        self.X = x
        self.KEEPPRO = keep_prob
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = model_path
        self.buildCNN()
        if model_path == 'DEFAULT':
            self.MODELPATH = 'bvlc_alexnet.npy'
        else:
            self.MODELPATH = model_path

    def buildCNN(self):
        ### 1st Conv Layer InputSize (227*227*3)###
        conv1 = ConvLayer(self.X, 11, 11, 4, 4, 96, name = 'conv1', padding = 'VALID')
        ###   OutPut (227 - 11)/4 + 1 = 55; 55*55*96   ###
        lrn1 = LRN(conv1, 2, 2e-5, 0.75, name= 'lrn1')
        pool1 = MaxPoolLayer(lrn1, 3, 3, 2, 2, name = 'pool1', padding='VALID')
        ###   Output (55-3)/2 + 1 = 27, 27*27*96   ###

        ###   pad 2
        conv2 = ConvLayer(pool1, 5, 5, 1, 1, 256, name= 'conv2', groups=2)
        ###   Output (27-5+2*2)/2 + 1 = 27, 27*27*256
        lrn2 = LRN(conv2, 2, 2e-5, 0.75, name='lrn2')
        pool2 = MaxPoolLayer(lrn2, 3, 3, 2, 2, name = 'pool2', padding = 'VALID')
        ###   Output (27-3)/2+1 = 13, 13*13*256

        ###   pad 1
        conv3 = ConvLayer(pool2, 3, 3, 1, 1, 384, name='conv3')
        ### Output (13+2*1-3)/1 +1 =13, 13*13*384
        conv4 = ConvLayer(conv3, 3, 3, 1, 1, 384, name='conv4', groups=2)
        ###   Output (13+2*1-3)/1 +1 =13 (Pad 1), 13*13*384
        ###   Pad 1
        conv5 = ConvLayer(conv4, 3, 3, 1, 1, 256, name='conv5', groups=2)
        ###   Output (13+2*1-3)/1 +1 = 13, 13*13*256
        pool5 = MaxPoolLayer(conv5, 3, 3, 2, 2, name='pool5', padding='VALID')
        ###   Output (13 -3)/2+1 =6, 6*6*256

        ###   FC Layer   ###
        flattened = tf.reshape(pool5, [-1, 6*6*256]) ### -1 : auto-calculate
        fc6 =Fclayer(flattened, 6*6*256, 4096, name='fc6', relu=True)
        dropout6 = Dropout(fc6, self.KEEPPRO)

        fc7 = Fclayer(dropout6, 4096, 4096, name='fc7', relu=True)
        dropout7 = Dropout(fc7, self.KEEPPRO)

        self.fc8 = Fclayer(dropout7, 4096, self.CLASSNUM, name='fc8', relu=True)

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