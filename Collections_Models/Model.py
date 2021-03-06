
from dataload import *
import tensorflow as tf
from Alexnet import AlexNet
from sklearn.model_selection import train_test_split
import Resnet_50_101_152
from VGG_19 import VGG19
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self, sess, type_of_model, epoch, dataset_name, batch_size, img_size, y_dim, resnet_type):
        self.sess = sess
        self.type_of_model = type_of_model
        self.epoch = epoch
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.img_size = img_size
        self.y_dim = y_dim
        self.resnet_type = resnet_type

        # load dataset
        if self.dataset_name == 'satellite':
            print('loading  satellite .............')
            self.data_X, self.data_Y = load_satetile_image(self.img_size, y_dim=self.y_dim)
            # split the data into train set and valid set
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.data_X, self.data_Y,
                                                                                  shuffle=True, test_size=0.1,
                                                                                  random_state=2019)
            print('self.X_train, self.y_train:',self.X_train.shape, self.y_train.shape, self.X_val.shape, self.y_val.shape)

    def build_model(self):

        # the placeholder of image and label
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, 3], name='img')
        self.keep_prob = tf.placeholder("float")

        # chose the model
        if self.type_of_model == 'Alexnet':
            network = AlexNet(self.inputs, self.y_dim, self.keep_prob)
            score = network.fc3
        elif self.type_of_model == 'ResNet':
            network = Resnet_50_101_152.resnet(self.inputs, self.resnet_type, self.y_dim)
            score = tf.squeeze(network, axis=(1, 2))
        elif self.type_of_model == 'VGG19':
            network = VGG19(self.inputs, self.keep_prob, self.y_dim)
            score = network.fc8

        else:
            print('these is no %s'%self.type_of_model)

        softmax_result = tf.nn.softmax(score)

        # Define Loss Fun, Optimizer
        cross_entropy = -tf.reduce_sum(self.y * tf.log(softmax_result))
        self.Optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # Judge the scores
        correct_prediction = tf.equal(tf.argmax(softmax_result, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def plot_acc(self, train_acc, val_acc):
        fig, ax = plt.subplots()
        plt.xlabel('Iter')
        plt.ylabel('Accuracy')

        x = [i for i in range(len(train_acc))]
        plt.plot(x, train_acc, "x-", label="Train Accuracy")
        plt.plot(x, val_acc, "+-", label="Valid Accuracy")

        """open the grid"""
        plt.grid(True)

        plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

        plt.show()

    def train(self):
        print('begin training ...........')
        tf.global_variables_initializer().run()
        train_acc_list = []
        valid_acc_list = []

        for k in range(self.epoch):
            for i in range(len(self.X_train) // self.batch_size):
                if (i+1) == (len(self.X_train) // self.batch_size):
                    print('Reload Imaging')
                    self.data_X, self.data_Y = load_satetile_image(self.img_size, y_dim=self.y_dim)
                    # split the data into train set and valid set
                    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.data_X, self.data_Y, shuffle=True, test_size=0.1,
                                                                      random_state=2019)

                dropout_rate = 0.5
                _, train_acc = self.sess.run([self.Optimizer, self.accuracy], feed_dict={self.inputs: self.X_train[i*self.batch_size:(i+1)*self.batch_size],
                                     self.y: self.y_train[i*self.batch_size:(i+1)*self.batch_size], self.keep_prob: dropout_rate})


                all_val_acc = 0
                for j in range((len(self.X_val) // self.batch_size) - 1):
                    dropout_rate = 0
                    val_acc = self.sess.run(self.accuracy, feed_dict={
                        self.inputs: self.X_val[j * self.batch_size:(j + 1) * self.batch_size],
                        self.y: self.y_val[j * self.batch_size:(j + 1) * self.batch_size],
                        self.keep_prob: dropout_rate})
                    all_val_acc += val_acc

                print("step %d, train accuracy %f valid accuracy %f:" % (i,train_acc, all_val_acc / (j + 1)))
                train_acc_list.append(train_acc)
                valid_acc_list.append(all_val_acc)
        self.plot_acc(train_acc_list, valid_acc_list)










