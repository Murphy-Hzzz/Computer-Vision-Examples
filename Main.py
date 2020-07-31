
import os
import urllib.request
import argparse
import sys
import AlexNet_Model
import cv2 as cv
import tensorflow as tf
import numpy as np
import caffe_classes

parser = argparse.ArgumentParser(description='Classify Input Images.')
parser.add_argument('-m', '--mode', choices=['folder', 'url'], default='folder')
parser.add_argument('-p', '--path',  default = 'testModel')
args = parser.parse_args(sys.argv[1:])
###   Pycharm Configurations


if args.mode == 'folder':

    withPath = lambda f: '{}/{}'.format(args.path, f)
    # Get Images in withPath，==> Matrix
    testImg = dict((f, cv.imread(withPath(f))) for f in os.listdir(args.path) if os.path.isfile(withPath(f)))
elif args.mode == 'url':
    def url2img(url):
        '''url to image'''
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype='uint8')
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        return image


    testImg = {args.path: url2img(args.path)}


if testImg.values():
    #some params
    dropoutPro = 1
    classNum = 1000
    skip = []


    imgMean = np.array([104, 117, 124], np.float)
    x = tf.placeholder("float", [1, 227, 227, 3])

    model = AlexNet_Model.alexnet(x, dropoutPro, classNum, skip)
    score = model.fc8
    softmax = tf.nn.softmax(score)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.LoadModel(sess)

        for key,img in testImg.items():
            #img preprocess
            resized = cv.resize(img.astype(np.float), (227, 227)) - imgMean
            maxx = np.argmax(sess.run(softmax, feed_dict = {x: resized.reshape((1, 227, 227, 3))}))
            res = caffe_classes.class_names[maxx]

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 255, 0), 2)
            print("{}: {}\n----".format(key,res))
            cv.imshow("demo", img)
            cv.waitKey(0)



