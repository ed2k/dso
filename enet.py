#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script visualize the semantic segmentation of ENet.
"""
import numpy as np
import sys
caffe_root = '../ENet/caffe-enet/'  # Change this to the absolute directory to ENet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2

def predict(net, input_image):
    input_image = input_image.astype(np.float32)
    inshape = _input_shape
    input_image = cv2.resize(input_image, (inshape[3], inshape[2]))
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.asarray([input_image])

    net.forward_all(**{net.inputs[0]: input_image})

    prediction = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)
    print(prediction.shape)
    prediction = np.squeeze(prediction)
    print(prediction.shape)
    prediction = np.resize(prediction, (3, inshape[2], inshape[3]))
    print(prediction.shape)
    prediction = prediction.transpose(1, 2, 0).astype(np.uint8)
    print(prediction.shape)

    prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
    label_colours_bgr = _label_colours[..., ::-1]
    #print(label_colours.shape, label_colours_bgr.shape)
    cv2.LUT(prediction, label_colours_bgr, prediction_rgb)
    return prediction_rgb

_model = '../ENet/prototxts/enet_deploy_final.prototxt'
_weights = '../ENet/enet_weights_zoo/cityscapes_weights.caffemodel'
_colors = '../ENet/scripts/cityscapes19.png'

def create_default():
    global _input_shape
    global _label_colours
    caffe.init_log(1)
    caffe.set_mode_cpu()
    net = caffe.Net(_model, caffe.TEST, weights=_weights)
    _input_shape = net.blobs['data'].data.shape
    _label_colours = cv2.imread(_colors, 1).astype(np.uint8)
    return net
if __name__ == '__main__':
    net = create_default()

    cnt = 1
    while cnt < 10000:
      img = '%s/%05d.jpg' % (sys.argv[1], cnt)
      print(img)
      img = cv2.imread(img, 1)
      prediction_rgb = predict(net, img)
      cv2.imshow("ENet", prediction_rgb)
      cnt += 1
      key = cv2.waitKey(1)







