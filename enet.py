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



def predict(net, input_image, label_colours, input_shape):
    input_image = input_image.astype(np.float32)

    input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.asarray([input_image])

    net.forward_all(**{net.inputs[0]: input_image})

    prediction = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)

    prediction = np.squeeze(prediction)
    prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
    prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

    prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
    label_colours_bgr = label_colours[..., ::-1]
    print(label_colours.shape, label_colours_bgr.shape)
    cv2.LUT(prediction, label_colours_bgr, prediction_rgb)
    return prediction_rgb

_model = '../ENet/prototxts/enet_deploy_final.prototxt'
_weights = '../ENet/enet_weights_zoo/cityscapes_weights.caffemodel'
_colors = '../ENet/scripts/cityscapes19.png'

def create_default():
    caffe.set_mode_cpu()
    return caffe.Net(_model, _weights, caffe.TEST)
if __name__ == '__main__':
    net = create_default()

    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['deconv6_0_0'].data.shape
    print(input_shape, output_shape)
    label_colours = cv2.imread(_colors, 1).astype(np.uint8)
    cnt = 1
    while cnt < 10000:
      img = '%s/%05d.jpg' % (sys.argv[1], cnt)
      print(img)
      img = cv2.imread(img, 1)
      prediction_rgb = predict(net, img, label_colours, input_shape)
      cv2.imshow("ENet", prediction_rgb)
      cnt += 1
      key = cv2.waitKey(1)







