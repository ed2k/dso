import numpy as np
import cv2
import tensorflow as tf
from utils import label_map_util
#from utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    (im_height, im_width, d) = image.shape
    return np.array(image).reshape(
           (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_CKPT = '../ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
PATH_TO_LABELS = '../ssd_mobilenet_v1_coco_11_06_2017/graph.pbtxt'
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


cap = cv2.VideoCapture(0)
r, image = cap.read()
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict = {image_tensor: image_np_expanded}
    )
    print boxes, classes, scores
