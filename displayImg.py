import cv2
import os

class myreader:
  def __init__(m, f=''):
    if f == '':
      m.handle = cv2.VideoCapture(0);
      m.idx = -1;
    elif f.endswith('mp4'):
      m.handle = cv2.VideoCapture(f)
      m.idx = -1;
    elif os.path.isdir(f):
      m.idx = 3001;
      m.path = f;
  def isOpened(m):
    if m.idx == -1: return m.handle.isOpened()
    p = os.path.join(m.path,str(m.idx)+'.jpg')
    return os.path.exists(p)
  def read(m):
    if m.idx == -1: return m.handle.read()
    p = os.path.join(m.path, str(m.idx)+'.jpg')
    m.idx+=1
    print(p)
    img = cv2.imread(p)
    return 1,img
  def release(m):
    if m.idx == -1: m.handle.release()

def down_scale_img(img, new_width):
    h,w,c = img.shape
    return cv2.resize(img, (new_width, h*new_width/w))

def overlay_img(dst_img,src_img, x,y):
    h,w,c = src_img.shape
    dst_img[y:y+h,x:x+w] = src_img

import sys
if len(sys.argv) < 2:
  cap = myreader('/home/a/SEQ_0/v.mp4')
else:
  cap = myreader(sys.argv[1])

cnt = 1
while(cap.isOpened()):
    ret, f = cap.read()
    if f is None : break
    print (f.shape)
    frame = down_scale_img(f,640)
    f2 = down_scale_img(frame, 160)
    f3 = down_scale_img(f2, 80)
    f4 = down_scale_img(f3, 40)
    overlay_img(frame, f2, 0,0)
    overlay_img(frame, f3, frame.shape[1]-f3.shape[1],0)
    overlay_img(frame, f4, frame.shape[1]-f4.shape[1]-f3.shape[1],0)
    cv2.imshow('f', frame)
    cnt += 1
    key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

