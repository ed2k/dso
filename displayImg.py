import cv2, numpy as np
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

def test_diff(prev_img, img):
    return cv2.absdiff(prev_img,img)
def test_canny(img):
    return cv2.Canny(img,100,200)

import sys
if len(sys.argv) < 2:
  cap = myreader('/home/a/SEQ_0/v.mp4')
else:
  cap = myreader(sys.argv[1])

cnt = 1
prevImg = None
while(cap.isOpened()):
    ret, f = cap.read()
    if f is None : break
    #print (f.shape)
    frame = down_scale_img(f,640)
    grey = down_scale_img(f,640)
    f2 = down_scale_img(frame, 160)
    f3 = down_scale_img(f2, 80)
    f4 = down_scale_img(f3, 40)
    if cnt > 1:
        frame = test_diff(prevImg, frame)
        #dup2 = test_canny(frame)
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #_, grey = cv2.threshold(grey,20,255,cv2.THRESH_TOZERO)
        #_,cnts,_ = cv2.findContours(grey.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame, [c],0,(0,255,0), 1)
        #grey = np.minimum(grey+frame,255)
    prevImg = down_scale_img(f, 640)
    overlay_img(frame, f2, 0,0)
    overlay_img(frame, f3, frame.shape[1]-f3.shape[1],0)
    overlay_img(frame, f4, frame.shape[1]-f4.shape[1]-f3.shape[1],0)
    cv2.imshow('f', grey)
    cnt += 1
    key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

