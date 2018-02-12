import cv2, numpy as np
import os

class myreader:
  def __init__(m, f=''):
    if f == '':
      m.handle = cv2.VideoCapture(0);
      m.idx = -1;
    elif f.endswith('.mp4') or f.endswith('.webm'):
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
    h,w,_ = img.shape
    return cv2.resize(img, (new_width, h*new_width/w))

def overlay_img(dst_img,src_img, x,y):
    h,w,c = src_img.shape
    dst_img[y:y+h,x:x+w] = src_img

def test_diff(prev_img, img):
    return cv2.absdiff(prev_img,img)
def test_canny(img):
    return cv2.Canny(img,100,200)

def get_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))

def find_lane(img):
    h,w,_ = img.shape
    offx = int(w/4)
    offy = int(h/2)
    grey = cv2.cvtColor(img[-h/2:,:w/2], cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(grey, 20,255, cv2.THRESH_TOZERO)
    mask = np.zeros((h/2,w/2), np.uint8)
    pts = [(0,h/4),(0,h/2),(w/4,h/2),(w/2-w/16,0),(w/2-w/8,0)]
    cv2.fillConvexPoly(mask, np.int32(pts), 1.0)
    thr = mask * thr
    _,cnts,_ = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thr = np.zeros((h/2,w/2), np.uint8)
    for cnt in cnts:
        if len(cnt) < 20: continue
        cv2.drawContours(img, [cnt], -1, (0,255,0), offset=(0,offy))
        cv2.drawContours(thr, [cnt], -1, 255, offset=(0,0))
    #edges = cv2.Canny(grey,50,150,apertureSize = 3)
    lines = cv2.HoughLines(thr,1,np.pi/180,90)
    if lines is None: return
    c = 0
    for line in lines:
        rho,theta = line[0]
        if theta < np.pi*20/180: continue
        if theta > np.pi*70/180: continue
        c += 1
        if c>9: return
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho+(h/2)
        x1 = int(x0 + offy*(-b))
        y1 = int(y0 + offy*(a))
        x2 = int(x0 - offy*(-b))
        y2 = int(y0 - offy*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)


import sys
if len(sys.argv) < 2:
  cap = myreader('/home/a/SEQ_0/v.mp4')
else:
  cap = myreader(sys.argv[1])

cnt = 1
_, f = cap.read()
prevImg= down_scale_img(f,640)
while(cap.isOpened()):
    ret, f = cap.read()
    if f is None : break
    #f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    frame = down_scale_img(f,640)
    f2 = down_scale_img(frame, 160)
    f3 = down_scale_img(f2, 80)
    f4 = down_scale_img(f2, 40)
    frame = test_diff(prevImg, frame)
    #dup2 = test_canny(frame)
    find_lane(frame)
    prevImg = down_scale_img(f, 640)
    #f2 = get_fft(f2)
    overlay_img(frame, f2, 0,0)
    overlay_img(frame, f3, frame.shape[1]-f3.shape[1],0)
    overlay_img(frame, f4, frame.shape[1]-f4.shape[1]-f3.shape[1],0)
    cv2.imshow('f', frame)
    cnt += 1
    key = cv2.waitKey(1)
print(cnt)
cap.release()
cv2.destroyAllWindows()

