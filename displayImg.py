import cv2, numpy as np
import os
import enet

GREEN = (0,255,0)

class myreader:
  def __init__(m, f=''):
    if f == '':
      m.handle = cv2.VideoCapture(0);
      m.idx = -1;
    elif f.endswith('.mp4') or f.endswith('.webm'):
      m.handle = cv2.VideoCapture(f)
      m.idx = -1;
    elif os.path.isdir(f):
      fs = os.listdir(f)
      fs.sort()
      m.idx = int(fs[0][:-4])
      m.end = int(fs[-1][:-4])
      m.num_digits = len(fs[0])-4
      m.path = f;
  def get_img_path(m):
    d = str(m.idx)
    d = d.zfill(m.num_digits)
    return os.path.join(m.path, d+'.jpg')
  def isOpened(m):
    if m.idx == -1: return m.handle.isOpened()
    p = m.get_img_path()
    return os.path.exists(p)
  def read(m):
    if m.idx == -1: return m.handle.read()
    p = m.get_img_path()
    m.idx+=1
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

def fg_mask(img):
    img = fgbg.apply(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img
def test_diff(prev_img, img):
    img = fgbg.apply(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img
    #return cv2.absdiff(prev_img,img)
def test_canny(img):
    return cv2.Canny(img,100,200)

def get_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return 20*np.log(np.abs(fshift))

def get_lane_area(rho, theta, width, height):
    def intersect(rho1):
        c = np.cos(theta)
        s = np.sin(theta)
        x = int(rho1/c)
        y = int(rho1/s)
        if y<height: return ((x,0),(0,y))
        x1 = int((y-height)*s/c)
        return ((x,0),(x1,y))
    pts = intersect(rho+10)
    p0 = intersect(rho-10)
    return [p0[0],p0[1],pts[1],pts[0]]
def draw_line(img, rho, theta, offy, h):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho+(h/2)
    x1 = int(x0 + offy*(-b))
    y1 = int(y0 + offy*(a))
    x2 = int(x0 - offy*(-b))
    y2 = int(y0 - offy*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
def find_lane(img, track=None):
    h,w,_ = img.shape
    offy = int(h/2)
    grey = cv2.cvtColor(img[-h/2:,:w/2], cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(grey, 12,255, cv2.THRESH_TOZERO)
    _, thr = cv2.threshold(grey, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = np.zeros((h/2,w/2), np.uint8)
    cnts_thr = 20
    lines_thr = 90
    # focus area
    pts = [(0,h/4),(0,h/2),(w/4,h/2),(w/2-w/16,0),(w/2-w/8,0)]
    if track is not None:
        pts = get_lane_area(track[0],track[1],w/2,h/2)
        cnts_thr = 2
    cv2.fillConvexPoly(mask, np.int32(pts), 1.0)
    thr = mask * thr
    _,cnts,_ = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thr = np.zeros((h/2,w/2), np.uint8)
    for cnt in cnts:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx) < cnts_thr: continue
        cv2.drawContours(img, [cnt], -1, (0,255,0), offset=(0,offy))
        cv2.drawContours(thr, [cnt], -1, 255, offset=(0,0))
    #edges = cv2.Canny(grey,50,150,apertureSize = 3)
    lines = cv2.HoughLines(thr,1,np.pi/180,lines_thr)
    if lines is None: return None
    for line in lines:
        rho,theta = line[0]
        if theta < np.pi*20/180: continue
        if theta > np.pi*70/180: continue
        print(rho, theta)
        return (rho, theta)
def find_corners(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    img[dst>0.01*dst.max()]=[0,0,255]
def draw_center_path(img):
    h,w,_ = img.shape
    pts = np.int32([(0.41*w,0.44*h), (0.09*w,h),(0.71*w,h)])
    cv2.polylines(img,[pts.reshape((-1,1,2))],True,GREEN)

def get_interests_area(img):
    h,w,_ = img.shape
    dx = w/10
    dy = h/18
    dh = h/8
    # left, center, right drive area contour
    pts = np.int32([(w/2-dx-dx,h/2-dy), (0,h-dh),(0,h),(w,h),(w,h-dh),(w/2,h/2-dy)])
    return pts

def find_calib(img):
    h,w,_ = img.shape
    dx = w/10
    dy = h/18
    dh = h/8
    pts = get_interests_area(img)
    mask = np.zeros((h,w,3), np.uint8)
    cv2.fillConvexPoly(mask, pts, (255,255,255))
    #img = mask*img
    cv2.bitwise_and(img,mask,img)
    # center path
    return (dx,dy,dh)
def find_box2(img, contour):
    h,w,_ = img.shape
    grey = cv2.cvtColor(img[h/4:,:], cv2.COLOR_RGB2GRAY)
    #grey = cv2.Canny(grey,50,150,apertureSize = 3)
    _, thr = cv2.threshold(grey, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts_thr = w/6
    _,cnts,_ = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r = []
    for cnt in cnts:
      if len(cnt) > cnts_thr or len(cnt)<w/40: continue
      rect = cv2.minAreaRect(cnt)
      box = cv2.boxPoints(rect)
      box = np.int0(box)
      box = box+[0,h/4]
      for pt in box:
        p = tuple(pt)
        if cv2.pointPolygonTest(contour,p,False)>0:
          r.append(box)
          break
    return r
def find_box(img, track=None):
    h,w,_ = img.shape
    grey = cv2.cvtColor(img[h/4:,:], cv2.COLOR_RGB2GRAY)
    #grey = cv2.Canny(grey,50,150,apertureSize = 3)
    _, thr = cv2.threshold(grey, 20,255,cv2.THRESH_BINARY)
    cnts_thr = w/4
    _,cnts,_ = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r = []
    for cnt in cnts:
      if len(cnt) > cnts_thr or len(cnt) < w/40: continue
      rect = cv2.minAreaRect(cnt)
      box = cv2.boxPoints(rect)
      box = np.int0(box)
      r.append(box+[0,h/4])
    return r
def cnts2mask(mask, cnts, h = 0, w = 0):
    if mask is None:
      mask = np.zeros((h,w,3), np.uint8)
    for cnt in cnts:
      cv2.fillConvexPoly(mask, cnt, (255,255,255))
    return mask
def cut_half(img):
    h,w,_=img.shape
    img[:h/2,:] = (0,0,0)

def rotate_img(img):
  rows,cols,_ = img.shape
  M = cv2.getRotationMatrix2D((cols/2,rows/2),2.5,1)
  dst = cv2.warpAffine(img,M,(cols,rows))
  return dst

import sys
if len(sys.argv) < 2:
  cap = myreader('../SEQ_0/v.mp4')
else:
  cap = myreader(sys.argv[1])

# TODO, move enet to separte process
#net = enet.create_default()
track_lane = None
cnt = 1
_, prev = cap.read()
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
print(prev.shape)
prev = rotate_img(prev)
while(cap.isOpened()):
    ret, f = cap.read()
    if f is None : break
    h,w,_ = f.shape
    f = rotate_img(f)
    #f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    frame = down_scale_img(f,640)
    f2 = down_scale_img(frame, 160)
    f3 = down_scale_img(f2, 80)
    f4 = down_scale_img(f2, 20)
    cut_half(f4)
    f4 = down_scale_img(f4, 160)
    #dup2 = test_canny(frame)
    img = down_scale_img(f, 640)
    #track_lane = find_lane(f2,track_lane)
    find_box(f2,None)
    if track_lane is not None:
        #print (track_lane)
        draw_line(f2, track_lane[0],track_lane[1], 250, h*640/w)
    #f2 = get_fft(f2)
    cnts = find_box2(frame,get_interests_area(frame))
    #print(len(cnts))
    find_calib(frame)
    img = down_scale_img(f,640)

    mask = None
    h,w,_ = frame.shape
    mask = cnts2mask(mask,cnts,h,w)
    cv2.bitwise_and(frame,mask,frame)
    draw_center_path(frame)
    find_corners(f2)

    #cmap = enet.predict(net,f)
    cmap = enet.get_offline_result(cnt)
    cmap = down_scale_img(cmap, 160)
    overlay_img(frame, f2, 0,0)
    overlay_img(frame, cmap, f2.shape[1],0)
    overlay_img(frame, f3, frame.shape[1]-f3.shape[1],0)
    overlay_img(frame, f4, frame.shape[1]-f4.shape[1]-f3.shape[1],0)
    cv2.imshow('f', frame)
    cnt += 1
    key = cv2.waitKey(1)
print(cnt)
cap.release()
cv2.destroyAllWindows()

