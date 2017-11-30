import numpy as np
import cv2


def draw(img, pts):
    x1,y1 = pts[0][0]
    x2,y2 = pts[1][0]
    x3,y3 = pts[2][0]
    x4,y4 = pts[3][0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
    cv2.line(img,(x1,y1),(x3,y3),(0,255,0),1)
    cv2.line(img,(x1,y1),(x4,y4),(0,255,0),1)

def draw_pts(img, x,y):
  x = int(x)
  y = int(y)
  cv2.rectangle(img, (x,y),(x+1,y+1),(0,255,0),1);


def get_calib():
    line=open('sequence_11/ncalib.txt').readline()
    f = line.split()
    print (len(f),f)
    r = []
    for x in f: r.append(float(x))
    return r
#axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

cap = cv2.VideoCapture(0)

calib = get_calib()
while(True):
    ret, f = cap.read()
    frame = f
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for x,y in enumerate(calib): draw_pts(frame, x, y)
    cv2.imshow('f', frame)


    key = cv2.waitKey(1)
    if key != 255: print(key)
    # key 'q' is 113
    if key == 113: break

cap.release()
cv2.destroyAllWindows()


