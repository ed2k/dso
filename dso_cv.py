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


#axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

cap = cv2.VideoCapture(0)

t1 = 15
t2 = 15
rho = 100
theta = 1
threshold = 1
r, frame = cap.read()
edges = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(edges, t1, t2)
objpoints = np.array([[10,10,100],[100,10,100],[10,100,100],[100,100,100]], np.float32)
imgpoints = np.array([[5,5], [50,5],[5,50],[50,50]], np.float32)
camM = np.array([[.5,0,0,],[0,.5,0],[0,0,1]],np.float32)
r, rvecs,tvecs = cv2.solvePnP(objpoints, imgpoints,  camM, None)
while(True):
    ret, f = cap.read()
    frame = f
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('f', frame)


    key = cv2.waitKey(1)
    if key != 255: print(key)
    # key 'q' is 99
    if key == 99: break

cap.release()
#img = cv2.imread('sequence_11/vignette.png')
#cv2.imshow('img', img)
#cv2.waitKey(0)
cv2.destroyAllWindows()


