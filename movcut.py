import cv2



cap = cv2.VideoCapture('/home/a/videoplayback.mp4')
cnt = 1
while cnt < 0:
    cap.grab()
    cnt += 1

while(cap.isOpened()):
    ret, f = cap.read()
    if f is None or cnt > 9999900: break
    frame = cv2.resize(f,(1280,720))
    cv2.imwrite('/home/a/SEQ_0/z0/%05d.jpg' % (cnt), frame)
    print (cnt, frame.shape)
    #h,w,c = frame.shape
    #cv2.imshow('f', frame)
    cnt += 1
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()


