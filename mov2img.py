import cv2



cap = cv2.VideoCapture('/home/a/SEQ_0/v.mp4')
cnt = 1
while cnt < 3000:
    cap.grab()
    cnt += 1
while(cap.isOpened()):
    ret, f = cap.read()
    if f is None or cnt > 5900: break
    frame = f
    h,w,c = frame.shape
    frame = cv2.resize(frame, (1280*h/w,1280))
    #cv2.imshow('f', frame)
    if cnt > 3000:
      cv2.imwrite('/home/a/SEQ_0/z3/%04d.jpg' % (cnt), frame)
    cnt += 1
    print (cnt, frame.shape)

cap.release()
cv2.destroyAllWindows()


