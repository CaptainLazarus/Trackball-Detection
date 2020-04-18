import cv2
from imutils.video import VideoStream
from collections import deque
import numpy as np 
import time
import argparse
import imutils

def initArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v" , "--video" , help="Path to video file")
    parser.add_argument("-b" , "--buffer" , help="Buffer size" , type=int , default=64)
    return vars(parser.parse_args())

if __name__ == "__main__":
    args = initArgs()

    # colorLower = (29, 86, 6)
    # colorUpper = (64, 255, 255)

    colorUpper = (110,50,50)
    colorLower = (130,255,255)

    pts = deque(maxlen=args["buffer"])

    if not args.get("video" , False):
        vs = VideoStream(src=0).start()

    else:
        vs = cv2.VideoCapture(args["video"])

    # not really necessary
    time.sleep(2.0)

    while True:
        frame = vs.read()

        frame = frame[1] if args.get("video",False) else frame

        if frame is None:
            break

        frame = imutils.resize(frame , width=600)
        blurred = cv2.GaussianBlur(frame , (11,11) , 0)
        hsv = cv2.cvtColor(blurred , cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv , colorUpper , colorLower)
        mask = cv2.erode(mask , None , iterations=1)
        mask = cv2.dilate(mask , None , iterations=1)

        cnts = cv2.findContours(mask.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center=None

        if len(cnts) > 0:
            c = max(cnts , key = cv2.contourArea)
            ((x,y) , radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M['m10']/M['m00']) , int(M['m01']/M['m00']))

            if radius > 20:
                cv2.circle(frame , (int(x),int(y)) , int(radius) , (0,255,255) , 2)
                cv2.circle(frame , center , 5 , (0,255,0) , -1)

                pts.append(center)

        for i in range(1,len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue

            thickness = int(np.sqrt(args["buffer"]/float(i+1)) * 2.5)
            cv2.line(frame , pts[i-1] , pts[i] , (255,0,0) , thickness)

        cv2.imshow("Frame" , frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    if not args.get('video' , False):
        vs.stop()

    else:
        vs.release()

    cv2.destroyAllWindows()