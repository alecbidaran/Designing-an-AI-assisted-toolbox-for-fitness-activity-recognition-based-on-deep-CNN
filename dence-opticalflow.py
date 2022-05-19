import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

vid=cv.VideoCapture('place the video dataset location')
vid.set(cv.CAP_PROP_POS_MSEC, 1 * 1000)
ret1,frame1=vid.read()
prvs=cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv=np.zeros_like(frame1)
hsv[...,1]=255
while(1):
  for i in range(1,5):
   vid.set(cv.CAP_PROP_POS_MSEC, i * 1000)
   ret2,frame2=vid.read()
   next=cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)

   flow=cv.calcOpticalFlowFarneback(prvs,next,  None, 0.5,5,3,3,10,2,1)
   mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
   hsv[..., 0] = ang *( 180/ np.pi/2)
   hsv[..., 2] = cv.normalize(mag, None, 0,255, cv.NORM_MINMAX)
   bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
   bgr = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
   bgr=bgr*8
   bgr=cv.GaussianBlur(bgr,(11,11),1)
   prvs=next

   plt.imsave('dense/test/lunge/lunge08'+str(i)+".jpg",bgr)
