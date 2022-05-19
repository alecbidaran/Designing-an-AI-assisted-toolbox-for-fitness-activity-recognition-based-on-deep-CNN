import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
vid=cv.VideoCapture('')
ret,frame=vid.read()
while(1):
    for i in range(74):
        ret,frame=vid.read()
        plt.imshow(frame)
        plt.show()
        plt.imsave("rgb/train/handspushups/handspushup15"+str(i)+".jpg",frame)
