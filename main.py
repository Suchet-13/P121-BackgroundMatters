import cv2
import time
import numpy as np

#saving the file as 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

#starting the camera
cap = cv2.VideoCapture(0)

#allowing the webcam to start by making the code sleep for 2 seconds
time.sleep(2)
bg = 0

#capturing the background for 60 frames
for i in range(60):
    ret, bg = cap.read()
#flipping the bg
bg = np.flip(bg, axis=1)

#readiing the captured frame until the camera is open
while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    #flipping the image for consistency
    img = np.flip(img, axis=1)

    #converting the colour from bgr to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #generating mask to detect black colour. These values can also be changed as per the colour
    lower_black = np.array([0,0,0])
    upper_black = np.array([0,0,5])
    mask_1 = cv2.inRange(hsv, lower_black, upper_black)

    lower_black = np.array([0,0,6])
    upper_black = np.array([0,0,10])
    mask_2 = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = mask_1 + mask_2

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    mask_2 = cv2.bitwise_not(mask_1)

    res_1 = cv2.bitwise_and(img, img, mask = mask_2)

    res_2 = cv2.bitwise_and(bg, bg, mask = mask_1)

    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    output_file.write(final_output)

    cv2.imshow('magic', final_output)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()