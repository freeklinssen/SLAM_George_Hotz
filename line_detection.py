import cv2
from display import display
import numpy as np
from extractor import extractor

w = 1920//2
h = 1080//2

F = 270
k = np.array([[F, 0, w//2], [0, F, h//2], [0, 0, 1]])

display = display(w,h)
orb = cv2.ORB_create()


fe = extractor(k)

def process_frame(img):
  img = cv2.resize(img,(w, h))
  # print(img.shape)
  print("okay")
  matches, Rt = fe.extract(img)

  print("Matches", len(matches))

  #if Rt is None:
   # return


  for pt1, pt2 in matches:
    u1,v1 = fe.denormalize(pt1)
    u2, v2 = fe.denormalize(pt2)
    cv2.circle(img,(u1,v1), color = (0,255,0), radius=3)
    cv2.circle(img, (u2, v2), color=(0, 0, 255), radius=3)
    cv2.line(img, (u1,v1), (u2, v2), color=(255, 0, 0))

  display.show(img)
  #cv2.imshow("image", img)






if __name__ == "__main__":
  cap = cv2.VideoCapture("SLAM_George_Hotz/test_countryroad.mp4")
  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      print("no")
      break