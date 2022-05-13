import sys

#sys.path.append("pangolin")


import cv2
from display import display
import numpy as np
from Frame import Frame, denormalize, match_frames

from pointmap  import Map, Point
#import g2o

w = 1920//2
h = 1080//2

F = 270
k = np.array([[F, 0, w//2], [0, F, h//2], [0, 0, 1]])
kinv = np.linalg.inv(k)


# global map
    #self.viewer_refresh()

mapp = Map()
display = display(w,h)

def triangulate( pose1, pose2, pts1, pts2):
  ret = np.zeros((pts1.shape[0],4))
  pose1 = np.linalg.inv(pose1)
  pose2 = np.linalg.inv(pose2)
  for i, p in enumerate(zip(pts1, pts2)):
    A = np.zeros((4,4))
    A[0] = p[0][0] * pose1[2] - pose1[0]
    A[1] = p[0][1] * pose1[2] - pose1[1]
    A[2] = p[1][0] * pose2[2] - pose2[0]
    A[3] = p[1][1] * pose2[2] - pose2[1]
    _, _, vt = np.linalg.svd(A)
    ret[i] = vt[3]
  return ret

def process_frame(img):
  img = cv2.resize(img,(w, h))
  # print(img.shape)]
  frame = Frame(mapp, img, k)
  if frame.id == 0:
    return

  f1 = mapp.frames[-1]
  f2 = mapp.frames[-2]

  Rt, idx1, idx2 = match_frames(f1, f2)
  f1.pose = np.dot(Rt, f2.pose)

  #print(f1.pose)
  #print(f1.pts)
  # triangulate

  pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
  # homogenious 3d coordinates (just make the last coordinate zero)
  pts4d /= pts4d[:, 3:]
  #print(pts4d)


  #reject points without enough "paralax" and reject poinit behind the camera
  good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
  #print(len(good_pts4d), sum(good_pts4d))


  for i, location in enumerate(pts4d):
    if not good_pts4d[i]:
      continue
    pt = Point(mapp, location)
    pt.add_observation(f1, idx1[i])
    pt.add_observation(f2, idx2[i])

  #print(sum(pts4d), len(pts4d))




  print("Matches", len(f1.pts))

  for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
    u1,v1 = denormalize(k, pt1)
    u2, v2 = denormalize(k, pt2)
    cv2.circle(img,(u1,v1), color = (0,255,0), radius=3)
    cv2.circle(img, (u2, v2), color=(0, 0, 255), radius=3)
    cv2.line(img, (u1,v1), (u2, v2), color=(255, 0, 0))

  if  display is not None:
    display.show(img)
  #mapp.display()
  #cv2.imshow("image", img)




if __name__ == "__main__":
  cap = cv2.VideoCapture("SLAM_George_Hotz/test_videos/test_drone.mp4")
  #cap = cv2.VideoCapture("SLAM_George_Hotz/test_videos/test_countryroad.mp4")
  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      print("no")
      break