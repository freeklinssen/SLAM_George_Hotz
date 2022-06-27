import sys

#sys.path.append("pangolin")


import cv2
from display import display
import numpy as np
from Frame import Frame, denormalize, match_frames

from pointmap import Map, Point
#import g2o

w, h = 1920//2, 1080//2
#w, h = 1242, 375

F = 800
#calibration matrix, part of the intrinsic matrix
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
  img = cv2.resize(img,  (w, h))
  # print(img.shape)]
  frame = Frame(mapp, img, k)
  if frame.id == 0:
    return

  print("\n***frame %d***", (frame.id))

  #match with previous frame

  f1 = mapp.frames[-1]
  f2 = mapp.frames[-2]
  #print(f1.kps, f1.des)
# Rt is the intrinsic matrix, maps 3d point to pixel ip photo
  Rt, idx1, idx2 = match_frames(f1, f2)
  print(idx1, idx2)
  f1.pose = np.dot(Rt, f2.pose)

  # filter out points we already used
  for i, idx in enumerate(idx2):
    if f2.pts[idx] is not None:
      f2.pts[idx].add_observation(f1, idx1[i])



  print("f1 pose:", f1.pose)
  #print(f1.pts)
  #triangulate

  good_pts4d = np.array([f1.pts[1] is None for i in idx1])

  #homogenious 3d coordinates (just make the last coordinate zero)
  pts4d = triangulate(f1.pose, f2.pose, f1.kpus[idx1], f2.kpus[idx2])
  print(pts4d)
  good_pts4d &= np.abs(pts4d[:, 3]) > 0.005

  pts4d /= pts4d[:, 3:]
  print(pts4d)


  pts4d_lp = np.dot(np.linalg.inv(f1.pose), pts4d.T).T
  good_pts4d &= pts4d_lp[:, 2] > 0
  #reject points without enough "paralax" and reject poinit behind the camera
  #unmatched_pts = np.array([f1.pts[i] is None for i in idx1])
  #np.dot(f1.pose, pts4d)


  #good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_pts
  print("adding: %d points" % np.sum(good_pts4d))

  #print(len(good_pts4d), sum(good_pts4d))


  for i, location in enumerate(pts4d):
    if not good_pts4d[i]:
      continue
    u,v = int(round(f1.kpus[idx1[i], 0])), int(round(f1.kpus[idx1[i], 1]))
    pt = Point(mapp, location, img[u,v])
    pt.add_observation(f1, idx1[i])
    pt.add_observation(f2, idx2[i])

  #print(sum(pts4d), len(pts4d))




  print("Matches", len(f1.kpus))

  for pt1, pt2 in zip(f1.kpus[idx1], f2.kpus[idx2]):
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
  cap = cv2.VideoCapture("SLAM_George_Hotz/test_videos/test_ohio.mp4")
  #cap = cv2.VideoCapture("SLAM_George_Hotz/test_videos/fest_countryroad.mp4")
  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      print("no")
      break