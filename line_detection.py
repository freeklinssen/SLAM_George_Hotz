import cv2
from display import display
import numpy as np
from Frame import Frame, denormalize, match_frames, IRt
#import g2o

w = 1920//2
h = 1080//2

F = 270
k = np.array([[F, 0, w//2], [0, F, h//2], [0, 0, 1]])

display = display(w,h)
orb = cv2.ORB_create()


class Point(object):
  # point is a 3-D point in the world
  # each point is observed in muliple frames


  def __init__(self, location):
    self.location = location
    self.frames = []
    self.index = []


  def add_observation(self, frame, index):
    self.frames.append(frame)
    self.index.append(index)


frames =[]
def process_frame(img):
  img = cv2.resize(img,(w, h))
  # print(img.shape)]
  frame = Frame(img, k)
  frames.append(frame)
  if len(frames) <= 1:
    return

  Rt, idx1, idx2 = match_frames(frames[-1], frames[-2])
  #print(pts)
  print(Rt)
  print(IRt)

  # triangulate
  pts4d = cv2.triangulatePoints(IRt[:3], Rt[:3], frames[-1].pts[idx1].T, frames[-2].pts[idx2].T).T

  # homogenious 3d coordinates (just make the last coordinate zero)
  pts4d /= pts4d[:, 3:]
  print(pts4d)

  frames[-1].pose = np.dot(Rt, frames[-2].pose)

  #rint(frames[-1].pose)

  #reject points without enough "paralax" and reject poinit behind the camera
  good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)


  for i, location in enumerate(pts4d):
    if not good_pts4d[i]:
      continue
    pt = Point(location)
    pt.add_observation(frames[-1], idx1[i])
    pt.add_observation(frames[-2], idx2[i])

  print(sum(pts4d), len(pts4d))




  print("Matches", len(frames[-1].pts))

  for pt1, pt2 in zip(frames[-1].pts[idx1], frames[-2].pts[idx2]):
    u1,v1 = denormalize(k, pt1)
    u2, v2 = denormalize(k, pt2)
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