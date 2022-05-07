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

# global map
class Map(object):
  def __init__(self):
    self.frames = []
    self.points = []

  def display(self ):
    for f in self.frames:
      print(f.id)
      print(f.pose)
      print(" ")

mapp = Map()

class Point(object):
  # point is a 3-D point in the world
  # each point is observed in muliple frames
  def __init__(self, mapp, location):
    self.location = location
    self.frames = []
    self.index = []

    self.id = len(mapp.points)
    mapp.points.append(self)

  def add_observation(self, frame, index):
    self.frames.append(frame)
    self.index.append(index)



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
  #print(pts)
  #print(Rt)
  #print(IRt)

  # triangulate
  pts4d = cv2.triangulatePoints(f1.pose[:3], f2.pose[:3], f1.pts[idx1].T, f2.pts[idx2].T).T

  # homogenious 3d coordinates (just make the last coordinate zero)
  pts4d /= pts4d[:, 3:]
  #print(pts4d)


  #reject points without enough "paralax" and reject poinit behind the camera
  good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)


  for i, location in enumerate(pts4d):
    if not good_pts4d[i]:
      continue
    pt = Point(mapp, location)
    pt.add_observation(f1, idx1[i])
    pt.add_observation(f2, idx2[i])

  print(sum(pts4d), len(pts4d))




  print("Matches", len(f1.pts))

  for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
    u1,v1 = denormalize(k, pt1)
    u2, v2 = denormalize(k, pt2)
    cv2.circle(img,(u1,v1), color = (0,255,0), radius=3)
    cv2.circle(img, (u2, v2), color=(0, 0, 255), radius=3)
    cv2.line(img, (u1,v1), (u2, v2), color=(255, 0, 0))

  display.show(img)
  mapp.display()
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