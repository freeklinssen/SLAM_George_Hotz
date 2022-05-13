import sys

#sys.path.append("pangolin")


import cv2
from display import display
import numpy as np
from Frame import Frame, denormalize, match_frames

import OpenGL.GL as gl
import pangolin
from multiprocessing import Process, Queue
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

    #create viewer proces
    #elf.q = Queue()
    #self.viewer = Process(target=self.viewer_thread, args=(self.q,))
    #self.viewer.deamon = True
    #self.viewer.start()
    self.viewer_init()
    self.state = None
    self.q =Queue()
    p = Process(target = self.viewer_thread, arg =(self.q,))
    p.daemon = True
    p.start()

  def viewer_thread(self, q):
    self.viewer_init()
    while 1:
      self.viewer_refresh(q)


  def viewer_init(self):

    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    self.handler = pangolin.Handler3D(self.scam)

    # Create Interactive View in window
    self.dcam = pangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
    self.dcam.SetHandler(self.handler)

    self.state = None

  def viewer_refresh(self):
      #if self.state is None or not q.empty():
        #state = q.get(True)
      ppts = np.array(d[:3, 3] for d in self.state[0])
      spts = np.array([self.state[1]])
      print(ppts.shape)
      print(spts.shape)

      gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
      gl.glClearColor(1.0, 1.0, 1.0, 1.0)
      self.dcam.Activate(self.scam)


      gl.glPointSize(10)
      gl.glColor3f(0.0, 1.0, 0.0)
      pangolin.DrawPoins(ppts)

      gl.glPointSize(2)
      gl.glColor3f(0.0, 1.0, 0.0)
      pangolin.DrawPoins(spts)

      pangolin.FinishFrame()

  def display(self ):
    poses, pts  = [], []
    for f in self.frames:
      poses.append(f.pose)
    for p in self.points:
      pts.append(p.location)
    self.q.put((poses, pts))

    #self.viewer_refresh()

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
  #print(f1.pose)
  #print(Rt)
  #print(f2.pose)

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

  #display.show(img)
  mapp.display()
  cv2.imshow("image", img)




if __name__ == "__main__":
  cap = cv2.VideoCapture("SLAM_George_Hotz/test_countryroad.mp4")
  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      print("no")
      break