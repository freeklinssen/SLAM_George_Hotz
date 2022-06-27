import numpy as np
import OpenGL.GL as gl
import pangolin
from multiprocessing import Process, Queue

class Point(object):
  # point is a 3-D point in the world
  # each point is observed in muliple frames
  def __init__(self, mapp, location, color):
    self.location = location
    self.frames = []
    self.index = []
    self.color = color


    self.id = mapp.max_points
    mapp.max_points += 1
    mapp.points.append(self)

  def delete(self):
    for f in self.frames:
      f.pts[f.pts.index(self)] = None
      del self

  def add_observation(self, frame, index):
    frame.pts[index] = self
    self.frames.append(frame)
    self.index.append(index)





class Map(object):
  def __init__(self):
    self.frames = []
    self.points = []
    self.max_points = 0

    #self.viewer_init()
    self.state = None
    self.q = Queue()
    #p = Process(target = self.viewer_thread, arg =(self.q,))
    #p.daemon = True
    #p.start()

  """optimizer"""
  #def optimizer:
  #with g2o that iis not working ofcourse


  """
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
  """
  def display(self ):
    poses, pts  = [], []
    for f in self.frames:
      poses.append(f.pose)
    for p in self.points:
      pts.append(p.location)
    self.q.put((poses, pts))
