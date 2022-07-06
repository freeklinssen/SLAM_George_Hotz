import numpy as np
import OpenGL.GL as gl
import pangolin
from multiprocessing import Process, Queue
import json
from helper import PoseRt



class Point(object):
  # point is a 3-D point in the world
  # each point is observed in muliple frames
  def __init__(self, mapp, location, color):
    self.location = location
    self.frames = []
    self.index = []
    self.color = np.copy(color)

    self.id = mapp.add_observation(self)

  def homogeneous(self):
    return np.array([self.location[0], self.location[1], self.location[2], 1.0])

  # 4:09:26
  def orb(self):
     return [f.des[idx] for f, idx in zip(self.frames, self.index)]


  def delete(self):
    for f, idx in zip(self.frames, self.index):
      f.pts[idx] = None
      del self

  def add_observation(self, frame, index):
    frame.pts[index] = self
    self.frames.append(frame)
    self.index.append(index)




class Map(object):
  def __init__(self):
    self.frames = []
    self.points = []
    self.max_point = 0
    self.max_frame = 0



  def serialize(self):
    ret = {}
    ret["points"] = [{"pt": p.location, "id": p.id.topist(), "color": p.color.tolist()} for p in self.points]
    for f in self.frames:
      ret["frames"].append({
        "id": f.id, "K": f.k, "pose": f.pose.tolist(), "h": f.h, "w": f.w,
         "kpus": f.kpus.tolist(), "des": f.des.tolist(),
         "pts": [p.id if p is not None else -1 for p in f.pts] })
    ret["max_frames"] = self.max_frame
    ret['max_points'] = self.max_point
    return json.dumps(ret)

  def deserialize(self, s):
      ret = json.loads(s)
      self.max_frame = ret['max_frame']
      self.max_point = ret['max_point']
      self.points = []
      self.frames = []

      pids = {}
      for p in ret['points']:
        pp = Point(self, p['pt'], p['color'], p['id'])
        self.points.append(pp)
        pids[p['id']] = pp

      for f in ret['frames']:
        ff = Frame(self, None, f['K'], f['pose'], f['id'])
        ff.w, ff.h = f['w'], f['h']
        ff.kpus = np.array(f['kpus'])
        ff.des = np.array(f['des'])
        ff.pts = [None] * len(ff.kpus)
        for i, p in enumerate(f['pts']):
          if p != -1:
            ff.pts[i] = pids[p]
        self.frames.append(ff)



  def add_observation(self, point):
    ret = self.max_point
    self.max_point += 1
    self.points.append(point)
    return ret

  def add_frame(self, frame):
    ret = self.max_frame
    self.max_frame += 1
    self.frames.append(frame)
    return ret


  #def load(self):



  #def save(self):
   # for f in self.frames:


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
