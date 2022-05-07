import cv2
import numpy as np
from skimage.measure import ransac
import skimage


def add_ones(x):
  result = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
  return result

IRt = np.eye(4)


def extractRt(E):
  W = np.mat([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
  U,d,Vt = np.linalg.svd(E)
  #print(U)
  #print(d)
  #print(Vt)
  assert np.linalg.det(U) > 0
  if np.linalg.det(Vt) > 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  #print(R)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, 2]
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t

  #Rt = np.concatenate([R, t.reshape(3,1)], axis = 1)
  return ret


def extract(img):
  orb = cv2.ORB_create()
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel= 0.01, minDistance=3)
  #extraction retrun points and des
  kps = [cv2.KeyPoint(x= f[0][0], y=f[0][1], _size=20) for f in pts]
  kps, des = orb.compute(img, kps)

  return np.array([(kp.pt[0], kp .pt[1]) for kp in kps]), des



def normalize(Kinv, pt):
  return np.dot(Kinv, add_ones(pt).T).T[:, 0:2]

def denormalize(k, pt):
  ret = np.dot(k, np.array([pt[0], pt[1], 1.0]).T)
  return int(round(ret[0])), int(round(ret[1]))

def match_frames(f1, f2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  ret = []
  idx1, idx2 = [],[]
  matches = bf.knnMatch(f1.des, f2.des, k=2)

  for m, n in matches:
    if m.distance < 0.70 * n.distance:
      p1 = f1.pts[m.queryIdx]
      p2 = f2.pts[m.trainIdx]
      ret.append((p1,  p2))
      idx1.append(m.queryIdx)
      idx2.append(m.trainIdx)

  assert len(ret) > 8
  ret= np.array(ret)
  idx1 = np.array((idx1))
  idx2 = np.array((idx2))


  model, inliers = ransac((ret[:, 0], ret[:,1]),
                              #skimage.transform.FundamentalMatrixTransform,
                              skimage.transform.EssentialMatrixTransform,
                              min_samples=8,
                              residual_threshold=0.005,
                              max_trials=200)

  pts = ret[inliers]
  #print(model.params)
  Rt = extractRt(model.params)

  return  Rt, idx1[inliers], idx2[inliers]


class Frame( ):
  def __init__(self, mapp, img, k):
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    self.k = k
    self.kinv = np.linalg.inv(self.k)
    self.pose = IRt

    pts, self.des = extract(img)
    self.pts = normalize(self.kinv, pts)

    self.id = len(mapp.frames)
    mapp.frames.append(self)


    # matching

"""
    if self.last is not None:



    # filter
    Rt = None
    if len(ret) > 0:
      ret = np.array(ret)

      print(".,")
      print(ret.shape)
"""




