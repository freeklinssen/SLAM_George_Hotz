import cv2
import numpy as np
from skimage.measure import ransac
import skimage

#import g2o


def add_ones(x):
  result = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
  return result

def PoseRt(R,t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret



def extractRt(E):
  # extract from the essential matrix the extrinsic matrix
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

  #essential matrix is translation matrix times the rotation matrix  (TxR)
  # fundametal metrix is essental matrix with two times the dot product with the intrisic/calibration matrix
  # so wath we do here is extracting the rotation matrix and the transation matrix (or vector) from the essentail matrix

  # return the extrinsic matrix = the roation and and translation vector in one
  # projection matrix is extrinsic matrix and intrinsic/calibration matrix together

  #Rt = np.concatenate([R, t.reshape(3,1)], axis = 1)
  return ret


def extract(img):
  orb = cv2.ORB_create()
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 1000, qualityLevel= 0.01, minDistance=7)
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
    if m.distance < 0.75 * n.distance:
      p1 = f1.kpus[m.queryIdx]
      p2 = f2.kpus[m.trainIdx]


      # travel les than 10% of diagonal and be in orb distance 32
      if np.linalg.norm((p1 - p2)) < 0.2 * np.linalg.norm([f1.w, f1.h]) and m.distance < 32:
        if m.queryIdx not in idx1 and m.trainIdx not in idx2:
          idx1.append(m.queryIdx)
          idx2.append(m.trainIdx)
          ret.append((p1, p2))



  assert len(ret) > 8
  ret = np.array(ret)
  idx1 = np.array((idx1))
  idx2 = np.array((idx2))


  model, inliers = ransac((ret[:, 0], ret[:,1]),
                              skimage.transform.FundamentalMatrixTransform,
                              #skimage.transform.EssentialMatrixTransform,
                              min_samples=8,
                              residual_threshold=0.001,
                              max_trials=100)

  print(len(matches), len(inliers))

  #pts = ret[inliers]
  #print(model.params)
  Rt = extractRt(model.params)

  return Rt, idx1[inliers], idx2[inliers]


class Frame( ):
  def __init__(self, mapp, img, k):
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    self.k = k
    if img is not None:
      self.h, self.w = img.shape[0:2]

    self.kinv = np.linalg.inv(self.k)
    self.pose = np.eye(4)

    kpus, self.des = extract(img)
    self.kpus = normalize(self.kinv, kpus)
    self.pts = [None]*len(self.kpus)

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




