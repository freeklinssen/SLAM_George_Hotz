import cv2
import numpy as np
from skimage.measure import ransac
import skimage
from scipy.spatial import cKDTree
from helper import FundamentalToRt, normalize
#import g2o






def ExtractFeatures(img):
  orb = cv2.ORB_create()
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel= 0.01, minDistance=7)
  #extraction retrun points and des
  kps = [cv2.KeyPoint(x= f[0][0], y=f[0][1], _size=20) for f in pts]
  kps, des = orb.compute(img, kps)

  return np.array([(kp.pt[0], kp .pt[1]) for kp in kps]), des




def match_frames(f1, f2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  ret = []
  idx1, idx2 = [],[]
  idx1s, idx2s = set(), set()
  matches = bf.knnMatch(f1.des, f2.des, k=2)

  for m, n in matches:
    if m.distance < 0.75 * n.distance:
      p1 = f1.kps[m.queryIdx]
      p2 = f2.kps[m.trainIdx]


      # travel les than 10% of diagonal and be in orb distance 32
      if m.distance < 32:
        if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
          idx1.append(m.queryIdx)
          idx2.append(m.trainIdx)
          idx1s.add(m.queryIdx)
          idx2s.add(m.trainIdx)
          ret.append((p1, p2))



  assert len(ret) > 8
  ret = np.array(ret)
  idx1 = np.array((idx1))
  idx2 = np.array((idx2))


  model, inliers = ransac((ret[:, 0],ret[:,1]),
                          skimage.transform.FundamentalMatrixTransform,
                          #skimage.transform.EssentialMatrixTransform,
                          min_samples=8,
                          residual_threshold=0.001,
                          max_trials=100)

  print(len(matches), len(inliers))

  #pts = ret[inliers]
  #print(model.params)

  return FundamentalToRt(model.params), idx1[inliers], idx2[inliers]


class Frame( ):
  def __init__(self, mapp, img, k, pose = np.eye(4)):
    #self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    self.k = k
    self.pose = pose

    #if img is not None:
    self.h, self.w = img.shape[0:2]


    self.kpus, self.des = ExtractFeatures(img)
    self.pts = [None]*len(self.kps)
    self.id = mapp.add_frame(self)



  #inverse of intrinsics matrix
  @property
  def kinv(self):
    if not hasattr(self, "_kinv"):
      self._kinv = np.linalg.inv(self.k)
    return self._kinv

  #normalized keypoints
  @property
  def kps(self):
    if not hasattr(self, "_kps"):
      self._kps = normalize(self.kinv, self.kpus)
    return self._kps

  #KD treee
  @property
  def kd(self):
    if not hasattr(self, "_kd"):
      self._kd = cKDTree(self.kpus)
    return self._kd




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




