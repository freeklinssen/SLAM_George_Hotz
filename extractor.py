import cv2
import numpy as np
from skimage.measure import ransac
import skimage


def add_ones(x):
  result = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
  return result

def extractRt(E):
  W = np.mat([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
  U,d,Vt = np.linalg.svd(E)
  assert np.linalg.det(U) > 0
  if np.linalg.det(Vt) > 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, 2]
  Rt = np.concatenate([R, t.reshape(3,1)], axis = 1)
  print(Rt)
  return Rt

class extractor(object):
  def __init__(self, k):
    self.orb = cv2.ORB_create()
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    self.last = None
    self.k = k
    self.Kinv = np.linalg.inv(self.k)

  def normalize(self, pt):
    return np.dot(self.Kinv, add_ones(pt).T).T[:, 0:2]

  def denormalize(self, pt):
    #print(np.array([pt[0], pt[1], 1]))
    ret = np.dot(self.k, np.array([pt[0], pt[1], 1.0]).T)
    return int(round(ret[0])), int(round(ret[1]))

    #return int(round(pt[0])+self.w), int(round(pt[1])+self.h)




  def extract(self, img):
    feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel= 0.01, minDistance=3)

    #extraction
    kps = [cv2.KeyPoint(x= f[0][0], y=f[0][1], _size=20) for f in feats]
    kps, des = self.orb.compute(img, kps)

    # matching
    ret = []

    if self.last is not None:
      matches = self.bf.knnMatch(des, self.last["des"], k=2)
      for m,n in matches:
        if m.distance < 0.70 * n.distance:
          kp1 = kps[m.queryIdx].pt
          kp2 = self.last["kps"][m.trainIdx].pt
          ret.append((kp1, kp2))


    # filter
    Rt = None
    if len(ret) > 0:
      ret = np.array(ret)

      print(".,")
      print(ret.shape)


      ret[:, 0, :] = self.normalize(ret[:, 0, :])
      ret[:, 1, :] = self.normalize(ret[:, 1, :])


      model, inliers = ransac((ret[:, 0], ret[:,1]),
                              #skimage.transform.FundamentalMatrixTransform,
                              skimage.transform.EssentialMatrixTransform,
                              min_samples=8,
                              residual_threshold=0.005,
                              max_trials=200)

      ret = ret[inliers]


      Rt = extractRt(model.params)

      #print(model.params)
      #u,d,vt = np.linalg.svd(model.params)
      #f_est = (np.sqrt(2)/ d[0]+d[1])/2
      #print(f_est)

    self.last = {"kps": kps, "des": des}

    return ret, Rt