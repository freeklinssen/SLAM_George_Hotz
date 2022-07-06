import numpy as np



def hamming_distance(a, b):
  r = (1 << np.arange(8))[:,None]
  return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)



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


def add_ones(x):
  result = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
  return result


def PoseRt(R,t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret


def FundamentalToRt(E):
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


def normalize(Kinv, pt):
  return np.dot(Kinv, add_ones(pt).T).T[:, 0:2]

#def denormalize(k, pt):
 # ret = np.dot(k, np.array([pt[0], pt[1], 1.0]).T)
  #return int(round(ret[0])), int(round(ret[1]))