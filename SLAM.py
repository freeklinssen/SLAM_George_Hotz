import sys
import os

#sys.path.append("pangolin")


import cv2
from display import display2d
import numpy as np
from Frame import Frame, match_frames
from helper import hamming_distance, triangulate
from pointmap import Map, Point
#import g2o

w, h = 1920//2, 1080//2
#w, h = 1242, 375

F = 800
#calibration matrix, part of the intrinsic matrix

# global map
    #self.viewer_refresh()

mapp = Map()



def process_frame(img):
  img = cv2.resize(img,  (w, h))
  # print(img.shape)]
  frame = Frame(mapp, img, k)
  if frame.id == 0:
    return

  print("\n***frame %d***", (frame.id))

  #match with previous frame

  f1 = mapp.frames[-1]
  f2 = mapp.frames[-2]

  #print(f1.kps, f1.des)
# Rt is the intrinsic matrix, maps 3d point to pixel ip photo
  Rt, idx1, idx2 = match_frames(f1, f2)
  print(idx1, idx2)

  # add new observations if the point is already observed in the previous frame
  # TODO

  for i, idx in enumerate(idx2):
    if f2.pts[idx] is not None and f1.pts[idx1[i]] is None:
      f2.pts[idx].add_observation(f1, idx1[i])


  if frame.id < 5 or True:
    f1.pose = np.dot(Rt, f2.pose)
  else:
    velocity = np.dot(f2.pose, np.linalg.inv(mapp.frames[-3].pose))
    f1.pose = np.dot(velocity, f2.pose)

  # add new observations if point is already ob served in previous frames
  # TODO: find out if this is better to do before or after search by projection
  for i, idx in enumerate(idx2):
    if f2.pts[idx] is not None and f1.pts[idx1[i]] is None:
      f2.pts[idx].add_observation(f1, idx1[i])

  # search by projection
  if len(mapp.points) > 0:
    map_points = np.array([p.homogeneous() for p in mapp.points])
    projs = np.dot(np.dot(k, f1.pose[:3]), map_points.T).T
    projs = projs[:, 0:2] / projs[:, 2:]
    good_points = (projs[:, 0] > 0) & (projs[:, 0] < w) & \
                  (projs[:, 1] > 0) & (projs[:, 1] < h)

    print(good_points)
    print("f1.pts", f1.pts)

    for i, p in enumerate(mapp.points):
      if not good_points[i]:
        continue

      for m_idx in f1.kd.query_ball_point(projs[i], 5):
        if f1.pts[m_idx] is None:
          for o in p.orb():
            o_dist = hamming_distance(o, f1.des[m_idx])
            if o_dist < 32.0:
              p.add_observation(f1, m_idx)
              break


  # pose optimization
  #print("f1 pose:", f1.pose)
  #print(f1.pts)
  #triangulate

  # triangulate points we don't have matches for
  good_pts4d = np.array([f1.pts[i] is None for i in idx1])

  # reject points without enough "paralax"
  pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
  good_pts4d &= np.abs(pts4d[:, 3]) > 0.005

  # homogenious 3d coordinates (just make the last coordinate zero)
  pts4d /= pts4d[:, 3:]
  #print(pts4d)


  #location in front of camara:
  #pts_tri_local =triangulate(Rt, np.eye(4), f1.kps[idx1], f2.kps[idx2])
  #pts_tri_local /= pts_tri_local[:, 3:]
  #good_pts4d &= pts_tri_local[:, 2] > 0


  # another way to reject points behind the camara better:
  #TODO: maybe not needed
  #pts4d_lp = np.dot(f1.pose, pts4d.T).T
  #good_pts4d &= pts4d_lp[:, 2] > 0


  print("adding: %d points" % np.sum(good_pts4d))

  #print(len(good_pts4d), sum(good_pts4d))


  # Add new points to the map from pairwise matches
  for i, location in enumerate(pts4d):
    if not good_pts4d[i]:
      continue
    u,v = int(round(f1.kps[idx1[i], 0])), int(round(f1.kps[idx1[i], 1]))
    pt = Point(mapp, location[0:3], img[u,v])
    pt.add_observation(f1, idx1[i])
    pt.add_observation(f2, idx2[i])

  #print(sum(pts4d), len(pts4d))



  print(f1.pts)
  print("Matches", len(f1.kps))

  if display2d is not None:
    #paint points on the image
    for i1, i2 in zip(idx1, idx2):
      u1, v1 = int(round(f1.kpus[i1][0])), int(round(f1.kpus[i1][1]))
      u2, v2 = int(round(f2.kpus[i2][0])), int(round(f2.kpus[i2][1]))
      if f1.pts[i1] is not None:
        #if len(f1.lts[i1].frames) >= 5:

        cv2.circle(img, (u1, v1), color=(255, 0, 0), radius=3)
        cv2.circle(img, (u2, v2), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))
      else:
        cv2.circle(img, (u1, v1), color=(0, 0, 255), radius=3)
    display2d.show(img)
  #mapp.display()
  #cv2.imshow("image", img)




if __name__ == "__main__":
  cap = cv2.VideoCapture("SLAM_George_Hotz/test_videos/test_ohio.mp4")
  #cap = cv2.VideoCapture("SLAM_George_Hotz/test_videos/fest_countryroad.mp4")

  W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  CNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  F = float(os.getenv("F", "525"))

  #skip to particular frame
  if os.getenv("SEEK") is not None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(os.getenv("SEEK")))

  if W > 1024:
    downscale = 1024.0/W
    F *= downscale
    H = int(H * downscale)
    W = 1024


  #camara intrinsics
  k = np.array([[F, 0, w // 2], [0, F, h // 2], [0, 0, 1]])
  kinv = np.linalg.inv(k)

  display2d = display2d(w, h)

  '''
  mapp.deserialize(open('map.json').read())
  while 1:
    time.sleep(1)
  '''

  i = 1
  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      print("no")
      break
    """
    if you want to write the map to a json file
    i += 1
    if i == 10:
      with open("map.json", "w") as f:
          f.write(mapp.serialize())
          exit(0)
    """







  #reject points without enough "paralax" and reject poinit behind the camera
  #unmatched_pts = np.array([f1.pts[i] is None for i in idx1])
  #np.dot(f1.pose, pts4d)
  #good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_pts