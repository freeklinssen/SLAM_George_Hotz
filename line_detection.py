import cv2
import sdl2
import sdl2.ext
from display import display


w = 1920//2
h = 1080//2
display = display(w,h)

class featureextractor(object):
  def init(self):
    self.orb = cv2.orb_create(100)

  def extract(self, image):
    kps = 1
    return kps


fe = featureextractor()
def process_frame(img):
  img = cv2.resize(img,(w, h))
  print(img.shape)
  display.show(img)
  #cv2.imshow("image", img)

  kp = fe.extract(img)




if __name__ == "__main__":
  cap = cv2.VideoCapture("test_countryroad.mp4")
  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      print("no")
      break;