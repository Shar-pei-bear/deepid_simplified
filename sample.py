import argparse
from interface import FaceVerification

def test(path1, path2):
    Y = FaceVerification(path1, path2)
    print (Y)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--img_path1', type=str, default='./sample_image/003.jpg', help='Directory for storing face 1')
  parser.add_argument('--img_path2', type=str, default='./sample_image/006.jpg', help='Directory for storing face 2')
  FLAGS, unparsed = parser.parse_known_args()
  test(FLAGS.img_path1, FLAGS.img_path2)
