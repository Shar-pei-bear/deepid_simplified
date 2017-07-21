import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
import csv

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_csv_pair_file(csv_file):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y = [], [], [], [], [], [], [], [], [], [], []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            file1, image1, file2, image2, label1, label2 = line.strip().split() 
            
            img1   = 'rgb_large_right_mouth_' + image1
            img2   = 'rgb_large_right_mouth_' + image2
            img3   = 'rgb_large_left_mouth_'   + image1
            img4   = 'rgb_large_left_mouth_'   + image2
            img5   = 'rgb_large_right_eye_'      + image1
            img6   = 'rgb_large_right_eye_'      + image2
            img7   = 'rgb_large_left_eye_'        + image1
            img8   = 'rgb_large_left_eye_'        + image2
            img9   = 'rgb_large_nose_point_'   + image1
            img10 = 'rgb_large_nose_point_'   + image2
            
            p1   = os.path.join('C:/Users/bear/cropwebface', file1, img1)
            p2   = os.path.join('C:/Users/bear/cropwebface', file2, img2)
            p3   = os.path.join('C:/Users/bear/cropwebface', file1, img3)
            p4   = os.path.join('C:/Users/bear/cropwebface', file2, img4)
            p5   = os.path.join('C:/Users/bear/cropwebface', file1, img5)
            p6   = os.path.join('C:/Users/bear/cropwebface', file2, img6)
            p7   = os.path.join('C:/Users/bear/cropwebface', file1, img7)
            p8   = os.path.join('C:/Users/bear/cropwebface', file2, img8)
            p9   = os.path.join('C:/Users/bear/cropwebface', file1, img9)
            p10 = os.path.join('C:/Users/bear/cropwebface', file2, img10)
            
            x1.append(p1)
            x2.append(p2)
            x3.append(p3)
            x4.append(p4)
            x5.append(p5)
            x6.append(p6)
            x7.append(p7)
            x8.append(p8)
            x9.append(p9)
            x10.append(p10)
            
            y.append([label1, label2])
            
    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, np.asarray(y, dtype='int32')

def read_images_from_disk(input_queue):
    img_file1   = tf.read_file(input_queue[0])
    img_file2   = tf.read_file(input_queue[1])
    img_file3   = tf.read_file(input_queue[2])
    img_file4   = tf.read_file(input_queue[3])
    img_file5   = tf.read_file(input_queue[4])
    img_file6   = tf.read_file(input_queue[5])
    img_file7   = tf.read_file(input_queue[6])
    img_file8   = tf.read_file(input_queue[7])
    img_file9   = tf.read_file(input_queue[8])
    img_file10 = tf.read_file(input_queue[9])
    
    label = input_queue[10]
    
    image1   = tf.image.decode_jpeg(img_file1)
    image2   = tf.image.decode_jpeg(img_file2)
    image3   = tf.image.decode_jpeg(img_file3)
    image4   = tf.image.decode_jpeg(img_file4)
    image5   = tf.image.decode_jpeg(img_file5)
    image6   = tf.image.decode_jpeg(img_file6)
    image7   = tf.image.decode_jpeg(img_file7)
    image8   = tf.image.decode_jpeg(img_file8)
    image9   = tf.image.decode_jpeg(img_file9)
    image10 = tf.image.decode_jpeg(img_file10)

    return image1, image2,  image3, image4,  image5, image6,  image7, image8,  image9, image10, label

def load_data(filenames, tfrecord_file):
    img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, labels = read_csv_pair_file(filenames)
    img1 = img1[0:126000]
    img2 = img2[0:126000]
    img3 = img3[0:126000]
    img4 = img4[0:126000]
    img5 = img5[0:126000]
    img6 = img6[0:126000]
    img7 = img7[0:126000]
    img8 = img8[0:126000]
    img9 = img9[0:126000]
    img10 = img10[0:126000]
    labels = labels[0:126000,:]
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    batch_size = 25
    length = len(img1)
    address = 0
    number = 0
    while address < length:
      if (address + batch_size) < length:
        img1_part   = img1[address:address+batch_size]
        img2_part   = img2[address:address+batch_size]
        img3_part   = img3[address:address+batch_size]
        img4_part   = img4[address:address+batch_size]
        img5_part   = img5[address:address+batch_size]
        img6_part   = img6[address:address+batch_size]
        img7_part   = img7[address:address+batch_size]
        img8_part   = img8[address:address+batch_size]
        img9_part   = img9[address:address+batch_size]
        img10_part = img10[address:address+batch_size]
        labels_part = labels[address:address+batch_size,:]
      else :
        img1_part   = img1[address:]
        img2_part   = img2[address:]
        img3_part   = img3[address:]
        img4_part   = img4[address:]
        img5_part   = img5[address:]
        img6_part   = img6[address:]
        img7_part   = img7[address:]
        img8_part   = img8[address:]
        img9_part   = img9[address:]
        img10_part = img10[address:]
        labels_part = labels[address:,:]
      address = address + batch_size
      tf.reset_default_graph()
      sess = tf.Session()
      filename_queue = tf.train.slice_input_producer([img1_part, img2_part, img3_part, img4_part, img5_part, img6_part, img7_part, img8_part, img9_part, img10_part, labels_part], num_epochs=1, shuffle=True)
      init_op = tf.global_variables_initializer()
      init_again = tf.local_variables_initializer()
      sess.run(init_op)
      sess.run(init_again)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      try:
          while not coord.should_stop():
            image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, label = read_images_from_disk(filename_queue)
            image1_tensor, image2_tensor,  image3_tensor, image4_tensor, image5_tensor, image6_tensor, image7_tensor, image8_tensor, image9_tensor, image10_tensor,label_tensor = sess.run([image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, label])

            image1_raw = image1_tensor.tostring()
            image2_raw = image2_tensor.tostring()
            image3_raw = image3_tensor.tostring()
            image4_raw = image4_tensor.tostring()
            image5_raw = image5_tensor.tostring()
            image6_raw = image6_tensor.tostring()
            image7_raw = image7_tensor.tostring()
            image8_raw = image8_tensor.tostring()
            image9_raw = image9_tensor.tostring()
            image10_raw = image10_tensor.tostring()
            
            examples  = tf.train.Example(features=tf.train.Features(feature={
                'image_raw1': _bytes_feature(image1_raw),
                'image_raw2': _bytes_feature(image2_raw),
                'image_raw3': _bytes_feature(image3_raw),
                'image_raw4': _bytes_feature(image4_raw),
                'image_raw5': _bytes_feature(image5_raw),
                'image_raw6': _bytes_feature(image6_raw),
                'image_raw7': _bytes_feature(image7_raw),
                'image_raw8': _bytes_feature(image8_raw),
                'image_raw9': _bytes_feature(image9_raw),
                'image_raw10': _bytes_feature(image10_raw),
                'label': _int64_feature(label_tensor),}))
            writer.write(examples.SerializeToString())
            number +=1
      except tf.errors.OutOfRangeError:
          print("Done writing ", address, ' data')
      finally:
          coord.request_stop()
      coord.join(threads)
      
    writer.close()
    sess.close()

if __name__ == '__main__':
    load_data('C:/Users/bear/cropwebface/test_set_15.csv', 'C:/Users/bear/data/RBM_test_set.tfrecords')
    #load_data('C:/Users/bear/cropwebface/valid_set_4.csv', 'D:/deep_mouth/cropwebface/valid_set_4.tfrecords')
    #load_data('C:/Users/bear/cropwebface/train_set_4.csv', 'D:/deep_mouth/cropwebface/train_set_4.tfrecords')
