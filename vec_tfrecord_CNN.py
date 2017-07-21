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
    x1, x2, y = [], [], []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            file1, img1, file2, img2, label1, label2 = line.strip().split() 
            
            img1 = 'rgb_large_right_mouth_' + img1
            img2 = 'rgb_large_right_mouth_' + img2
            
            p1 = os.path.join('C:/Users/bear/cropwebface', file1, img1)
            p2 = os.path.join('C:/Users/bear/cropwebface', file2, img2)
            
            x1.append(p1)
            x2.append(p2)
            
            y.append([label1, label2])
            
    return x1, x2, np.asarray(y, dtype='int32')

def read_images_from_disk(input_queue):
    img_file1 = tf.read_file(input_queue[0])
    img_file2 = tf.read_file(input_queue[1])
    label = input_queue[2]
    image1 = tf.image.decode_jpeg(img_file1)
    image2 = tf.image.decode_jpeg(img_file2)
    return image1, image2, label

def load_data(filenames, tfrecord_file):
    img1, img2, labels = read_csv_pair_file(filenames)
##    img1 = img1[0:4000]
##    img2 = img2[0:4000]
##    labels = labels[0:4000,:]
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    batch_size = 100
    length = len(img1)
    address = 0
    number = 0
    while address < length:
      if (address + batch_size) < length:
        img1_part = img1[address:address+batch_size]
        img2_part = img2[address:address+batch_size]
        labels_part = labels[address:address+batch_size,:]
      else :
        img1_part = img1[address:]
        img2_part = img2[address:]
        labels_part = labels[address:,:]
      address = address + batch_size
      tf.reset_default_graph()
      sess = tf.Session()
      filename_queue = tf.train.slice_input_producer([img1_part, img2_part, labels_part], num_epochs=1, shuffle=True)
      init_op = tf.global_variables_initializer()
      init_again = tf.local_variables_initializer()
      sess.run(init_op)
      sess.run(init_again)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      try:
          while not coord.should_stop():
            image1, image2, label = read_images_from_disk(filename_queue)
            image1_tensor, image2_tensor, label_tensor = sess.run([image1, image2, label])
            image1_raw = image1_tensor.tostring()
            image2_raw = image2_tensor.tostring()
            examples  = tf.train.Example(features=tf.train.Features(feature={
                'image_raw1': _bytes_feature(image1_raw),
                'image_raw2': _bytes_feature(image2_raw),
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
    load_data('C:/Users/bear/cropwebface/train_set_27.csv', 'C:/Users/bear/data/train_set_right_mouth.tfrecords')
    #load_data('C:/Users/bear/cropwebface/valid_set_4.csv', 'D:/deep_mouth/cropwebface/valid_set_4.tfrecords')
    #load_data('C:/Users/bear/cropwebface/train_set_4.csv', 'D:/deep_mouth/cropwebface/train_set_4.tfrecords')
