from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
from My_functions import weight_variable, bias_variable, conv2d, max_pool_2x2
#from vec import load_data

FLAGS = None

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()  
    _, serialized_example = reader.read(filename_queue)  
    # 读取 serialized_example 的格式  
    features = tf.parse_single_example(  
        serialized_example,  
        features={  
                'image_raw1': tf.FixedLenFeature([], tf.string),
                'image_raw2': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([2],tf.int64),
        })

    image1 = tf.decode_raw(features['image_raw1'], tf.uint8)
    image2 = tf.decode_raw(features['image_raw2'], tf.uint8)
    
    image1 = tf.reshape(image1, [31, 31, 3])
    image2 = tf.reshape(image2, [31, 31, 3])

    image1 = tf.cast(image1, tf.float32) * (1. / 255) - 0.5
    image2 = tf.cast(image2, tf.float32) * (1. / 255) - 0.5
    
    label = tf.cast(features['label'], tf.int64)
    
    return image1, image2, label

def inputs(train_dir, train_file, batch_size, num_epochs):
  if not num_epochs: num_epochs = None
  filename = os.path.join(train_dir, train_file)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    # Even when reading in multiple threads, share the filename queue.
    image1, image2, label = read_and_decode(filename_queue)
    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images1, images2, labels = tf.train.shuffle_batch([image1, image2, label], batch_size=batch_size, num_threads=2,capacity=1000 + 3 * batch_size,min_after_dequeue=1000)
        # Ensures a minimum amount of shuffling of examples.
    return images1, images2, labels

def CNN_Computaion(x_image, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, keep_prob):
    # first convolutional layer
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # second convolutional layer
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # third convolutional layer
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    # forth convolutional layer
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    # densely connected layer
    h_pool2_flat = tf.reshape(h_conv4 , [-1, 1*1*80])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # drop out
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    return h_fc1_drop

def My_CNN(train_dir, train_file, checkpoint_dir, part):
    checkpoint2_dir = os.path.join(checkpoint_dir, 'withoutput')
   
    sess = tf.InteractiveSession()
    
    images1, images2, labels = inputs(train_dir, train_file, batch_size=50, num_epochs=1)
    images = tf.concat([images1, images2], 3)
    
    x_image  = tf.placeholder(tf.float32, [None, 31, 31, 6], name='x')
    # first convolutional layer
    W_conv1 = weight_variable([4, 4, 6, 20], 'W_conv1_' + part)
    b_conv1 = bias_variable([20],'b_conv1_' + part)
    # second convolutional layer
    W_conv2 = weight_variable([3, 3, 20, 40],'W_conv2_' + part)
    b_conv2 = bias_variable([40],'b_conv2_' + part)
    # third convolutional layer
    W_conv3 = weight_variable([3, 3, 40, 60],'W_conv3_' + part)
    b_conv3 = bias_variable([60],'b_conv3_' + part)
    # forth convolutional layer
    W_conv4 = weight_variable([2, 2, 60, 80], 'W_conv4_' + part)
    b_conv4 = bias_variable([80],'b_conv4_' + part)
    # densely connected layer
    W_fc1 = weight_variable([1*1*80, 80],'W_fc1_' + part)
    b_fc1 = bias_variable([80],'b_fc1_' + part)
    # drop out
    keep_prob = tf.placeholder(tf.float32)
    # readout layer
    h_fc1_drop = CNN_Computaion(x_image, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4,W_fc1, b_fc1, keep_prob)
    # set check point
    saver = tf.train.Saver({'W_conv1_' + part: W_conv1, 'b_conv1_' + part: b_conv1, 'W_conv2_' + part: W_conv2, 'b_conv2_' + part: b_conv2,'W_conv3_' + part: W_conv3,
                            'b_conv3_' + part: b_conv3, 'W_conv4_' + part: W_conv4, 'b_conv4_' + part: b_conv4,'W_fc1_' + part: W_fc1, 'b_fc1_' + part:b_fc1})
    
    W_fc2 = weight_variable([80, 2],'W_fc2')
    b_fc2 = bias_variable([2], 'b_fc2')
    saver2 = tf.train.Saver()
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2], name='y')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state(checkpoint2_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)  
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        step = 0
        while not coord.should_stop():
            images_array, labels_array = sess.run([images, labels])
            _, train_accuracy = sess.run([train_step,  accuracy], feed_dict={x_image : images_array, y_ : labels_array,  keep_prob: 0.5})
            # Print an overview fairly often.
            if step % 1000 == 0:
                print('Step %d: accuracy = %.2f ' % (step, train_accuracy))
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training for  %d steps.' % (step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    checkpoint1_dir = os.path.join(checkpoint_dir, 'nooutput', 'save_net_nose.ckpt')
    checkpoint2_dir = os.path.join(checkpoint_dir, 'withoutput', 'save_net_nose.ckpt')
    save_path = saver.save(sess, checkpoint1_dir)
    save_path = saver2.save(sess, checkpoint2_dir) 
    sess.close()

def Accuracy_Eval(valid_dir, valid_file, check_dir, part):
    sess = tf.InteractiveSession()
    #train_dir = 'C:/Users/bear/data/'
    images1, images2, labels = inputs(valid_dir, valid_file, batch_size=5400, num_epochs=1)
    images = tf.concat([images1, images2], 3)
    
    x_image  = tf.placeholder(tf.float32, [None, 31, 31, 6], name='x')
    # first convolutional layer
    W_conv1 = weight_variable([4, 4, 6, 20], 'W_conv1_' + part)
    b_conv1 = bias_variable([20],'b_conv1_' + part)
    # second convolutional layer
    W_conv2 = weight_variable([3, 3, 20, 40],'W_conv2_' + part)
    b_conv2 = bias_variable([40],'b_conv2_' + part)
    # third convolutional layer
    W_conv3 = weight_variable([3, 3, 40, 60],'W_conv3_' + part)
    b_conv3 = bias_variable([60],'b_conv3_' + part)
    # forth convolutional layer
    W_conv4 = weight_variable([2, 2, 60, 80], 'W_conv4_' + part)
    b_conv4 = bias_variable([80],'b_conv4_' + part)
    # densely connected layer
    W_fc1 = weight_variable([1*1*80, 80],'W_fc1_' + part)
    b_fc1 = bias_variable([80],'b_fc1_' + part)
    # drop out
    keep_prob = tf.placeholder(tf.float32)
    # readout layer
    h_fc1_drop = CNN_Computaion(x_image, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, keep_prob)
    # set check point
    W_fc2 = weight_variable([80, 2],'W_fc2')
    b_fc2 = bias_variable([2], 'b_fc2')
    saver = tf.train.Saver()
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2], name='y')
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state(check_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    valid_accuracy = 0.0
    try:
        step = 0.0
        while not coord.should_stop():
            images_array, labels_array = sess.run([images, labels])
            valid_accuracy = valid_accuracy + sess.run(accuracy, feed_dict={x_image : images_array, y_ : labels_array,  keep_prob: 1})
            step += 1
    except tf.errors.OutOfRangeError:
        pass
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    valid_accuracy = valid_accuracy / step
    return valid_accuracy

def main(data_dir, train_file, valid_file, test_file, check_dir, part, epoches):
   checkpoint1_dir  = os.path.join(check_dir, part)
   checkpoint2_dir  = os.path.join(checkpoint1_dir, 'withoutput')
   for epoch in range(epoches):
      tf.reset_default_graph()
      My_CNN(data_dir, train_file, checkpoint1_dir, part)
      tf.reset_default_graph()
      valid_accuracy = Accuracy_Eval(data_dir, valid_file, checkpoint2_dir, part)
      print ('valid accuracy:', valid_accuracy)

   tf.reset_default_graph()
   test_accuracy = Accuracy_Eval(data_dir, test_file, checkpoint2_dir, part)
   print ('test accuracy:', test_accuracy)
   
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='C:/Users/bear/data/', help='Directory for storing input data')
  parser.add_argument('--train_file', type=str, default='train_set_nose.tfrecords', help='name for storing train data')
  parser.add_argument('--valid_file', type=str, default='valid_set_nose.tfrecords', help='name for storing valid data')
  parser.add_argument('--test_file', type=str, default='test_set_nose.tfrecords', help='name for storing test data')
  parser.add_argument('--check_dir', type=str, default='C:/Users/bear/Face_verification/checkpoint', help='path for storing checkpoint file')
  parser.add_argument('--part', type=str, default='nose', help='which part of the face')
  parser.add_argument('--epoches', type=str, default=15, help='number of epoches to run training')
  FLAGS, unparsed = parser.parse_known_args()
  main(FLAGS.data_dir, FLAGS.train_file, FLAGS.valid_file, FLAGS.test_file, FLAGS.check_dir, FLAGS.part, FLAGS.epoches)



