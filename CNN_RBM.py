from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
from My_functions import weight_variable, bias_variable, conv2d, max_pool_2x2
from CNN import CNN_Computaion

FLAGS = None

def CNN_RBM_Computaion(x_image, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, keep_prob):
    img1, img2, img3, img4, img5, img6, img7, img8 = tf.split(x_image, 8, 0)
    
    img1 = tf.squeeze(img1,0)
    img2 = tf.squeeze(img2,0)
    img3 = tf.squeeze(img3,0)
    img4 = tf.squeeze(img4,0)
    img5 = tf.squeeze(img5,0)
    img6 = tf.squeeze(img6,0)
    img7 = tf.squeeze(img7,0)
    img8 = tf.squeeze(img8,0)

    h_fc1_drop_1 =  CNN_Computaion(img1, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, keep_prob)
    h_fc1_drop_2 =  CNN_Computaion(img2, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, keep_prob)
    h_fc1_drop_3 =  CNN_Computaion(img3, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, keep_prob)
    h_fc1_drop_4 =  CNN_Computaion(img4, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, keep_prob)
    h_fc1_drop_5 =  CNN_Computaion(img5, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, keep_prob)
    h_fc1_drop_6 =  CNN_Computaion(img6, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, keep_prob)
    h_fc1_drop_7 =  CNN_Computaion(img7, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, keep_prob)
    h_fc1_drop_8 =  CNN_Computaion(img8, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1, keep_prob)
    
    h_fc1_drop = tf.concat([h_fc1_drop_1, h_fc1_drop_2, h_fc1_drop_3, h_fc1_drop_4, h_fc1_drop_5, h_fc1_drop_6, h_fc1_drop_7, h_fc1_drop_8],1)
    return h_fc1_drop



def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()  
    _, serialized_example = reader.read(filename_queue)  
    # 读取 serialized_example 的格式  
    features = tf.parse_single_example(  
        serialized_example,  
        features={  
                'image_raw1': tf.FixedLenFeature([], tf.string),
                'image_raw2': tf.FixedLenFeature([], tf.string),
                'image_raw3': tf.FixedLenFeature([], tf.string),
                'image_raw4': tf.FixedLenFeature([], tf.string),
                'image_raw5': tf.FixedLenFeature([], tf.string),
                'image_raw6': tf.FixedLenFeature([], tf.string),
                'image_raw7': tf.FixedLenFeature([], tf.string),
                'image_raw8': tf.FixedLenFeature([], tf.string),
                'image_raw9': tf.FixedLenFeature([], tf.string),
                'image_raw10': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([2],tf.int64),
        })

    image1 = tf.decode_raw(features['image_raw1'], tf.uint8)
    image2 = tf.decode_raw(features['image_raw2'], tf.uint8)
    image3 = tf.decode_raw(features['image_raw3'], tf.uint8)
    image4 = tf.decode_raw(features['image_raw4'], tf.uint8)
    image5 = tf.decode_raw(features['image_raw5'], tf.uint8)
    image6 = tf.decode_raw(features['image_raw6'], tf.uint8)
    image7 = tf.decode_raw(features['image_raw7'], tf.uint8)
    image8 = tf.decode_raw(features['image_raw8'], tf.uint8)
    image9 = tf.decode_raw(features['image_raw9'], tf.uint8)
    image10 = tf.decode_raw(features['image_raw10'], tf.uint8)
    
    image1   = tf.reshape(image1, [31, 31, 3])
    image2   = tf.reshape(image2, [31, 31, 3])
    image3   = tf.reshape(image3, [31, 31, 3])
    image4   = tf.reshape(image4, [31, 31, 3])
    image5   = tf.reshape(image5, [31, 31, 3])
    image6   = tf.reshape(image6, [31, 31, 3])
    image7   = tf.reshape(image7, [31, 31, 3])
    image8   = tf.reshape(image8, [31, 31, 3])
    image9   = tf.reshape(image9, [31, 31, 3])
    image10 = tf.reshape(image10, [31, 31, 3])

    image1 = tf.cast(image1, tf.float32) * (1. / 255) - 0.5
    image2 = tf.cast(image2, tf.float32) * (1. / 255) - 0.5
    image3 = tf.cast(image3, tf.float32) * (1. / 255) - 0.5
    image4 = tf.cast(image4, tf.float32) * (1. / 255) - 0.5
    image5 = tf.cast(image5, tf.float32) * (1. / 255) - 0.5
    image6 = tf.cast(image6, tf.float32) * (1. / 255) - 0.5
    image7 = tf.cast(image7, tf.float32) * (1. / 255) - 0.5
    image8 = tf.cast(image8, tf.float32) * (1. / 255) - 0.5
    image9 = tf.cast(image9, tf.float32) * (1. / 255) - 0.5
    image10 = tf.cast(image10, tf.float32) * (1. / 255) - 0.5
    
    label = tf.cast(features['label'], tf.int64)
    
    return image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, label

def inputs(train_dir, train_file, batch_size, num_epochs):
  if not num_epochs: num_epochs = None
  filename = os.path.join(train_dir, train_file)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    image1, image2, image3, image4, image5, image6, image7,image8, image9, image10, label = read_and_decode(filename_queue)
    
    images1, images2, images3, images4, images5, images6, images7, images8, images9, images10, labels = tf.train.shuffle_batch(
        [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, label], batch_size=batch_size, num_threads=2,capacity=1000 + 3 * batch_size,min_after_dequeue=1000)

    images_right_mouth = bundle(images1, images2)
    images_left_mouth   = bundle(images3, images4)
    images_right_eye = bundle(images5, images6)
    images_left_eye   = bundle(images7, images8)
    images_nose = bundle(images9, images10)
    return images_right_mouth, images_left_mouth, images_right_eye, images_left_eye, images_nose, labels

def bundle(img1, img2):
    
    batch_1 = tf.concat([img1, img2], 3)
    batch_2 = tf.concat([img1, tf.reverse(img2, [2])], 3)
    batch_3 = tf.reverse(batch_2, [2])
    batch_4 = tf.reverse(batch_1, [2])
                        
    batch_5 = tf.concat([img2, img1], 3)
    batch_6 = tf.concat([img2, tf.reverse(img1, [2])], 3)
    batch_7 = tf.reverse(batch_6, [2])
    batch_8 = tf.reverse(batch_5, [2])
    
    batch = tf.stack([batch_1, batch_2, batch_3, batch_4, batch_5, batch_6, batch_7, batch_8], 0)
    return batch

def CNN_RBM(train_dir, train_file, checkpoint_dir):
    sess = tf.InteractiveSession()
    train_right_mouth, train_left_mouth, train_right_eye, train_left_eye, train_nose, train_label = inputs(train_dir, train_file, batch_size=50, num_epochs=1)
    
    x_image_nose  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')    
    x_image_left_mouth  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')
    x_image_right_mouth  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')
    x_image_left_eye  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')
    x_image_right_eye= tf.placeholder(tf.float32, [8,None, 31, 31, 6], name='x')

    # drop out
    keep_prob = tf.placeholder(tf.float32)

    # left mouth part
    # first convolutional layer
    W_conv1_left_mouth = weight_variable([4, 4, 6, 20], 'W_conv1_left_mouth')
    b_conv1_left_mouth = bias_variable([20],'b_conv1_left_mouth')
    # second convolutional layer
    W_conv2_left_mouth = weight_variable([3, 3, 20, 40],'W_conv2_left_mouth')
    b_conv2_left_mouth = bias_variable([40],'b_conv2_left_mouth')
    # third convolutional layer
    W_conv3_left_mouth = weight_variable([3, 3, 40, 60],'W_conv3_left_mouth')
    b_conv3_left_mouth = bias_variable([60],'b_conv3_left_mouth')
    # forth convolutional layer
    W_conv4_left_mouth = weight_variable([2, 2, 60, 80], 'W_conv4_left_mouth')
    b_conv4_left_mouth = bias_variable([80],'b_conv4_left_mouth')
    # densely connected layer
    W_fc1_left_mouth = weight_variable([1*1*80, 80],'W_fc1_left_mouth')
    b_fc1_left_mouth = bias_variable([80],'b_fc1_left_mouth')
    # right mouth part
    # first convolutional layer
    W_conv1_right_mouth = weight_variable([4, 4, 6, 20], 'W_conv1_right_mouth')
    b_conv1_right_mouth = bias_variable([20],'b_conv1_right_mouth')
    # second convolutional layer
    W_conv2_right_mouth = weight_variable([3, 3, 20, 40],'W_conv2_right_mouth')
    b_conv2_right_mouth = bias_variable([40],'b_conv2_right_mouth')
    # third convolutional layer
    W_conv3_right_mouth = weight_variable([3, 3, 40, 60],'W_conv3_right_mouth')
    b_conv3_right_mouth = bias_variable([60],'b_conv3_right_mouth')
    # forth convolutional layer
    W_conv4_right_mouth = weight_variable([2, 2, 60, 80], 'W_conv4_right_mouth')
    b_conv4_right_mouth = bias_variable([80],'b_conv4_right_mouth')
    # densely connected layer
    W_fc1_right_mouth = weight_variable([1*1*80, 80],'W_fc1_right_mouth')
    b_fc1_right_mouth = bias_variable([80],'b_fc1_right_mouth')
    # left eye part
    # first convolutional layer
    W_conv1_left_eye = weight_variable([4, 4, 6, 20], 'W_conv1_left_eye')
    b_conv1_left_eye = bias_variable([20],'b_conv1_left_eye')
    # second convolutional layer
    W_conv2_left_eye = weight_variable([3, 3, 20, 40],'W_conv2_left_eye')
    b_conv2_left_eye = bias_variable([40],'b_conv2_left_eye')
    # third convolutional layer
    W_conv3_left_eye = weight_variable([3, 3, 40, 60],'W_conv3_left_eye')
    b_conv3_left_eye = bias_variable([60],'b_conv3_left_eye')
    # forth convolutional layer
    W_conv4_left_eye = weight_variable([2, 2, 60, 80], 'W_conv4_left_eye')
    b_conv4_left_eye = bias_variable([80],'b_conv4_left_eye')
    # densely connected layer
    W_fc1_left_eye = weight_variable([1*1*80, 80],'W_fc1_left_eye')
    b_fc1_left_eye = bias_variable([80],'b_fc1_left_eye')
    # right eye part
    # first convolutional layer
    W_conv1_right_eye = weight_variable([4, 4, 6, 20], 'W_conv1_right_eye')
    b_conv1_right_eye = bias_variable([20],'b_conv1_right_eye')
    # second convolutional layer
    W_conv2_right_eye = weight_variable([3, 3, 20, 40],'W_conv2_right_eye')
    b_conv2_right_eye = bias_variable([40],'b_conv2_right_eye')
    # third convolutional layer
    W_conv3_right_eye = weight_variable([3, 3, 40, 60],'W_conv3_right_eye')
    b_conv3_right_eye = bias_variable([60],'b_conv3_right_eye')
    # forth convolutional layer
    W_conv4_right_eye = weight_variable([2, 2, 60, 80], 'W_conv4_right_eye')
    b_conv4_right_eye = bias_variable([80],'b_conv4_right_eye')
    # densely connected layer
    W_fc1_right_eye = weight_variable([1*1*80, 80],'W_fc1_right_eye')
    b_fc1_right_eye = bias_variable([80],'b_fc1_right_eye')
    # nose part
    # first convolutional layer
    W_conv1_nose = weight_variable([4, 4, 6, 20], 'W_conv1_nose')
    b_conv1_nose = bias_variable([20],'b_conv1_nose')
    # second convolutional layer
    W_conv2_nose = weight_variable([3, 3, 20, 40],'W_conv2_nose')
    b_conv2_nose = bias_variable([40],'b_conv2_nose')
    # third convolutional layer
    W_conv3_nose = weight_variable([3, 3, 40, 60],'W_conv3_nose')
    b_conv3_nose = bias_variable([60],'b_conv3_nose')
    # forth convolutional layer
    W_conv4_nose = weight_variable([2, 2, 60, 80], 'W_conv4_nose')
    b_conv4_nose = bias_variable([80],'b_conv4_nose')
    # densely connected layer
    W_fc1_nose = weight_variable([1*1*80, 80],'W_fc1_nose')
    b_fc1_nose = bias_variable([80],'b_fc1_nose')
    
    h_fc1_drop_left_mouth = CNN_RBM_Computaion(x_image_left_mouth, W_conv1_left_mouth, b_conv1_left_mouth, W_conv2_left_mouth, b_conv2_left_mouth, W_conv3_left_mouth, b_conv3_left_mouth, W_conv4_left_mouth,
                                             b_conv4_left_mouth,W_fc1_left_mouth, b_fc1_left_mouth, keep_prob)
    h_fc1_drop_right_mouth = CNN_RBM_Computaion(x_image_right_mouth, W_conv1_right_mouth, b_conv1_right_mouth, W_conv2_right_mouth, b_conv2_right_mouth, W_conv3_right_mouth, b_conv3_right_mouth, W_conv4_right_mouth,
                                             b_conv4_right_mouth,W_fc1_right_mouth, b_fc1_right_mouth, keep_prob) 
    h_fc1_drop_left_eye = CNN_RBM_Computaion(x_image_left_eye, W_conv1_left_eye, b_conv1_left_eye, W_conv2_left_eye, b_conv2_left_eye, W_conv3_left_eye, b_conv3_left_eye, W_conv4_left_eye,
                                             b_conv4_left_eye,W_fc1_left_eye, b_fc1_left_eye, keep_prob)
    h_fc1_drop_right_eye = CNN_RBM_Computaion(x_image_right_eye, W_conv1_right_eye, b_conv1_right_eye, W_conv2_right_eye, b_conv2_right_eye, W_conv3_right_eye, b_conv3_right_eye, W_conv4_right_eye,
                                             b_conv4_right_eye,W_fc1_right_eye, b_fc1_right_eye, keep_prob)
    h_fc1_drop_nose = CNN_RBM_Computaion(x_image_nose, W_conv1_nose, b_conv1_nose, W_conv2_nose, b_conv2_nose, W_conv3_nose, b_conv3_nose, W_conv4_nose, b_conv4_nose,W_fc1_nose, b_fc1_nose, keep_prob)
    
    # RBM implentation
    h_fc1_drop = tf.concat([h_fc1_drop_left_mouth, h_fc1_drop_right_mouth, h_fc1_drop_left_eye, h_fc1_drop_right_eye, h_fc1_drop_nose], 1)
    #h_fc1_drop_stop = tf.stop_gradient(h_fc1_drop)
        
    W_hidden = weight_variable([3200, 8],'W_hidden', 0.5/tf.sqrt(3200.0))
    b_hidden = bias_variable([8],'b_hidden')
    h_dc = tf.matmul(h_fc1_drop, W_hidden) + b_hidden
    
    W_output_1 = weight_variable([8],'W_output_1', 0.5/tf.sqrt(8.0))
    W_output_2 = weight_variable([8],'W_output_2', 0.5/tf.sqrt(8.0))
    
    b_output = bias_variable([2],'b_output')
   
    y_ = tf.placeholder(tf.float32, [None, 2], name='y')
    Probability_part1 = tf.exp(h_dc+ W_output_1) +1
    Probability_part2 = tf.exp(h_dc+ W_output_2) +1
    Probability_part3 =tf.transpose(tf.stack([Probability_part1, Probability_part2] , 2),perm=[1, 0, 2])
    
    Probability_part4 = tf.reduce_sum(tf.multiply(Probability_part3,y_), 2, keep_dims=True)
    Probability_numerator = tf.reduce_sum(tf.multiply(y_, tf.exp(b_output)),1)
    Probability_denominator = tf.reduce_sum(tf.multiply(tf.reduce_prod(tf.div(Probability_part3, Probability_part4), 0), tf.exp(b_output)),1) 
    Probability_true = tf.log(tf.div(Probability_numerator, Probability_denominator))

    Probability_part5 = tf.reduce_sum(tf.multiply(Probability_part3, [1.0, 0.0]), 2, keep_dims=True)
    Probability_pos_numerator = tf.reduce_sum(tf.multiply([1.0, 0.0], tf.exp(b_output)))
    Probability_pos_denominator = tf.reduce_sum(tf.multiply(tf.reduce_prod(tf.div(Probability_part3, Probability_part5), 0), tf.exp(b_output)),1)
    Probability_pos = tf.div(Probability_pos_numerator, Probability_pos_denominator)

    Probability_part6 = tf.reduce_sum(tf.multiply(Probability_part3, [0.0, 1.0]), 2, keep_dims=True)
    Probability_neg_numerator = tf.reduce_sum(tf.multiply([0.0, 1.0], tf.exp(b_output)))
    Probability_neg_denominator = tf.reduce_sum(tf.multiply(tf.reduce_prod(tf.div(Probability_part3, Probability_part6), 0), tf.exp(b_output)),1)
    Probability_neg = tf.div(Probability_neg_numerator, Probability_neg_denominator)
    
    Probability_dis =  tf.stack([Probability_pos, Probability_neg] , 1)
    loss = -tf.reduce_mean(Probability_true)
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
    check_op = tf.add_check_numerics_ops()
    correct_prediction = tf.equal(tf.argmax(Probability_dis,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # set check point
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # parameter restore
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print ('restore checkpoint data')
        print (ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

##    saver2 = tf.train.Saver({'W_hidden': W_hidden, 'b_hidden': b_hidden, 'W_output_1': W_output_1, 'W_output_2': W_output_2,'b_output': b_output})
##    checkpoint_file = os.path.join('C:/Users/bear/Face_verification/checkpoint/RBM/partial', 'save_net_RBM.ckpt')
##    print (checkpoint_file)
##    save_path = saver2.save(sess, checkpoint_file)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        step = 0
        while not coord.should_stop():
            right_mouth_array, left_mouth_array, right_eye_array, left_eye_array, nose_array, labels_array = sess.run([train_right_mouth, train_left_mouth, train_right_eye, train_left_eye, train_nose, train_label])
            sess.run([train_step,check_op], feed_dict={x_image_nose: nose_array, x_image_left_eye : left_eye_array, x_image_right_eye : right_eye_array, x_image_left_mouth : left_mouth_array, x_image_right_mouth : right_mouth_array,
                                                    y_ : labels_array,  keep_prob: 0.5})
            # Print an overview fairly often.
            if step % 1000 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x_image_nose: nose_array, x_image_left_eye : left_eye_array, x_image_right_eye : right_eye_array, x_image_left_mouth : left_mouth_array, x_image_right_mouth : right_mouth_array,
                                                    y_ : labels_array,  keep_prob: 1})
                print('Step %d: accuracy = %.2f ' % (step, train_accuracy))
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training for  %d steps.' % (step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    checkpoint_file = os.path.join(checkpoint_dir, 'save_net_RBM.ckpt')
    save_path = saver.save(sess, checkpoint_file)
    sess.close()
    
def Accuracy_Eval(valid_dir, valid_file, check_dir):
    
    sess = tf.InteractiveSession()
    train_right_mouth, train_left_mouth, train_right_eye, train_left_eye, train_nose, train_label = inputs(valid_dir, valid_file, batch_size=300, num_epochs=1)
    
    x_image_nose  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')    
    x_image_left_mouth  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')
    x_image_right_mouth  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')
    x_image_left_eye  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')
    x_image_right_eye= tf.placeholder(tf.float32, [8,None, 31, 31, 6], name='x')

    # drop out
    keep_prob = tf.placeholder(tf.float32)

    # left mouth part
    # first convolutional layer
    W_conv1_left_mouth = weight_variable([4, 4, 6, 20], 'W_conv1_left_mouth')
    b_conv1_left_mouth = bias_variable([20],'b_conv1_left_mouth')
    # second convolutional layer
    W_conv2_left_mouth = weight_variable([3, 3, 20, 40],'W_conv2_left_mouth')
    b_conv2_left_mouth = bias_variable([40],'b_conv2_left_mouth')
    # third convolutional layer
    W_conv3_left_mouth = weight_variable([3, 3, 40, 60],'W_conv3_left_mouth')
    b_conv3_left_mouth = bias_variable([60],'b_conv3_left_mouth')
    # forth convolutional layer
    W_conv4_left_mouth = weight_variable([2, 2, 60, 80], 'W_conv4_left_mouth')
    b_conv4_left_mouth = bias_variable([80],'b_conv4_left_mouth')
    # densely connected layer
    W_fc1_left_mouth = weight_variable([1*1*80, 80],'W_fc1_left_mouth')
    b_fc1_left_mouth = bias_variable([80],'b_fc1_left_mouth')
    # right mouth part
    # first convolutional layer
    W_conv1_right_mouth = weight_variable([4, 4, 6, 20], 'W_conv1_right_mouth')
    b_conv1_right_mouth = bias_variable([20],'b_conv1_right_mouth')
    # second convolutional layer
    W_conv2_right_mouth = weight_variable([3, 3, 20, 40],'W_conv2_right_mouth')
    b_conv2_right_mouth = bias_variable([40],'b_conv2_right_mouth')
    # third convolutional layer
    W_conv3_right_mouth = weight_variable([3, 3, 40, 60],'W_conv3_right_mouth')
    b_conv3_right_mouth = bias_variable([60],'b_conv3_right_mouth')
    # forth convolutional layer
    W_conv4_right_mouth = weight_variable([2, 2, 60, 80], 'W_conv4_right_mouth')
    b_conv4_right_mouth = bias_variable([80],'b_conv4_right_mouth')
    # densely connected layer
    W_fc1_right_mouth = weight_variable([1*1*80, 80],'W_fc1_right_mouth')
    b_fc1_right_mouth = bias_variable([80],'b_fc1_right_mouth')
    # left eye part
    # first convolutional layer
    W_conv1_left_eye = weight_variable([4, 4, 6, 20], 'W_conv1_left_eye')
    b_conv1_left_eye = bias_variable([20],'b_conv1_left_eye')
    # second convolutional layer
    W_conv2_left_eye = weight_variable([3, 3, 20, 40],'W_conv2_left_eye')
    b_conv2_left_eye = bias_variable([40],'b_conv2_left_eye')
    # third convolutional layer
    W_conv3_left_eye = weight_variable([3, 3, 40, 60],'W_conv3_left_eye')
    b_conv3_left_eye = bias_variable([60],'b_conv3_left_eye')
    # forth convolutional layer
    W_conv4_left_eye = weight_variable([2, 2, 60, 80], 'W_conv4_left_eye')
    b_conv4_left_eye = bias_variable([80],'b_conv4_left_eye')
    # densely connected layer
    W_fc1_left_eye = weight_variable([1*1*80, 80],'W_fc1_left_eye')
    b_fc1_left_eye = bias_variable([80],'b_fc1_left_eye')
    # right eye part
    # first convolutional layer
    W_conv1_right_eye = weight_variable([4, 4, 6, 20], 'W_conv1_right_eye')
    b_conv1_right_eye = bias_variable([20],'b_conv1_right_eye')
    # second convolutional layer
    W_conv2_right_eye = weight_variable([3, 3, 20, 40],'W_conv2_right_eye')
    b_conv2_right_eye = bias_variable([40],'b_conv2_right_eye')
    # third convolutional layer
    W_conv3_right_eye = weight_variable([3, 3, 40, 60],'W_conv3_right_eye')
    b_conv3_right_eye = bias_variable([60],'b_conv3_right_eye')
    # forth convolutional layer
    W_conv4_right_eye = weight_variable([2, 2, 60, 80], 'W_conv4_right_eye')
    b_conv4_right_eye = bias_variable([80],'b_conv4_right_eye')
    # densely connected layer
    W_fc1_right_eye = weight_variable([1*1*80, 80],'W_fc1_right_eye')
    b_fc1_right_eye = bias_variable([80],'b_fc1_right_eye')
    # nose part
    # first convolutional layer
    W_conv1_nose = weight_variable([4, 4, 6, 20], 'W_conv1_nose')
    b_conv1_nose = bias_variable([20],'b_conv1_nose')
    # second convolutional layer
    W_conv2_nose = weight_variable([3, 3, 20, 40],'W_conv2_nose')
    b_conv2_nose = bias_variable([40],'b_conv2_nose')
    # third convolutional layer
    W_conv3_nose = weight_variable([3, 3, 40, 60],'W_conv3_nose')
    b_conv3_nose = bias_variable([60],'b_conv3_nose')
    # forth convolutional layer
    W_conv4_nose = weight_variable([2, 2, 60, 80], 'W_conv4_nose')
    b_conv4_nose = bias_variable([80],'b_conv4_nose')
    # densely connected layer
    W_fc1_nose = weight_variable([1*1*80, 80],'W_fc1_nose')
    b_fc1_nose = bias_variable([80],'b_fc1_nose')
    
    h_fc1_drop_left_mouth = CNN_RBM_Computaion(x_image_left_mouth, W_conv1_left_mouth, b_conv1_left_mouth, W_conv2_left_mouth, b_conv2_left_mouth, W_conv3_left_mouth, b_conv3_left_mouth, W_conv4_left_mouth,
                                             b_conv4_left_mouth,W_fc1_left_mouth, b_fc1_left_mouth, keep_prob)
    h_fc1_drop_right_mouth = CNN_RBM_Computaion(x_image_right_mouth, W_conv1_right_mouth, b_conv1_right_mouth, W_conv2_right_mouth, b_conv2_right_mouth, W_conv3_right_mouth, b_conv3_right_mouth, W_conv4_right_mouth,
                                             b_conv4_right_mouth,W_fc1_right_mouth, b_fc1_right_mouth, keep_prob)
    h_fc1_drop_left_eye = CNN_RBM_Computaion(x_image_left_eye, W_conv1_left_eye, b_conv1_left_eye, W_conv2_left_eye, b_conv2_left_eye, W_conv3_left_eye, b_conv3_left_eye, W_conv4_left_eye,
                                             b_conv4_left_eye,W_fc1_left_eye, b_fc1_left_eye, keep_prob)
    h_fc1_drop_right_eye = CNN_RBM_Computaion(x_image_right_eye, W_conv1_right_eye, b_conv1_right_eye, W_conv2_right_eye, b_conv2_right_eye, W_conv3_right_eye, b_conv3_right_eye, W_conv4_right_eye,
                                             b_conv4_right_eye,W_fc1_right_eye, b_fc1_right_eye, keep_prob)
    h_fc1_drop_nose = CNN_RBM_Computaion(x_image_nose, W_conv1_nose, b_conv1_nose, W_conv2_nose, b_conv2_nose, W_conv3_nose, b_conv3_nose, W_conv4_nose, b_conv4_nose,W_fc1_nose, b_fc1_nose, keep_prob)
    
    # RBM implentation
    h_fc1_drop = tf.concat([h_fc1_drop_left_mouth, h_fc1_drop_right_mouth, h_fc1_drop_left_eye, h_fc1_drop_right_eye, h_fc1_drop_nose], 1)
    #h_fc1_drop_stop = tf.stop_gradient(h_fc1_drop)
        
    W_hidden = weight_variable([3200, 8],'W_hidden', 0.5/tf.sqrt(3200.0))
    b_hidden = bias_variable([8],'b_hidden')
    h_dc = tf.matmul(h_fc1_drop, W_hidden) + b_hidden
    
    W_output_1 = weight_variable([8],'W_output_1', 0.5/tf.sqrt(8.0))
    W_output_2 = weight_variable([8],'W_output_2', 0.5/tf.sqrt(8.0))
    
    b_output = bias_variable([2],'b_output')

    y_ = tf.placeholder(tf.float32, [None, 2], name='y')
    Probability_part1 = tf.exp(h_dc+ W_output_1) +1
    Probability_part2 = tf.exp(h_dc+ W_output_2) +1
    Probability_part3 =tf.transpose(tf.stack([Probability_part1, Probability_part2] , 2),perm=[1, 0, 2])
    
    Probability_part5 = tf.reduce_sum(tf.multiply(Probability_part3, [1.0, 0.0]), 2, keep_dims=True)
    Probability_pos_numerator = tf.reduce_sum(tf.multiply([1.0, 0.0], tf.exp(b_output)))
    Probability_pos_denominator = tf.reduce_sum(tf.multiply(tf.reduce_prod(tf.div(Probability_part3, Probability_part5), 0), tf.exp(b_output)),1)
    Probability_pos = tf.div(Probability_pos_numerator, Probability_pos_denominator)

    Probability_part6 = tf.reduce_sum(tf.multiply(Probability_part3, [0.0, 1.0]), 2, keep_dims=True)
    Probability_neg_numerator = tf.reduce_sum(tf.multiply([0.0, 1.0], tf.exp(b_output)))
    Probability_neg_denominator = tf.reduce_sum(tf.multiply(tf.reduce_prod(tf.div(Probability_part3, Probability_part6), 0), tf.exp(b_output)),1)
    Probability_neg = tf.div(Probability_neg_numerator, Probability_neg_denominator)
    
    Probability_dis =  tf.stack([Probability_pos, Probability_neg] , 1)
    check_op = tf.add_check_numerics_ops()
    correct_prediction = tf.equal(tf.argmax(Probability_dis,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # set check point
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(check_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print ('restore checkpoint data')
        saver.restore(sess, ckpt.model_checkpoint_path)
       
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    valid_accuracy = 0.0    
    try:
        step = 0
        while not coord.should_stop():
            right_mouth_array, left_mouth_array, right_eye_array, left_eye_array, nose_array, labels_array = sess.run([train_right_mouth, train_left_mouth, train_right_eye, train_left_eye, train_nose, train_label])
            valid_accuracy = valid_accuracy + sess.run(accuracy, feed_dict={x_image_nose: nose_array, x_image_left_eye : left_eye_array, x_image_right_eye : right_eye_array, x_image_left_mouth : left_mouth_array, x_image_right_mouth : right_mouth_array,
                                                    y_ : labels_array,  keep_prob: 1})
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

def main(data_dir, train_file, valid_file, test_file, check_dir, epoches):
   for epoch in range(epoches):
      tf.reset_default_graph()
      CNN_RBM(data_dir, train_file, check_dir)
      tf.reset_default_graph()
      valid_accuracy = Accuracy_Eval(data_dir, valid_file, check_dir)
      print ('valid accuracy:', valid_accuracy)

   tf.reset_default_graph()
   test_accuracy = Accuracy_Eval(data_dir, test_file, check_dir)
   print ('test accuracy:', test_accuracy)
   
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='C:/Users/bear/data/', help='Directory for storing input data')
  parser.add_argument('--train_file', type=str, default='RBM_train_set.tfrecords', help='name for storing train data')
  parser.add_argument('--valid_file', type=str, default='RBM_valid_set.tfrecords', help='name for storing valid data')
  parser.add_argument('--test_file', type=str, default='RBM_test_set.tfrecords', help='name for storing test data')
  parser.add_argument('--check_dir', type=str, default='C:/Users/bear/Face_verification/checkpoint/RBM/finetune', help='path for storing checkpoint file')
  parser.add_argument('--epoches', type=str, default=6, help='number of epoches to run training')
  FLAGS, unparsed = parser.parse_known_args()
  main(FLAGS.data_dir, FLAGS.train_file, FLAGS.valid_file, FLAGS.test_file, FLAGS.check_dir, FLAGS.epoches)

