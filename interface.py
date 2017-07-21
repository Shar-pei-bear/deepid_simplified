import tensorflow as tf
import numpy as np
from PIL import Image
import dlib
import argparse
FLAGS = None

def get_landmarks(im):
    detector = dlib.get_frontal_face_detector()
    rects = detector(im, 1)
    predictor_path = "./data/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def img_seg(im, location, length):
    
   dx = length
   dy = length
   
   if location[0] <0:
       location[0] = 0;
   if location[1] <0:
       location[1] = 0;
   if  location[0]  >= im.shape[1]:
       location[0] = im.shape[1] - 1;
   if  location[1]  >= im.shape[0]:
       location[1] = im.shape[0] - 1; 

   if (location[1] - dx) < 0:
        dx = location[1]
   if  (location[1] + dx) > im.shape[0]:
        dx = im.shape[0] - location[1]
   if (location[0] - dy) < 0:
        dy = location[0]
   if  (location[0] + dy) > im.shape[1]:
        dy = im.shape[1] - location[0]
        
   im = im[(location[1] - dx) : (location[1] + dx) , (location[0] - dy) : (location[0]+dy)]
   im = Image.fromarray(np.uint8(im))
   im = im.resize((31,31))
   im =  np.asarray(im).astype(np.float32)
   im.flags.writeable = True
   im = im *  (1. / 255) - 0.5
   return np.expand_dims(im, axis=0)

def face_region(img_path):
    sess = tf.Session()
    image_file = tf.read_file(img_path)
    image = tf.image.decode_jpeg(image_file)
    img = sess.run(image)
    sess.close()
    detector = dlib.get_frontal_face_detector()
    dets = detector(img,1)
    
    if (len(dets) == 1):
        #get_landmarks()函数会将一个图像转化成np数组，并返回一个68 x2元素矩阵，输入图像的每个特征点对应每行的一个x，y坐标。
        landmarks = get_landmarks(img)
        left_mouth_location   =  np.ravel(landmarks[48,:])
        right_mouth_location =  np.ravel(landmarks[54,:])
        nose_location       =  np.ravel(landmarks[30,:])
        left_eye_location =   np.ravel(np.mean(landmarks[36:42], axis=0).astype(int))
        right_eye_location = np.ravel(np.mean(landmarks[42:48], axis=0).astype(int))
        # left_mouth
        left_mouth_length = landmarks[48,1] - min([landmarks[40,1],landmarks[41,1],landmarks[46,1],landmarks[47,1]])
        left_mouth_length =  left_mouth_length * 5 // 4
        #right_mouth
        right_mouth_length = landmarks[54,1] - min([landmarks[40,1],landmarks[41,1],landmarks[46,1],landmarks[47,1]])
        right_mouth_length = right_mouth_length * 5 // 4
        #nose
        nose_length = landmarks[30,1] - min([landmarks[19,1],landmarks[24,1]])
        nose_length = nose_length * 5 // 4
        #left eye
        left_eye_length  = landmarks[33,1] - left_eye_location[1]
        left_eye_length  = left_eye_length * 5 // 3
        #right eye
        right_eye_length  = landmarks[33,1] - right_eye_location[1]
        right_eye_length  = right_eye_length * 5 // 3
    else:
        left_mouth_location   = np.multiply(np.asarray([0.33800892, 0.76620404]), np.asarray([img.shape[1], img.shape[0]])).astype(int);
        right_mouth_location = np.multiply(np.asarray([0.67456876,  0.76865733]), np.asarray([img.shape[1], img.shape[0]])).astype(int);
        nose_location             = np.multiply(np.asarray([0.50269671,  0.59609821]), np.asarray([img.shape[1], img.shape[0]])).astype(int);
        left_eye_location        = np.multiply(np.asarray([0.31630742, 0.4066947]), np.asarray([img.shape[1], img.shape[0]])).astype(int);
        right_eye_location      = np.multiply(np.asarray([0.69326357,  0.40719753]), np.asarray([img.shape[1], img.shape[0]])).astype(int);
        
        left_mouth_length = int((0.21425544*img.shape[0] * img.shape[1]) ** 0.5) 
        right_mouth_length = int((0.21731148*img.shape[0] * img.shape[1]) ** 0.5)
        nose_length = int((0.16410947*img.shape[0] * img.shape[1]) ** 0.5)
        left_eye_length = int((0.21643651*img.shape[0] * img.shape[1]) ** 0.5)
        right_eye_length = int((0.21571577*img.shape[0] * img.shape[1]) ** 0.5)               
        
    img_lm =  img_seg(img,left_mouth_location,left_mouth_length)
    img_rm =  img_seg(img,right_mouth_location,right_mouth_length)
    img_nose = img_seg(img,nose_location,nose_length)
    img_le = img_seg(img,left_eye_location,left_eye_length)
    img_re = img_seg(img,right_eye_location,right_eye_length)
    return img_lm, img_rm, img_nose, img_le, img_re

def bundle(img1, img2):
    
    batch_1 = np.concatenate((img1, img2),  axis = 3)
    batch_2 = np.concatenate((img1, np.flip(img2, 2)), axis = 3)
    batch_3 = np.flip(batch_2, 2)
    batch_4 = np.flip(batch_1, 2)
                        
    batch_5 = np.concatenate((img2, img1), axis = 3)
    batch_6 = np.concatenate((img2, np.flip(img1, 2)), axis = 3)
    batch_7 = np.flip(batch_6, 2)
    batch_8 = np.flip(batch_5, 2)
    
    batch = np.stack((batch_1, batch_2, batch_3, batch_4, batch_5, batch_6, batch_7, batch_8), 0)
    return batch

def weight_variable(shape, name, stddev = 0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name = name)

def bias_variable(shape,name): 
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

def conv2d(x, W):  
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def CNN_Computaion(x_image, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1):#, keep_prob
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
    return h_fc1

def CNN_Computaion2(x_image, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1): 
    img1, img2, img3, img4, img5, img6, img7, img8 = tf.split(x_image, 8, 0)
    
    img1 = tf.squeeze(img1,0)
    img2 = tf.squeeze(img2,0)
    img3 = tf.squeeze(img3,0)
    img4 = tf.squeeze(img4,0)
    img5 = tf.squeeze(img5,0)
    img6 = tf.squeeze(img6,0)
    img7 = tf.squeeze(img7,0)
    img8 = tf.squeeze(img8,0)

    h_fc1_drop_1 =  CNN_Computaion(img1, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1)
    h_fc1_drop_2 =  CNN_Computaion(img2, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1)
    h_fc1_drop_3 =  CNN_Computaion(img3, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1)
    h_fc1_drop_4 =  CNN_Computaion(img4, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1)
    h_fc1_drop_5 =  CNN_Computaion(img5, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1)
    h_fc1_drop_6 =  CNN_Computaion(img6, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1)
    h_fc1_drop_7 =  CNN_Computaion(img7, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1)
    h_fc1_drop_8 =  CNN_Computaion(img8, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_fc1, b_fc1)
    
    h_fc1_drop = tf.concat([h_fc1_drop_1, h_fc1_drop_2, h_fc1_drop_3, h_fc1_drop_4, h_fc1_drop_5, h_fc1_drop_6, h_fc1_drop_7, h_fc1_drop_8],1)
    return h_fc1_drop

def FaceVerification(img_path1, img_path2):
    img_lm_1, img_rm_1, img_nose_1, img_le_1, img_re_1 = face_region(img_path1)
    img_lm_2, img_rm_2, img_nose_2, img_le_2, img_re_2 = face_region(img_path2)
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    test_right_mouth = bundle(img_rm_1, img_rm_2)
    test_left_mouth   = bundle(img_lm_1 , img_lm_2)
    test_right_eye      = bundle(img_re_1 , img_re_2)
    test_left_eye        = bundle(img_le_1  , img_le_2)
    test_nose             = bundle(img_nose_1, img_nose_2)
 
    x_image_nose  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')    
    x_image_left_mouth  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')
    x_image_right_mouth  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')
    x_image_left_eye  = tf.placeholder(tf.float32, [8, None, 31, 31, 6], name='x')
    x_image_right_eye= tf.placeholder(tf.float32, [8,None, 31, 31, 6], name='x')

    # drop out
    #keep_prob = tf.placeholder(tf.float32)

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
    
    h_fc1_drop_left_mouth = CNN_Computaion2(x_image_left_mouth, W_conv1_left_mouth, b_conv1_left_mouth, W_conv2_left_mouth, b_conv2_left_mouth, W_conv3_left_mouth, b_conv3_left_mouth, W_conv4_left_mouth,
                                             b_conv4_left_mouth,W_fc1_left_mouth, b_fc1_left_mouth)
    h_fc1_drop_right_mouth = CNN_Computaion2(x_image_right_mouth, W_conv1_right_mouth, b_conv1_right_mouth, W_conv2_right_mouth, b_conv2_right_mouth, W_conv3_right_mouth, b_conv3_right_mouth, W_conv4_right_mouth,
                                             b_conv4_right_mouth,W_fc1_right_mouth, b_fc1_right_mouth)
    h_fc1_drop_left_eye = CNN_Computaion2(x_image_left_eye, W_conv1_left_eye, b_conv1_left_eye, W_conv2_left_eye, b_conv2_left_eye, W_conv3_left_eye, b_conv3_left_eye, W_conv4_left_eye,
                                             b_conv4_left_eye,W_fc1_left_eye, b_fc1_left_eye)
    h_fc1_drop_right_eye = CNN_Computaion2(x_image_right_eye, W_conv1_right_eye, b_conv1_right_eye, W_conv2_right_eye, b_conv2_right_eye, W_conv3_right_eye, b_conv3_right_eye, W_conv4_right_eye,
                                             b_conv4_right_eye,W_fc1_right_eye, b_fc1_right_eye)
    h_fc1_drop_nose = CNN_Computaion2(x_image_nose, W_conv1_nose, b_conv1_nose, W_conv2_nose, b_conv2_nose, W_conv3_nose, b_conv3_nose, W_conv4_nose, b_conv4_nose,W_fc1_nose, b_fc1_nose)
    
    # RBM implentation
    h_fc1_drop = tf.concat([h_fc1_drop_left_mouth, h_fc1_drop_right_mouth, h_fc1_drop_left_eye, h_fc1_drop_right_eye, h_fc1_drop_nose], 1)
        
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
    prediction = tf.argmin(Probability_dis,1)
    # set check point
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    check_dir = './check_point'
    ckpt = tf.train.get_checkpoint_state(check_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        
    prediction_array = sess.run(prediction, feed_dict={x_image_nose: test_nose, x_image_left_eye: test_left_eye, x_image_right_eye: test_right_eye, x_image_left_mouth: test_left_mouth, x_image_right_mouth: test_right_mouth})
    sess.close()
    return prediction_array[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path1', type=str, default='E:/webface/0004960/003.jpg', help='Directory for storing face 1')
    parser.add_argument('--img_path2', type=str, default='E:/webface/0004960/006.jpg', help='Directory for storing face 2')
    FLAGS, unparsed = parser.parse_known_args()
    prediction_array = FaceVerification(FLAGS.img_path1, FLAGS.img_path2)
    print (prediction_array)
else:
    pass
