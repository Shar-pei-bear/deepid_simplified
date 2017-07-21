import tensorflow as tf

def weight_variable(shape, name, stddev = 0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name = name)

def bias_variable(shape,name): 
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

# convolution and pooling
def conv2d(x, W):  
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

