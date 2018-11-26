# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 00:34:31 2018

@author: Tender

二维数据

多层神经网络

"""

import numpy as np
import tensorflow as tf

"""
1. Start tensorflow session
"""
sess = tf.Session()

"""
2. Load data,placeholder

  输入数组为10 * 10矩阵
"""
data_size = [10,10]
data_2d = np.random.normal(size=data_size)
x_input_2d = tf.placeholder(shape=data_size, dtype=tf.float32)

"""
3. 卷积层
"""

# 数据集使用宽度和高度，这里扩展 批量=1 颜色通道=1
# 本例子使用一个随机的 2*2 的过滤层
# 10*10 步长为 2*2 padding VALID => 卷积输出是  (10 - 2)/2 + 1 = 5
def conv_layer_1d(input_2d, my_filter):
    # Make 1d input 4d
    input_3d = tf.expand_dims(input_2d, 0) # width
    input_4d = tf.expand_dims(input_3d, 3) # channels
    
    # Perform convolution
    # 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
    # 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4 [batch_size, width, height, channel]
    # 第四个参数padding VALID是采用丢弃的方式; SAME的方式,采用的是补全的方式
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,2,2,1], padding="VALID")
    
    # Now drop extra dimensions
    conv_output_2d = tf.squeeze(convolution_output)
    return (conv_output_2d)
    
my_filter = tf.Variable(tf.random_normal(shape=[2,2,1,1]))
my_convolution_output = conv_layer_1d(x_input_2d, my_filter)


"""
4. 激励函数
"""
def activation(input_2d):
    return (tf.nn.relu(input_2d))

my_activation_output = activation(my_convolution_output)


"""
5. 池化层

扩展为2d,移动的宽度和高度 

"""
def max_pool(input_2d, width, height):
    # Make 1d input 4d
    input_3d = tf.expand_dims(input_2d, 0) # batch_size
    input_4d = tf.expand_dims(input_3d, 3) # channels
    
    pool_output = tf.nn.max_pool(input_4d, ksize=[1,height,width,1], strides=[1,1,1,1], padding="VALID")
    
    # Now drop extra dimensions
    pool_output_2d = tf.squeeze(pool_output)
    return (pool_output_2d)
    
my_maxpool_output = max_pool(my_activation_output, width=2, height=2)

"""
6. 全联接层
"""
def fully_connected(input_layer, num_outputs):
    # Flatten into 1d
    flat_input = tf.reshape(input_layer, [-1])
    
    # Create weights
    weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])
    
    # Make input into 2d
    input_2d = tf.expand_dims(flat_input, 0)
    
    # Perform fully connected operations
    full_output = tf.add(tf.matmul(input_2d, weight), bias)

    # Drop extra dimensions
    full_output_2d = tf.squeeze(full_output)
    
    return (full_output_2d)

my_full_output = fully_connected(my_maxpool_output, 5)


"""
7. Init tensorflow variable and feed_dict
"""
# init tensorflow variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_2d: data_2d}

"""
8. Print result
"""
print('Input = [10 * 10] array')
print('2*2 Convolution, stride size = [2*2], results is the [5*5] array:')
print(sess.run(my_convolution_output, feed_dict=feed_dict))

# Activation Output
print('\nInput = the above [5*5] array')
print('ReLU element wise returns the [5*5] array')
print(sess.run(my_activation_output, feed_dict=feed_dict))
    
# Maxpool Output
print('\nInput = the above [5*5] array')
print('Maxpool,window size = [2*2], stride size = [1*1], results is the [4*4] array:')
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

# Fully Connected Output
print('\nInput = the above [4*4] array')
print('Full connected layer on all four rows with five outputs:')
print(sess.run(my_full_output, feed_dict=feed_dict))
