# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 23:34:08 2018

@author: Tender


神经网络算法的层 能以任意形式组合

最常用的方法是使用卷积层和全联接层来创建特征

如果我们有许多特征，常用的处理方法是采用池化层。

最后在这些层后，引入激励函数

此处我们以一维数据进行演示

"""

import numpy as np
import tensorflow as tf

"""
1. Start tensorflow session
"""
sess = tf.Session()

"""
2. Load data,placeholder
"""
data_size = 25
data_1d = np.random.normal(size=data_size)
x_input_1d = tf.placeholder(shape=[data_size], dtype=tf.float32)

"""
3. 卷积层
"""

# Tensorflow的层函数是为四维数据设计（batch_size,width,height,channels) 我们需要扩展维度 降低维度
# 
# 此处1维
def conv_layer_1d(input_1d, my_filter):
    # Make 1d input 4d
    input_2d = tf.expand_dims(input_1d, 0) # width
    input_3d = tf.expand_dims(input_2d, 0) # batch_size
    input_4d = tf.expand_dims(input_3d, 3) # channels
    
    # Perform convolution
    # 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
    # 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,1,1,1], padding="VALID")
    
    # Now drop extra dimensions
    conv_output_1d = tf.squeeze(convolution_output)
    return (conv_output_1d)
    
my_filter = tf.Variable(tf.random_normal(shape=[1,5,1,1]))
my_convolution_output = conv_layer_1d(x_input_1d, my_filter)

"""
激励函数
"""
def activation(input_1d):
    return (tf.nn.relu(input_1d))

my_activation_output = activation(my_convolution_output)


"""
池化层
"""
def max_pool(input_1d, width):
    # Make 1d input 4d
    input_2d = tf.expand_dims(input_1d, 0) # width
    input_3d = tf.expand_dims(input_2d, 0) # batch_size
    input_4d = tf.expand_dims(input_3d, 3) # channels
    
    pool_output = tf.nn.max_pool(input_4d, ksize=[1,1,width,1], strides=[1,1,1,1], padding="VALID")
    
    # Now drop extra dimensions
    pool_output_1d = tf.squeeze(pool_output)
    return (pool_output_1d)
    
my_maxpool_output = max_pool(my_activation_output, width=5)

"""
全联接层
"""
def fully_connected(input_layer, num_outputs):
    # Create weights
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))
    
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])
    
    # Make input into 2d
    input_layer_2d = tf.expand_dims(input_layer, 0)
    
    # Perform fully connected operations
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)
    
    # Drop extra dimensions
    full_output_1d = tf.squeeze(full_output)
    
    return (full_output_1d)

my_full_output = fully_connected(my_maxpool_output, 5)


"""

"""
# init tensorflow variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_1d: data_1d}

# Convolution Output
print('Input = array of length 25')
print('Convolution w/filter, length = 5, strider size=1, results in an array of length 21:')
print(sess.run(my_convolution_output, feed_dict=feed_dict))

# Activation Output
print('\nInput = the above array of length 21')
print('ReLU element wise returns the array of length 21')
print(sess.run(my_activation_output, feed_dict=feed_dict))
    
# Maxpool Output
print('\nInput = the above array of length 21')
print('Maxpool, window length=5, stride size=1, results in the array of length (21-5+1 = 17)')
print(sess.run(my_maxpool_output, feed_dict=feed_dict))

# Fully Connected Output
print('\nInput = the above array of length 17')
print('Full connected layer on all four rows with five outputs:')
print(sess.run(my_full_output, feed_dict=feed_dict))