# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:10:55 2018

@author: Tender

最近领域算法常用于图像识别

MNIST手写数字样本数据集

"""

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

"""
Tensorflow session
"""
sess = tf.Session()

"""
读取数据集： one_hot编码是分类类别的数值化,样本包含十个类别（0到9）
类别0：1，0，0，0，0，0，0，0，0
类别1：0，1，0，0，0，0，0，0，0

直接下载数据集，下载地址：http://yann.lecun.com/exdb/mnist/
"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""

手写数字集较大，且直接计算784个特征值之间的距离是比较困难的，所以本例会抽样成小数据集进行训练
对测试集也进行抽样处理
"""
train_size = 1000
test_size = 102

rand_train_indices = np.random.choice(len(mnist.train.images), train_size, replace=False)
rand_test_indices = np.random.choice(len(mnist.test.images), test_size, replace=False)

x_vals_train = mnist.train.images[rand_train_indices]
x_vals_test = mnist.test.images[rand_test_indices]

y_vals_train = mnist.train.labels[rand_train_indices]
y_vals_test = mnist.test.labels[rand_test_indices]


k=4
batch_size=6

"""
Tensorflow placeholder
"""
# Tensorflor placeholder
x_data_train = tf.placeholder(shape=[None, 784], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, 784], dtype=tf.float32)

y_target_train = tf.placeholder(shape=[None, 10], dtype=tf.float32) 
y_target_test = tf.placeholder(shape=[None, 10], dtype=tf.float32)

"""
距离函数
"""

# 1000 * 784 - 6 * 1 * 784 = 6 * 1000 * 784
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=2)


"""

"""
# 6 * 4
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)

#按照指定的下标集合从axis=0中抽取子集
prediction_indices = tf.gather(y_target_train, top_k_indices)
count_of_predictions = tf.reduce_sum(prediction_indices, reduction_indices=1)

prediction = tf.argmax(count_of_predictions, dimension=1)

"""
"""
num_loops = int(np.ceil(len(x_vals_test)/batch_size))
test_output = []
actual_vals = []

for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size, len(x_vals_train))
    
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    
    feed_dict={x_data_train:x_vals_train, x_data_test:x_batch, y_target_train:y_vals_train, y_target_test:y_batch}
    
    predictions = sess.run(prediction, feed_dict=feed_dict)
    
    test_output.extend(predictions)
    
    actual_vals.extend(np.argmax(y_batch, axis=1))
    
    
"""
"""
accuracy = sum(1./test_size for i in range(test_size) if test_output[i]==actual_vals[i])

print('Accuracy on test set: ' + str(accuracy))

"""
"""
actuals = np.argmax(y_batch, axis=1)
Nrows = 2
Ncols = 3

for i in range(len(actuals)):
    plt.subplot(Nrows, Ncols, i+1)
    plt.imshow(np.reshape(x_batch[i], [28,28]), cmap='Greys_r')
    plt.title('Actual:' + str(actuals[i]) + ' Pred: ' + str(predictions[i]), fontsize=10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)