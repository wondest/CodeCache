# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 23:52:47 2018

@author: Tender


扩展最近邻域法进行多维度缩放（不同的特征值，应该用不同的的归一化方式进行缩放，用此思路来优化房价预测）

混合距离的计算：

加权距离函数的关键是使用加权权重矩阵

"""

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

import requests
import os

"""
Tensorflow session
"""
sess = tf.Session()

housing_filename='housing.dat'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Download data
if not os.path.exists(housing_filename):
    print('Not exists,download from remote')
    housing_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

    # Request data
    housing_file = requests.get(housing_url)

    # Write data to a file 
    with open(housing_filename, "wb") as data:
        data.write(housing_file.content)

# Load data from a file
file = open(housing_filename, 'r')

# Parse data
housing_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in file.read().split('\n') if len(y) >=1 ]
y_vals = np.transpose([np.array([y[13] for y in housing_data])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

# 缩放x_vals的值到 0-1之间
x_vals = (x_vals - x_vals.min(0))/x_vals.ptp(0)

# 创建对角权重矩阵，该矩阵提供归一化的距离度量，其值为特征的标准差（可以使用其他权值）
# tf.diag 生成对角矩阵，tf.cast 转化为float32类型
weight_diagonal = x_vals.std(0)
weight_matrix = tf.cast(tf.diag(weight_diagonal), dtype=tf.float32)

# Train data and Test data
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]

# Some constants
batch_size = len(x_vals_test)
num_features = len(cols_used)
k = 4

# Tensorflor placeholder
x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)

y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32) 
y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 声明距离函数  sqrt((x-y).T A (x-y))
# (x-y) 
# A 是对角权重矩阵


# 101 size of test
# 405 size of train
# 10  number of features

# 减法传播 (x-y) 101*405*10
substraction_term = tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))

#tensorflow中tile是用来复制tensor的指定维度
#a1 = tf.tile(a, [2, 2]) 表示把a的第一个维度复制两次，第二个维度复制2次

# 101*405*10 X 101*10*10 => 101*405*10
first_product = tf.matmul(substraction_term, tf.tile(tf.expand_dims(weight_matrix, 0), [batch_size,1,1]))

# 101*405*10 X 101*10*405 = 101*405*405
second_product = tf.matmul(first_product, tf.transpose(substraction_term, perm=[0,2,1]))

#matrix_diag_part 功能：返回批对角阵的对角元素
# 101*405*405 => 101*405
# 计算得到所有测试点到 训练集的距离
distance = tf.sqrt(tf.matrix_diag_part(second_product))


"""

"""
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1), 1)
x_sums_repeated = tf.matmul(x_sums, tf.ones([1, k], tf.float32))

# 距离越近，权重越小
x_val_weights = tf.expand_dims(tf.div(top_k_xvals, x_sums_repeated), 1)

top_k_yvals = tf.gather(y_target_train, top_k_indices)

prediction = tf.squeeze(tf.matmul(x_val_weights, top_k_yvals), axis=[1])

mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)    

"""
init tensorflow
"""
init = tf.global_variables_initializer()
sess.run(init)


"""
开始测试
"""
num_loops = int(np.ceil(len(x_vals_test)/batch_size))

for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size, len(x_vals_train))
    
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    
    predictions = sess.run(prediction, feed_dict={x_data_train:x_vals_train, x_data_test:x_batch, y_target_train:y_vals_train, y_target_test:y_batch})
    batch_mse = sess.run(mse, feed_dict={x_data_train:x_vals_train, x_data_test:x_batch, y_target_train:y_vals_train, y_target_test:y_batch})
    
    print('Batch #' + str(i+1) + 'MSE:' + str(np.round(batch_mse,3)))
          

"""
画图
"""
bins = np.linspace(5, 50, 45)

plt.hist(predictions, bins, alpha=0.5, label='Prediction')
plt.hist(y_batch, bins, alpha=0.5, label='Actual')
plt.title('Histogram of Predicted and Actual Values')
plt.xlabel('Med Home Value in $1,000s')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()          
    
"""
"""
file.close()










