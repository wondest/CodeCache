#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Nov 04 21:06:17 2018
@Author : tender
@Description : SVM核函数
"""
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

"""
tensorflow session
"""
sess = tf.Session()

"""
Import data
(1) If iris,then 1,else -1
"""
iris = datasets.load_iris()
x_vals = np.array([[x[0],x[3]] for x in iris.data])
y_vals = np.array([1 if y==0 else -1 for y in iris.target])

#同心圆1数据集
class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==1]
class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==1]

#同心圆2数据集
class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1]
class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1]

"""

"""
batch_size = 100
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

prediction_grid = tf.placeholder(shape=[None,2], dtype=tf.float32)

b = tf.Variable(tf.random_normal(shape=[1,batch_size]))

"""
高斯核
"""
gamma = tf.constant(-10.0)


#矩阵方式求欧式距离 A_sq + B_sq - 2AB_transpose
# A 一行代表一个向量
# B 一行代表一个向量

#矩阵内的元素平方
#reduce_sum(*,1) 按列方向,行内累加  0的话就是按行方向,列内累加
# -1：自动调整  1：1列,调整成为[*,1]的shape
A_sq = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1,1])

# 此处利用了矩阵广播实现了:A_sq的扩展
# A_sq + B_sq - 2AB  由于此处 A=transpose(B)  A_sq=transopose(B)
eu_dists = tf.add(tf.subtract(A_sq, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(A_sq))

#高斯核函数
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(eu_dists)))

"""
损失函数
"""
#
first_term = tf.reduce_sum(b)

#
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = tf.matmul(y_target, tf.transpose(y_target))

#此处是矩阵元素相乘
second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
loss = tf.negative(tf.subtract(first_term, second_term))

"""
预测函数和准确率函数
"""
## 与上面的高斯核类似
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1,1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))

## row(rA) * row(rB)
## row(rA) 训练时候的样本向量 row(rA)*dim
## row(rB) 需要预测的样本向量 row(rB)*dim
## A*B_transpose = row(rA)*row(rB)
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

## 1*row(rB)
prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)

##
prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

"""
优化函数
"""
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

"""
init tensorflow
"""
init = tf.global_variables_initializer()
sess.run(init)

"""
start train
"""
loss_vec = []
batch_accuracy = []
train_num = 2000

for i in range(train_num):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})


    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target:rand_y})

    loss_vec.append(temp_loss)
    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target:rand_y, prediction_grid:rand_x})

    batch_accuracy.append(acc_temp)

    if (i + 1) % 50 == 0:
        print('-----------------')
        print('Generation: ' + str(i))
        print('Step #' + str(i + 1) + ' Loss = ' + str(temp_loss))

"""
画图
"""
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]

rand_index = np.random.choice(len(x_vals), size=batch_size)
rand_x = x_vals[rand_index]
rand_y = np.transpose([y_vals[rand_index]])

[grid_predictions] = sess.run(prediction, feed_dict={x_data: rand_x, y_target:rand_y, prediction_grid:grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)
plt.contourf(xx, yy, grid_predictions, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class 2')
plt.legend(loc='lower right')
plt.ylim([-0.5,3.0])
plt.xlim([3.5,8.5])
plt.show()