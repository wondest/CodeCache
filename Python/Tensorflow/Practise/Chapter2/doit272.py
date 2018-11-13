#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:29:07 2018

@author: tender
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()

## batch size
batch_size = 20

## data placeholder variable
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))

## operation
my_output = tf.matmul(x_data, A)

## loss
loss = tf.reduce_mean(tf.square(my_output - y_target))

## opt
my_opt = tf.train.GradientDescentOptimizer(0.22)
train_step = my_opt.minimize(loss)

## train action

##
init = tf.global_variables_initializer()
sess.run(init)

loss_batch = []
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data:rand_x, y_target: rand_y})
    if (i+1)%5 == 0:
        print('Step #' + str(i+1) + ' A= ' + str(sess.run(A)))
        temp_loss=sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)

loss_stochastic = []
for i in range(100):
    rand_index = np.random.choice(100, size=1)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data:rand_x, y_target: rand_y})
    if (i+1)%5 == 0:
        print('Step #' + str(i+1) + ' A= ' + str(sess.run(A)))
        temp_loss=sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_stochastic.append(temp_loss)
        
        
plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss size=20')
plt.legend(loc='upper right', prop={'size':11})
plt.show()