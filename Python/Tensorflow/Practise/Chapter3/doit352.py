# -*- coding: utf-8 -*-

"""
Created on Tue Sep 18 23:55:27 2018

@author: tender
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()
iris = datasets.load_iris()


### Import data
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

###
learning_rate = 0.3
batch_size = 25
iterations = 100

x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

###
model_matmul=tf.matmul(x_data, A)
model_output = tf.add(model_matmul, b)

###
loss_l2 = tf.reduce_mean(tf.square(y_target - model_output))
loss_l1 = tf.reduce_mean(tf.abs(y_target - model_output))

###
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step_l2 = my_opt.minimize(loss_l2)
train_step_l1 = my_opt.minimize(loss_l1)

###
init = tf.global_variables_initializer()

###
sess.run(init)
loss_vec_l2 = []

for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step_l2, feed_dict={x_data:rand_x, y_target:rand_y})
    
    temp_loss = sess.run(loss_l2, feed_dict={x_data:rand_x, y_target:rand_y})
    loss_vec_l2.append(temp_loss)
    
    if (i+1)%25==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))
        
###
sess.run(init)

###
sess.run(init)
loss_vec_l1 = []

for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step_l1, feed_dict={x_data:rand_x, y_target:rand_y})
    
    temp_loss = sess.run(loss_l1, feed_dict={x_data:rand_x, y_target:rand_y})
    loss_vec_l1.append(temp_loss)
    
    if (i+1)%5==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

      

plt.plot(loss_vec_l2, 'k-', label='L2 Loss')
plt.plot(loss_vec_l1, 'r--', label='L1 Loss')

plt.title('L2 and L1 Loss per Generation')

plt.xlabel('Generation')

plt.ylabel('L1 Loss')

plt.legend(loc='upper right')

plt.show()