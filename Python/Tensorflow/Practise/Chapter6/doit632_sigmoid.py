# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 23:18:38 2018

@author: Tender

激励函数
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

"""
Tensorflow session
"""
sess = tf.Session()


"""
随机数
"""
tf.set_random_seed(5)
np.random.seed(42)

"""
Tensorflow variable and placeholder
"""
batch_size = 50

a1 = tf.Variable(tf.random_normal(shape=[1,1]))
b1 = tf.Variable(tf.random_normal(shape=[1,1]))
a2 = tf.Variable(tf.random_normal(shape=[1,1]))
b2 = tf.Variable(tf.random_normal(shape=[1,1]))

x = np.random.normal(2, 0.1, 500)
x_data = tf.placeholder(shape=[None,1], dtype=tf.float32)

"""
Train model
"""
# sigmoid(x * a1 + b1)
sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))

# relu(x * a2 + b2)
relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

"""
损失函数
"""
loss1 = tf.reduce_sum(tf.square(tf.subtract(sigmoid_activation, 0.75)))
loss2 = tf.reduce_sum(tf.square(tf.subtract(relu_activation, 0.75)))

"""
optmimizing
"""
my_opt = tf.train.GradientDescentOptimizer(0.1)
train_step_relu = my_opt.minimize(loss2)
train_step_sigmonid = my_opt.minimize(loss1)

"""
init tensorflow
"""
init = tf.global_variables_initializer()
sess.run(init)

"""
Train step
"""
loss_vec_sigmoid = []
loss_vec_relu = []
activation_sigmoid = []
activation_relu = []

for i in range(1000):
    
    rand_indices = np.random.choice(len(x), size=batch_size)
    x_vals = np.transpose([x[rand_indices]])
    
    sess.run(train_step_sigmonid, feed_dict={x_data:x_vals})
    sess.run(train_step_relu, feed_dict={x_data:x_vals})
    
    loss_vec_sigmoid.append(sess.run(loss1, feed_dict={x_data:x_vals}))
    loss_vec_relu.append(sess.run(loss2, feed_dict={x_data:x_vals}))
    
    activation_sigmoid.append(np.mean(sess.run(sigmoid_activation, feed_dict={x_data:x_vals})))
    activation_relu.append(np.mean(sess.run(relu_activation, feed_dict={x_data:x_vals})))

    if (i+1)%100 == 0:
        print('The genetion: ' + str(i) + ' a2 = ' + str(sess.run(a2)) + ' b2 = ' + str(sess.run(b2)) + ' activation = ' + str(np.mean(sess.run(relu_activation, feed_dict={x_data:x_vals}))) + ' loss2= ' + str(sess.run(loss1, feed_dict={x_data:x_vals})))
        #print('The genetion: ' + str(i) + ' a1 = ' + str(sess.run(a1)) + ' b1 = ' + str(sess.run(b1)) + ' activation = ' + str(np.mean(sess.run(sigmoid_activation, feed_dict={x_data:x_vals}))) + ' loss2= ' + str(sess.run(loss1, feed_dict={x_data:x_vals})))
"""
Paint
"""
plt.plot(activation_sigmoid, 'k-', label='Sigmonid Activation')
plt.plot(activation_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Activation Outputs')
plt.xlabel('Generation')
plt.ylabel('Outputs')
plt.legend(loc='upper right')
plt.show

"""
plt.plot(loss_vec_sigmoid, 'k-', label='Sigmonid Loss')
plt.plot(loss_vec_relu, 'r--', label='Relu Loss')
plt.ylim([0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show
"""