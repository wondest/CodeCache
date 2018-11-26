# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 17:32:53 2018

@author: Tender

简单的单层神经网络算法

Iris数据集上进行训练

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

"""
1. Start tensorflow session
"""
sess = tf.Session()


"""
2. Load data
"""
iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([y[3] for y in iris.data])

"""
3. 由于数据集较小，设置一个随机种子
"""
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)


"""
4. 准备数据集和 80:20 训练集和测试集，通过min-max缩放正则化x的特征值到0-1之间
"""
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

def normalize_cols(m):
    return(m-m.min(axis=0))/m.ptp(axis=0)
    
# turn 0/0 (nan) into
x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

"""
5. Tensorflow placeholder
"""
batch_size = 50
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

"""
6. model definition
"""
hidden_layer_nodes = 5

# input 3 -> hidden 5
A1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))

# hidden 5 -> output 1
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))

"""
7. output
"""
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

"""
8. loss
"""
loss = tf.reduce_mean(tf.square(y_target - final_output))

"""
9. optimizer
"""
learning_rate=0.005
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

# init tensorflow variables
init = tf.global_variables_initializer()
sess.run(init)


"""
10. start train step
"""
loss_vec = []
test_loss = []

for i in range(500):
    # First we select a random set of indices for the batch.
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    
    # select the training data
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    
    # Now we run the training step
    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    
    # We save the training loss
    temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})
    loss_vec.append(temp_loss)
    
    # Finally, we run the test-set loss and save it.
    test_temp_loss = sess.run(loss, feed_dict={x_data:x_vals_test, y_target:np.transpose([y_vals_test])})
    test_loss.append(test_temp_loss)
    
    if (i+1)%50==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))
        
        
"""
11. paint
"""
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')

plt.title('Loss(MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
    
    
    
    
    
    