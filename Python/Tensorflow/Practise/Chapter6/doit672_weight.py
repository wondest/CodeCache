# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 17:04:09 2018

@author: Tender

662 需要拟合的参数远超线性模型

神经网络算法模型来优化低出生体重的逻辑模型

使用1个带有两个隐藏层的全联接层的神经网络，并采用sigmoid激励函数来拟合低出生体重的概率

"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import requests
import os

"""
1. Start tensorflow session
"""
sess = tf.Session()

"""
2. Load data,placeholder

"""
birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birthdata_filename = 'lowbwt.dat'

# Download data
if not os.path.exists(birthdata_filename):
    print('Not exists,download from remote')
    birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
    
    # Request data
    birthdata_file = requests.get(birthdata_url)

    # Write data to a file 
    with open(birthdata_filename, "wb") as data:
        data.write(birthdata_file.content)

# Load data from a file
birth_file = open(birthdata_filename, 'r')
birth_data = birth_file.read().split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split(' ') if len(x)>=1]
birth_data = [[float(x) for x in y.split(' ') if len(x)>1] for y in birth_data[1:] if len(y)>=1]
birth_file.close()

y_vals = np.array([x[1] for x in birth_data])
x_vals = np.array([x[2:9] for x in birth_data])

"""
3. Random seed
"""
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)

batch_size = 90

"""
4. 分割测试数据和训练数据，使用min-max进行归一化特征数据
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
6. Tensorflow placeholder
"""
x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

def init_variable(shape):
    return(tf.Variable(tf.random_normal(shape=shape)))
    
def logistic(input_layer, multiplication_weight, bias_weight, activation = True):
    linear_layer = tf.add(tf.matmul(input_layer, multiplication_weight), bias_weight)
    
    if activation:
        return(tf.nn.sigmoid(linear_layer))
    else:
        return(linear_layer)

"""
7. Model
"""
# First logistic layer(7 inputs to 14 hidden nodes)
A1 = init_variable(shape=[7,14])
b1 = init_variable(shape=[14])
logistic_layer1 = logistic(x_data, A1, b1)

# Second logistic layer(14 inputs to 5 hidden nodes)
A2 = init_variable(shape=[14,5])
b2 = init_variable(shape=[5])
logistic_layer2 = logistic(logistic_layer1, A2, b2)

# Final output layer(5 inputs to 1hidden nodes)
A3 = init_variable(shape=[5,1])
b3 = init_variable(shape=[5])
final_output = logistic(logistic_layer2, A3, b3)


"""
8. Loss and optimizer: sigmoid 交叉熵，度量概率之间的距离
"""
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(final_output, y_target))

my_opt = tf.train.AdamOptimizer(learning_rate = 0.002)
train_step = my_opt.minimize(loss)

# init tensorflow variables
init = tf.global_variables_initializer()
sess.run(init)


"""
9. 评估预测准确率
"""
prediction = tf.round(tf.nn.sigmoid(final_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)


"""
10. start train step
"""
loss_vec = []
test_acc = []
train_acc = []

for i in range(200):
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
    test_acc_train = sess.run(accuracy, feed_dict={x_data:x_vals_test, y_target:np.transpose([y_vals_test])})
    test_acc.append(test_acc_train)
    train_acc_train = sess.run(accuracy, feed_dict={x_data:x_vals_train, y_target:np.transpose([y_vals_train])})
    train_acc.append(train_acc_train)
    
    if (i+1)%25==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))
        
"""
11.Paint
"""
plt.plot(loss_vec, 'k-', label='Train Loss')

plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()