# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:01:38 2018

@author: Tender

将多层神经网络应用到实际场景中，预测低出生体重数据集。

注意：数据文件无法下载，此处程序未经过实际检验

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

y_vals = np.array([x[10] for x in birth_data])

cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI', 'FTV']
x_vals = np.array([[x[ix] for ix,feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])

"""
3. Random seed
"""
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)

batch_size = 100

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
5. 初始化变量
"""
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(weight)
    
def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(bias)

"""
6. Tensorflow placeholder
"""
x_data = tf.placeholder(shape=[None, 8], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

"""
7. 多个 hidden layer,建立一个函数复用
"""
def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return(tf.nn.relu(layer))
    
"""
8. Create Model
"""

# Create first layer(25 hidden nodes)
weight_1 = init_weight(shape=[8, 25], std_dev=10.0)
bias_1 = init_bias(shape=[25], std_dev=10.0)
layer_1 = fully_connected(x_data, weight_1, bias_1)

# Create second layer(10 hidden nodes)
weight_2 = init_weight(shape=[25, 10], std_dev=10.0)
bias_2 = init_bias(shape=[10], std_dev=10.0)
layer_2 = fully_connected(layer_1, weight_2, bias_2)

# Create third layer(3 hidden nodes)
weight_3 = init_weight(shape=[10, 3], std_dev=10.0)
bias_3 = init_bias(shape=[3], std_dev=10.0)
layer_3 = fully_connected(layer_2, weight_3, bias_3)

# Create output layer(1 output value)
weight_4 = init_weight(shape=[3, 1], std_dev=10.0)
bias_4 = init_bias(shape=[1], std_dev=10.0)
final_output = fully_connected(layer_3, weight_4, bias_4)

"""
9. Loss and optimizer: L1
"""
loss = tf.reduce_mean(tf.abs(y_target - final_output))

my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(loss)

# init tensorflow variables
init = tf.global_variables_initializer()
sess.run(init)

"""
10. start train step
"""
loss_vec = []
test_loss = []

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
    test_temp_loss = sess.run(loss, feed_dict={x_data:x_vals_test, y_target:np.transpose([y_vals_test])})
    test_loss.append(test_temp_loss)
    
    if (i+1)%25==0:
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


"""
13. 预测体重:传入一个指示函数（判断是否大于2500克）
"""
actuals = np.array([x[1] for x in birth_data])
test_actuals = actuals[test_indices]
train_actuals = actuals[train_indices]

test_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data:x_vals_test})]
train_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data:x_vals_train})]

test_preds = np.array([1.0 if x<2500.0 else 0.0 for x in test_preds])
train_preds = np.array([1.0 if x<2500.0 else 0.0 for x in train_preds])

# Print out accuracies
test_acc = np.mean([x==y for x,y in zip(test_preds, test_actuals)])
train_acc = np.mean([x==y for x,y in zip(train_preds, train_actuals)])

print('On predicting the category of low birthweight from regression output (<2500g):')
print('Test Accuracy:{}' + format(test_acc))
print('Train Accuracy:{}' + format(train_acc))
