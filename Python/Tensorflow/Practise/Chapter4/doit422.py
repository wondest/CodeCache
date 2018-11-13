#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 21:14:25 2018

@author: tender

SVM 特殊损失函数
"""

import matplotlib.pyplot as plt
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

"""
Choose test data and train data
"""
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]

"""
A,b
"""
x_data = tf.placeholder(shape=[None,2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None,1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[2,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

"""
model
"""
model_output = tf.subtract(tf.matmul(x_data, A), b)

"""
loss function
"""
l2_norm = tf.reduce_mean(tf.square(A))

alpha = tf.constant([0.1])

classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))

loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

"""
accuracy function
"""
prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))


"""
optimize function
"""
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

"""
init tensorflow
"""
init = tf.global_variables_initializer()
sess.run(init)

"""
start train,test
"""

loss_vec = []
train_accuracy = []
test_accuracy = []

batch_size = 100
train_num = 500

for i in range(train_num):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    
    sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})
    
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target:rand_y})
    loss_vec.append(temp_loss)
    
    train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target:np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)
    
    test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target:np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)
    
    if (i+1)%100==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))
    
"""
paint: a1x1+a2x2-b=0, x1=(-a2/a1)x2+(b/a1)
"""

# Line
[[a1],[a2]] = sess.run(A)
[[b]] = sess.run(b)

slope = -a2/a1
y_intercept = b/a1
 
x2_vals = [d[1] for d in x_vals]

best_fit = []
for i in x2_vals:
    best_fit.append(slope*i+y_intercept)

# set points
setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==1]
setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==1]

# not_set points
not_setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==-1]
not_setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==-1]

# Paint line and points
plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x2_vals, best_fit, 'r-', label='Liner Separator', linewidth=3)

plt.ylim([0,10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

# Paint accuracy
plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend('lower right')
plt.show()

# Paint loss
#plt.plot(loss_vec, 'k-')
#plt.title('Loss per Generation')
#plt.xlabel('Generation')
#plt.ylabel('Loss')
#plt.plot