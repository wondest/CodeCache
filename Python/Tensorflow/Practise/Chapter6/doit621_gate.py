# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:19:04 2018

@author: Tender

"""

import tensorflow as tf

"""
Tensorflow session
"""
sess = tf.Session()


"""
values
"""
x_val = 5.
y_expect = 50.

"""
variable and placeholder
"""
a = tf.Variable(tf.constant(4.))
x_data = tf.placeholder(dtype = tf.float32)

"""
model f(x) = a * x
"""
multiplication = tf.multiply(a, x_data)

"""
loss function
"""
loss = tf.square(tf.subtract(multiplication,y_expect))

"""
init tensorflow
"""
init = tf.global_variables_initializer()
sess.run(init)

"""
optmimizing
"""
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

"""
train
"""
print('Optimizing a Multiplication Gate Output to 50.')

for i in range(10):
    sess.run(train_step, feed_dict={x_data:x_val})
    a_val = sess.run(a)
    mul_output = sess.run(multiplication, feed_dict={x_data:x_val})
    
    print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mul_output))