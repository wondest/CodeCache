import numpy as np
import tensorflow as tf

sess = tf.Session()

x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)

x_data = tf.placeholder(tf.float32, shape=x_shape)


## layer1
m_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
m_strides = [1, 2, 2, 1]
mov_avg_layer = tf.nn.conv2d(x_data, m_filter, m_strides, padding='SAME', name='Moving_Avg_Window')


## layer2
def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1, 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b)
    return(tf.sigmoid(temp))
    
with tf.name_scope('Custom_layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

print(sess.run(custom_layer1, feed_dict={x_data: x_val}))
    