# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 22:40:12 2018

@author: Tender

SVM 多类支持向量机

两种策略： 一对多法 one-versus-rest  或者  一对一法 one-versus-one
"""

"""
一对多法 one-versus-rest

 训练时依次把某个类别的样本归为一类,其他剩余的样本归为另一类，这样k个类别的样本就构造出了k个SVM。
 分类时将未知样本分类为具有最大分类函数值的那类。
　　假如我有四类要划分（也就是4个Label），他们是A、B、C、D。
　　于是我在抽取训练集的时候，分别抽取
　　（1）A所对应的向量作为正集，B，C，D所对应的向量作为负集；
　　（2）B所对应的向量作为正集，A，C，D所对应的向量作为负集；
　　（3）C所对应的向量作为正集，A，B，D所对应的向量作为负集；
　　（4）D所对应的向量作为正集，A，B，C所对应的向量作为负集；
　　使用这四个训练集分别进行训练，然后的得到四个训练结果文件。
　　在测试的时候，把对应的测试向量分别利用这四个训练结果文件进行测试。
　　最后每个测试都有一个结果f1(x),f2(x),f3(x),f4(x),于是最终的结果便是这四个值中最大的一个作为分类结果。
评价
优点：训练k个分类器，个数较少，其分类速度相对较快。
 缺点：
  ①每个分类器的训练都是将全部的样本作为训练样本，这样在求解二次规划问题时，训练速度会随着训练样本的数量的增加而急剧减慢；
  ②同时由于负类样本的数据要远远大于正类样本的数据，从而出现了样本不对称的情况，且这种情况随着训练数据的增加而趋向严重。解决不对称的问题可以引入不同的惩罚因子，对样本点来说较少的正类采用较大的惩罚因子C；
  ③还有就是当有新的类别加进来时，需要对所有的模型进行重新训练。
"""

"""
一对一法 one-versus-one

　　假设有四类A,B,C,D四类。在训练的时候我选择A,B; A,C; A,D; B,C; B,D;C,D所对应的向量作为训练集，然后得到六个训练结果，在测试的时候，把对应的向量分别对六个结果进行测试，然后采取投票形式，最后得到一组结果。

　　投票是这样的：
　　A=B=C=D=0;
　　(A,B)-classifier 如果是A win,则A=A+1;otherwise,B=B+1;
　　(A,C)-classifier 如果是A win,则A=A+1;otherwise, C=C+1;
　　...
　　(C,D)-classifier 如果是A win,则C=C+1;otherwise,D=D+1;
　　The decision is the Max(A,B,C,D)

评价：这种方法虽然好,但是当类别很多的时候,model的个数是n*(n-1)/2,代价还是相当大的。
"""

"""
作者：xfChen2 
来源：CSDN 
原文：https://blog.csdn.net/xfchen2/article/details/79621396 
版权声明：本文为博主原创文章，转载请附上博文链接！
"""

"""
一对多法 one-versus-rest

分成3类，那么需要构造3个SVM
"""
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

"""
Tensorflow session
"""
sess = tf.Session()

"""
Import data  150*n
"""
iris = datasets.load_iris()
x_vals = np.array([[x[0],x[3]] for x in iris.data])

y_vals1 = np.array([1 if y==0 else -1 for y in iris.target])
y_vals2 = np.array([1 if y==1 else -1 for y in iris.target])
y_vals3 = np.array([1 if y==2 else -1 for y in iris.target])

y_vals=np.array([y_vals1, y_vals2, y_vals3])

class1_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==0]
class1_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==0]

class2_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==1]
class2_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==1]

class3_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==2]
class3_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==2]

"""
Tensorflow placeholder
"""
batch_size = 5

x_data = tf.placeholder(shape=[None,2], dtype=tf.float32)
y_target = tf.placeholder(shape=[3,None], dtype=tf.float32)

prediction_grid = tf.placeholder(shape=[None,2], dtype=tf.float32)

b = tf.Variable(tf.random_normal(shape=[3,batch_size]))


"""
高斯核函数
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
损失函数 : let batch_size = 5

要进行三个SVM分类，用的是一个训练集

一个svm对应一个 拉格朗日对偶优化函数，三个对应三个，
那么  loss = 三个拉格朗日对偶优化函数之和 
"""

model_output = tf.matmul(b, my_kernel)

# 3*5
first_term = tf.reduce_sum(b, 1)

# 5*3 X 3*5 = 5*5 
b_vec_cross = tf.matmul(tf.transpose(b), b)


# 三维张量
# a = [100, 3, 4]
# b = [100, 4, 5]
# matmul(a,b) => [100, 3, 5]  做100次 3*4 X 4*5
def reshape_matmul(mat):
    # 3*1*5
    v1 = tf.expand_dims(mat, 1)
    # 3*5*1
    v2 = tf.reshape(v1, [3, batch_size, 1])
    # 3*5*5
    return(tf.matmul(v2, v1))

#
y_target_cross = reshape_matmul(y_target)

# 利用扩散机制，将优化参数 扩散到 3维
# y_target_cross 3 5*5   b_vec_cross 5*5
# 此处 reduce_sum 必须指定维度，只对 5*5 进行 reduce_sum
# 得到 3 * 5
second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)), [1,2])

#
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

"""
预测函数
"""
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1,1])

pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

prediction_output = tf.matmul(tf.multiply(y_target, b), pred_kernel)

## 多类别输出，预测值是分类器有最大返回值的类别
prediction = tf.argmax(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1),1),0)

accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 0)), tf.float32))

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
开始训练
"""

"""
start train
"""
loss_vec = []
batch_accuracy = []
train_num = 2000

for i in range(train_num):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[:,rand_index]

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

#rand_index = np.random.choice(len(x_vals), size=batch_size)
#rand_x = x_vals[rand_index]
#rand_y = y_vals[:,rand_index]

grid_predictions = sess.run(prediction, feed_dict={x_data: rand_x, y_target:rand_y, prediction_grid:grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)
plt.contourf(xx, yy, grid_predictions, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class 2')
plt.plot(class3_x, class3_y, 'gv', label='Class 2')
plt.xlabel('Spel Width')
plt.ylabel('Pedal Length')
plt.legend(loc='lower right')
plt.ylim([-0.5,3.0])
plt.xlim([3.5,8.5])
plt.show()              
