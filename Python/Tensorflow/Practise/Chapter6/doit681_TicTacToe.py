# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 17:32:29 2018

@author: Tender

使用神经网络来学习和优化井字棋

https://github.com/nfmcclure/tensorflow_cookbook

https://github.com/nfmcclure/tensorflow_cookbook/tree/master/06_Neural_Networks/08_Learning_Tic_Tac_Toe

"""

"""
1. Import libraries
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import random
import numpy as np

"""
2. Batch_size
"""
batch_size = 50

"""
3. print board[0:8]: -1 - 'O'   0 - ' '  1 - 'X'
"""
def print_board(board):
    symbols = ['o',' ','x']
    board_plus_1 = [int(x) + 1 for x in board[0:9]]
    print(board_plus_1)
    print(' ' + symbols[board_plus_1[0]] + ' | ' + symbols[board_plus_1[1]] + ' | ' + symbols[board_plus_1[2]])
    print('___________')
    print(' ' + symbols[board_plus_1[3]] + ' | ' + symbols[board_plus_1[4]] + ' | ' + symbols[board_plus_1[5]])
    print('___________')
    print(' ' + symbols[board_plus_1[6]] + ' | ' + symbols[board_plus_1[7]] + ' | ' + symbols[board_plus_1[8]])
    
"""
4. get_symmetry
"""
def get_symmetry(board, response, transformation):
    """
    description
    
    Parameters
    ----------
    board : array
        The name of the backend to use.
    
    response : str
        The
        
    transformation : str
        rotate180,rotate190,rotate1270,flip_v,flip_h
        
    Notes
    -----
    所有的棋局都可以通过旋转两次得到

    """
    if transformation == 'rotate180':
        new_response = 8 - response
        return(board[::-1], new_response)
        
    elif transformation == 'rotate90':
        new_response = [6, 3, 0, 7, 4, 1, 8, 5, 2].index(response)
        tuple_board = list(zip(*[board[6:9], board[3:6], board[0:3]]))
        return([value for item in tuple_board for value in item], new_response)
        
    elif transformation == 'rotate270':
        new_response = [2, 5, 8, 1, 4, 7, 0, 3, 6].index(response)
        tuple_board = list(zip(*[board[0:3], board[3:6], board[6:9]]))[::-1]
        return([value for item in tuple_board for value in item], new_response)
        
    elif transformation == 'flip_v':
        new_response = [6, 7, 8, 3, 4, 5, 0, 1, 2].index(response) 
        return(board[6:9] + board[3:6] + board[0:3], new_response)

    elif transformation == 'flip_h':
        new_response = [2, 1, 0, 5, 4, 3, 8, 7, 6].index(response)
        new_board = board[::-1]
        return(new_board[6:9] + new_board[3:6] + new_board[0:3], new_response)

    else:
        raise ValueError('Method not implemented.')

"""     
5. get_moves_from_csv
"""
def get_moves_from_csv(csv_file):
    moves=[]
    with open(csv_file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            moves.append(([int(x) for x in row[0:9]], int(row[9])))
    return(moves)
    
"""
6. get_rand_move
"""
def get_rand_move(moves, rand_transforms=2):
    (board, response) = random.choice(moves)
    possible_transforms = ['rotate90', 'rotate180', 'rotate270', 'flip_v', 'flip_h']
    for i in range(rand_transforms):
        random_transform = random.choice(possible_transforms)
        (board, response) = get_symmetry(board, response, random_transform)
    return(board, response)
    
    
"""
7. Init
"""
sess = tf.Session()
moves = get_moves_from_csv('base_tic_tac_toe_moves.csv')

# Create a train set:
train_length = 500
train_set = []
for t in range(train_length):
    train_set.append(get_rand_move(moves))

"""
8. 
"""   
test_board = [-1, 0, 0, 1, -1, -1, 0, 0, 1]
train_set = [x for x in train_set if x[0] != test_board]

"""
9. 
"""
def init_weight(shape, st_dev=1.0):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(weight)
    
def init_bias(shape, st_dev=1.0):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(bias)
    
def fully_connected(input_layer, weights, biases, activation = True):
    linear_layer = tf.add(tf.matmul(input_layer, weights), biases)
    
    if activation:
        return(tf.nn.sigmoid(linear_layer))
    else:
        return(linear_layer)

"""
10. 模型输出中并未包含 softmax函数，因为softmax会在损失函数中出现
"""
x_data = tf.placeholder(shape=[None, 9], dtype=tf.float32)
y_target = tf.placeholder(shape=[None], dtype=tf.int32)

# Create first layer(81 hidden nodes)
weight_1 = init_weight(shape=[9, 81])
bias_1 = init_bias(shape=[81])
layer_1 = fully_connected(x_data, weight_1, bias_1)

# 
weight_2 = init_weight(shape=[81, 9])
bias_2 = init_bias(shape=[9])
final_output = fully_connected(layer_1, weight_2, bias_2, activation=False)


"""
11. Loss: sparse_softmax_cross_entropy_with_logits 干啥的？多分类器
"""
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_output, labels=y_target))
my_opt = tf.train.GradientDescentOptimizer(0.025)
train_step = my_opt.minimize(loss)

prediction = tf.argmax(final_output, 1)

"""
12. 
"""
# init tensorflow variables
init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
test_loss = []

for i in range(10000):
    # First we select a random set of indices for the batch.
    rand_index = np.random.choice(len(train_set), size=batch_size, replace=False)
    
    # select the training data
    batch_data = [train_set[i] for i in rand_index]
    
    rand_x = [x[0] for x in batch_data]
    rand_y = [y[1] for y in batch_data]
    
    # Now we run the training step
    sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
    
    # We save the training loss
    temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})
    loss_vec.append(temp_loss)
    
    if (i+1)%500==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))


"""
13.
"""
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')

plt.title('Loss(MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


"""
1. 
"""
test_boards = [test_board]
feed_dict = {x_data:test_boards}
test_output = sess.run(final_output, feed_dict=feed_dict)
test_pred = sess.run(prediction, feed_dict=feed_dict)
print(test_pred)


"""
2.
"""
def checkWin(board):
    wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    
    for i in range(len(wins)):
        if board[wins[i][0]]== board[wins[i][1]]== board[wins[i][2]]==1.:
            return(1)
        elif board[wins[i][0]]== board[wins[i][1]]== board[wins[i][2]]==-1.:
            return(1)
    
    return(0)
    
"""
3.对弈
"""
game_tracker = [0., 0., 0., 0., 0., 0., 0., 0.,0.]

win_logical = False

num_moves = 0

while not win_logical:
    # Player move
    player_index = input('Input index of your move (0-8): ')
    num_moves = 0
    game_tracker[int(player_index)] = 1
    
    # Model move
    [potential_moves] = sess.run(final_output, feed_dict={x_data:[game_tracker]})
    
    # 
    allowed_moves = [ix for ix,x in enumerate(game_tracker) if x==0.0]
    
    model_move = np.argmax([x if ix in allowed_moves else -999.0 for ix,x in enumerate(potential_moves)])
    
    game_tracker[int(model_move)] = -1.
    
    print('Model has moved')
    print_board(game_tracker)
    
    if checkWin(game_tracker) == 1 or num_moves >=5:
        print('Game over')
        win_logical = True