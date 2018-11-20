# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:24:32 2018

@author: Tender


结合：既包含文本特征 又 包含数值特征的数据观测点间的距离

地址匹配中，地址中有许多打印错误，不同的城市或者不同的邮政编码，其实指向同一个地址

使用最近领域算法综合地址信息的数值部分和字符部分可以帮助鉴定实际相同的地址。

本例子中生成两个模拟数据，每个数据集包含街道地址和邮政编码。其中有一个数据集的街道地址有大量的打印错误。

我们将准确的地址数据集作为“标准”，为每个有打印错误的地址返回一个最接近的地址，采用综合字符距离（街道）和数值距离（邮政编码）

的距离函数度量地址间的相似度

"""

import random
import string
import numpy as np
import tensorflow as tf

"""
Tensorflow session
"""
sess = tf.Session()

"""
创建参考数据集
"""

n = 10
street_names = ['abbey', 'baker', 'canal', 'donner', 'elm']
street_types = ['rd', 'st', 'ln', 'pass', 'ave']
rand_zips = [random.randint(65000, 65999) for i in range(n)]

streets = [random.choice(street_names) for i in range(n)]
numbers = [random.randint(1, 9999) for i in range(n)]
street_suffs = [random.choice(street_types) for i in range(n)]
    
zips = [random.choice(rand_zips) for i in range(n)]
full_streets = [str(x) + ' ' + y + ' ' + z for x,y,z in zip(numbers, streets, street_suffs)]
reference_data = [list(x) for x in zip(full_streets, zips)]

"""
创建一个测试数集合 
"""
def create_typo(s, prob=0.75):
    if random.uniform(0,1) < prob:
        rand_ind = random.choice(range(len(s)))
        s_list = list(s)
        s_list[rand_ind] = random.choice(string.ascii_lowercase)
        s = ''.join(s_list)
    return(s)


typo_streets = [create_typo(x) for x in streets]
typo_full_streets = [str(x) + ' ' + y + ' ' + z for x,y,z in zip(numbers, typo_streets, street_suffs)]
test_data = [list(x) for x in zip(typo_full_streets, zips)]

"""
Tensorflow placeholder
"""
test_address = tf.sparse_placeholder(dtype = tf.string)
test_zip = tf.placeholder(shape=[None, 1], dtype=tf.float32)

ref_address = tf.sparse_placeholder(dtype = tf.string)
ref_zip = tf.placeholder(shape=[None, 1], dtype=tf.float32)

"""
距离函数
"""
zip_dist = tf.square(tf.subtract(ref_zip, test_zip))
address_dist = tf.edit_distance(test_address, ref_address, normalize=True)

"""
把距离转换成相似度：1 为完全相似；0 为完全不一致

邮政编码：sub/(max-min)
地址相似度：1 - address_dist
"""

zip_max = tf.gather(tf.squeeze(zip_dist), tf.argmax(zip_dist, 0))
zip_min = tf.gather(tf.squeeze(zip_dist), tf.argmin(zip_dist, 0)) 
zip_sim = tf.divide(tf.subtract(zip_max, zip_dist), tf.subtract(zip_max, zip_min))

address_sim = tf.subtract(1., address_dist)

"""
使用加权
"""
address_weight = 0.5
zip_weight = 1. - address_weight
weighted_sim = tf.add(tf.multiply(address_weight, address_sim), tf.multiply(zip_weight, zip_sim))

top_match_index = tf.argmax(weighted_sim, 0)


"""
"""
def create_sparse_vec(word_list):
    num_words = len(word_list)
    indices = [[xi, 0, yi] for xi,x in enumerate(word_list) for yi,y in enumerate(x)]
    chars = list(''.join(word_list))

    return(tf.SparseTensorValue(indices, chars, [num_words,1,1]))

reference_addresses = [x[0] for x in reference_data]
reference_zips = np.array([[x[1] for x in reference_data]])

sparse_ref_set = create_sparse_vec(reference_addresses)

for i in range(n):
    test_address_entry = test_data[i][0]
    test_zip_entry = np.transpose([[test_data[i][1]]])
    
    test_address_repeated = [test_address_entry] * n
    sparse_test_set = create_sparse_vec(test_address_repeated)
    
    feed_dict={test_address: sparse_test_set,
               test_zip: test_zip_entry,
               ref_address: sparse_ref_set,
               ref_zip: np.transpose(reference_zips)}
    
    [best_match] = sess.run(top_match_index, feed_dict=feed_dict)
    best_street = reference_addresses[best_match]
    
    best_zip = reference_zips[0][best_match]
    [[test_zip_]] = test_zip_entry
    
    print('Address: ' + str(test_address_entry) + ',' + str(test_zip_))
    print('Match : ' + str(best_street) + ',' + str(best_zip))

