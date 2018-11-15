
# Python Numpy broadcasting


广播用以描述numpy中对两个形状不同的阵列进行数学计算的处理机制。较小的阵列“广播”到较大阵列相同的形状尺度上，使它们对等以可以进行数学计算。广播提供了一种向量化阵列的操作方式，因此Python不需要像C一样循环。广播操作不需要数据复制，通常执行效率非常高。然而，有时广播是个坏主意，可能会导致内存浪费以致计算减慢。

## 形状一样

Numpy操作通常由成对的阵列完成，阵列间逐个元素对元素地执行。最简单的情形是两个阵列有一样的形状，例如：

```
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = np.array([2.0, 2.0, 2.0])
>>> a * b
array([ 2.,  4.,  6.])
```

## 形状不一样

Numpy的广播机制放宽了对阵列形状的限制。最简单的情形是一个阵列和一个尺度值相乘：
```
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = 2.0
>>> a * b
array([ 2.,  4.,  6.])
```
上面两种结果是一样的，我们可以认为尺度值b在计算时被延展得和a一样的形状。延展后的b的每一个元素都是原来尺度值的复制。延展的类比只是一种概念性的。实际上，Numpy并不需要真的复制这些尺度值，所以广播运算在内存和计算效率上尽量高效。

上面的第二个例子比第一个更高效，因为广播在乘法计算时动用更少的内存。

## General Broadcasting Rules

对两个阵进行操作时，NumPy逐元素地比较他们的形状，从后面的维度向前执行。当以下情形出现时，两个维度是兼容的：

1. 它们相等
2. 其中一个是1

如果这些条件都没有达到，将会抛出错误：frames are not aligned exception，表示两个阵列形状不兼容。结果阵列的尺寸与输入阵列的各维度最大尺寸相同。

阵列不需要有相同的维度。例如，如果你有一个256x256x3的RGB阵列，你想要对每一种颜色加一个权重，你就可以乘以一个拥有3个元素的一维阵列。将两个阵列的各维度尺寸展开，从后往前匹配，如果满足了上面的两个条件，则这两个阵列是兼容的。

```
Image  (3d array): 256 x 256 x 3
Scale  (1d array):             3
Result (3d array): 256 x 256 x 3
```

当任何一个维度是1，那么另一个不为1的维度将被用作最终结果的维度。也就是说，尺寸为1的维度将延展或“复制”到与另一个维度匹配。

下面的例子，A和B两个阵列中尺寸为1的维度在广播过程中都被拓展了
```
A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
```

更多例子：
```
A      (2d array):  5 x 4
B      (1d array):      1
Result (2d array):  5 x 4

A      (2d array):  5 x 4
B      (1d array):      4
Result (2d array):  5 x 4

A      (3d array):  15 x 3 x 5
B      (3d array):  15 x 1 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 5
Result (3d array):  15 x 3 x 5

A      (3d array):  15 x 3 x 5
B      (2d array):       3 x 1
Result (3d array):  15 x 3 x 5
```

下面这些例子不能广播
```
A      (1d array):  3
B      (1d array):  4 # trailing dimensions do not match  #维度尺寸不兼容

A      (2d array):      2 x 1
B      (3d array):  8 x 4 x 3 # second from last dimensions mismatched #倒数第二个维度不兼容
```