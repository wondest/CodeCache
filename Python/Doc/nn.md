# Nearest neighbor

## 算法思想

现将训练集看做训练模型，基于新数据点与训练集的距离来预测新数据点。

## 1-NN

让预测值与最接近的训练数据集作为同类

## k-NN

使用k个领域的加权平均

## 举例

假设样本训练集$(x_1,x_2,x_3,...,x_n)$，对应的目标值$(y_1,y_2,y_3,...,y_n)$，通过最近领域法预测点$z$：

### 离散型分类目标

$$
f(z) = \max_{j}\sum_{i=1}^k\varphi(d_{ij})I_{ij}
$$

其中：预测函数$f(z)$是所有分类$j$上的最大加权值，预测数据点到训练数据点$i$的距离用$\varphi(d_{{ij}})$表示，$I_{ij}$是指示函数，表示数据点$i$是否是属于分类$j$。

### 连续回归训练目标

预测值是所有k个最近领域数据点到预测数据点的加权平均公式：
$$
f(z) = \frac{1}{k}\sum_{i=1}^k\varphi(d_{i})
$$


### 训练数据

[Housing data](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data)