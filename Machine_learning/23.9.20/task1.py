import numpy as np
from sklearn.linear_model import LinearRegression

# 生成样本数据，特征维度为2
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
# 先试用sklearn自带的库来解决
model = LinearRegression().fit(X, y)
# 打印参数以及偏移量（bias）
print("基于sklearn的线性回归模型参数为coef：", model.coef_, " intercept:%.5f" % model.intercept_)
# 手动实现参数的求解。先把偏移量加到X里
# TODO:首先生成一个只有1列的一维数组， 每个元素都是1，行数为样本个数
a = np.ones((4, 1))
# TODO:然后更新矩阵X使其第一列全为1
X = np.concatenate((np.ones(4).reshape(-1, 1), X), axis=1)
# TODO:通过矩阵、向量乘法来求解参数，res已经包含了偏移量
res = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
# 打印参数偏移量（bias）
print("通过手动实现的线性回归模型参数为coef：", res[1:], " intercept：%.5f" % res[0])
