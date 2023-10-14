# 创建数据集,把数据写⼊到numpy数组
import numpy as np  # 引⽤numpy库,主要⽤来做科学计算
import matplotlib.pyplot as plt  # 引⽤matplotlib库,主要⽤来画图

data = np.array(
    [
        [152, 51],
        [156, 53],
        [160, 54],
        [164, 55],
        [168, 57],
        [172, 60],
        [176, 62],
        [180, 65],
        [184, 69],
        [188, 72],
    ]
)
# 打印⼤⼩
x, y = data[:, 0], data[:, 1]
print(x.shape, y.shape)
# 1. ⼿动实现⼀个线性回归算法，具体推导细节参考刚才5-6节的理论课内容
# TODO: 实现w和b参数, 这⾥w是斜率, b是偏移量
e_x = np.mean(x) # e_x:x的均值,下同
e_y = np.mean(y)
e_xy = np.mean(x*y)
e_x2 = np.mean(x*x)
w = (e_xy - e_x *e_y) / (e_x2 - (e_x**2))
b = e_y - w*e_x
print("通过⼿动实现的线性回归模型参数: %.5f %.5f" % (w, b))
# 2. 使⽤sklearn来实现线性回归模型, 可以⽤来⽐较⼀下跟⼿动实现的结果
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(x.reshape(-1, 1), y)
print("基于sklearn的线性回归模型参数:%.5f %.5f" % (model.coef_, model.intercept_))
