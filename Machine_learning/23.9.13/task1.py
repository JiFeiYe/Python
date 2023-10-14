# 创建数据集,把数据写⼊到numpy数组
import numpy as np  # 引⽤numpy库,主要⽤来做科学计算
import matplotlib.pyplot as plt  # 引⽤matplotlib库,主要⽤来画图

# 定义数据,总共10个样本,每个样本包含两个值,分别为身⾼和体重。
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
# 打印出数组的⼤⼩
print(data.shape)
# 从data中提取身⾼和体重的值,分别存放在x, y变量中。
# data[:,0]指的是取出所有第⼀列,也就是身⾼特征。
x, y = data[:, 0].reshape(-1, 1), data[:, 1]
# 在⼆维空间⾥画出身⾼和体重的分布图
plt.scatter(x, y, color="black")
plt.xlabel("height (cm)")
plt.ylabel("weight (kg)")
plt.show()
