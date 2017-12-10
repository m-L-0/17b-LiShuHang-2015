import os
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
path = '/home/vbuo/m-L-1/车牌字符识别训练数据/字母数字'
classes = os.listdir(path)
#
x = [i[0] for i in enumerate(classes)]

data = []
count = 0
ratio = []
mark = []

for i in classes:
    data.append(len(os.listdir(path + '/' + i)))  # 每个字符数量
    count = count + len(os.listdir(path + '/' + i))  # 计算字符总数
for i in data:
    ratio.append(i/count)  # 计算每个字符所占比例
for i in classes:
    mark.append(i)  # 打印标签

# 打印关键数据
print('所对应标签' + str(mark))
print('所对应数量' + str(data))
print('各标签所占比例' + str(ratio))
print('数据总数' + str(count))

# 打印直方图
plt.figure(figsize=(20, 10))
plt.xlabel('labes')
plt.ylabel('count')
plt.title("Quantity distribution diagram")
plt.bar(x, data)
plt.xticks(x, mark, rotation=0)
# plt.bar(x, data, 0.4, color="green")
plt.show()

# 打印饼状图
plt.figure(figsize=(10, 10))
plt.axes(aspect=1)
plt.pie(x=data, labels=mark, autopct='%.0f%%', pctdistance=0.7)
plt.title("Proportional distribution diagram")
plt.show()
