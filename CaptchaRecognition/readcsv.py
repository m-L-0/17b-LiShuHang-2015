import csv
import matplotlib.pyplot as plt

# 读取csv文件
csvFile = open("label/labels.csv", "r")
reader = csv.reader(csvFile)  # 返回的是迭代类型
i = 0
j = 0
k = 0
n = 0
f = open('/home/vbuo/m-L-2/labels', 'w')
for item in reader:
    print(item)
    f.write(str(item))
    f.write('\n')
    if len(item[1]) == 1:
        i = i+1
    elif len(item[1]) == 2:
        j = j+1
    elif len(item[1]) == 3:
        k = k+1
    else:
        n = n+1
num = i+j+k+n

# 打印关键数据
print('1位的验证码数量：', i, '比例:', i/num)
print('2位的验证码数量：', j, '比例:', j/num)
print('3位的验证码数量：', k, '比例:', k/num)
print('4位的验证码数量：', n, '比例:', n/num)
print('验证码总数：', num)
print('各类所占比例')

data = [
    i, j, k, n
]
mark = [
    'one bit verification code', 'two bit verification code', 
    'three bit verification code', 'four bit verification code'
]
# 打印直方图
plt.figure(figsize=(10, 5))
plt.xlabel('labes')
plt.ylabel('count')
plt.title("Quantity distribution diagram")
plt.bar(mark, data)
plt.show()

# 打印饼状图
plt.figure(figsize=(5, 5))
plt.axes(aspect=1)
plt.pie(x=data, labels=mark, autopct='%.0f%%', pctdistance=0.7)
plt.title("Proportional distribution diagram")
plt.show()

csvFile.close()