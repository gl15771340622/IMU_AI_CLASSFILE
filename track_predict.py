""" -*-coding : utf-8 -*-
    Author    : GaoLiang
    System    ：win10
    Date_Time ：2020/6/8 20:32
    File_Name ：track_predict.py
    Dev_Tool  ：PyCharm
    Country   : China
    Aim       : 
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import os
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN

#  解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
points = 3000
f = 10
t = np.linspace(0, 3, points)
p = 0.5*( np.sin(2*np.pi*f*t)+1 )
# plt.plot(t, p)
# plt.xlim([0, 1])
# plt.title('目标运动轨迹模拟图')
# plt.xlabel('时间/秒')
# plt.ylabel('幅度/电压')
# plt.show()

training_set = p[0:points-300]
test_set = p[points-300:]

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(60, len(training_set)):
	x_train.append( training_set[i - 60 : i ])
	y_train.append( training_set[i] )
# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)
# 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
# 此处整个数据集送入，送入样本数为x_train.shape[0]即2066组数据；输入60个开盘价，预测出第61天的开盘价，循环核时间展开步数为60; 每个时间步送入的特征是某一天的开盘价，只有1个数据，故每个时间步输入特征个数为1
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))

for i in range(60, len(test_set)):
	x_test.append(test_set[i - 60:i])
	y_test.append(test_set[i])
# 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))

model = tf.keras.Sequential([
    SimpleRNN(80, return_sequences=True),
    Dropout(0.2),
    SimpleRNN(100),
    Dropout(0.2),
    Dense(1)
])

checkpoint_save_path = "./checkpoint/track_train.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
	print('-------------load the model-----------------')
	model.load_weights(checkpoint_save_path)

	################## predict ######################
	# 测试集输入模型进行预测
	predicted_track = model.predict(x_test)
	real_track = y_test
	# 画出真实数据和预测数据的对比曲线
	plt.plot(real_track, color='blue', label='实际点迹')
	plt.plot(predicted_track, color='red', linestyle=':', label='预测点迹', linewidth=3)
	plt.title('实际点迹与预测点迹对比图')
	plt.xlabel('时间点索引')
	plt.ylabel('点迹值')
	plt.legend()
	plt.show()

