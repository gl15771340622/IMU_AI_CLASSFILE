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
# 模拟正弦函数生成数据点，值域映射到0-1之间
p = 0.5*( np.sin(2*np.pi*f*t)+1 )
# plt.plot(t, p)
# plt.xlim([0, 1])
# plt.title('目标运动轨迹模拟图')
# plt.xlabel('时间/秒')
# plt.ylabel('幅度/电压')
# plt.show()

# 前2700点数据作为训练数据
training_set = p[0:points-300]
# 后300点数据作为预测数据
test_set = p[points-300:]
x_train = []  # 训练集数据空间初始化
y_train = []  # 训练集标签空间初始化
x_test = []   # 测试集数据空间初始化
y_test = []   # 测试集标签空间初始化
#  前60点预测第61点，组成数据标签对
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
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))

for i in range(60, len(test_set)):
	x_test.append(test_set[i - 60:i])
	y_test.append(test_set[i])
# 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))

# 搭建模型
model = tf.keras.Sequential([
	# 80个记忆体的RNN第一层
    SimpleRNN(80, return_sequences=True),
    # 随机舍弃0.2的记忆体
    Dropout(0.2),
	# 100个记忆体的RNN第二层
    SimpleRNN(100),
    Dropout(0.2),
    Dense(1)  # 一层全连接
])
#  模型编译，损失函数用均方误差
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')
#  权重保存路径
checkpoint_save_path = "./checkpoint/track_train.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
	print('-----load the model-----')
	model.load_weights(checkpoint_save_path)
# 保存训练参数
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')
# 为模型输入数据并进行分批次的训练
history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
# 输出模型基本信息
model.summary()

file = open('./track_weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
	file.write(str(v.name) + '\n')
	file.write(str(v.shape) + '\n')
	file.write(str(v.numpy()) + '\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()





