import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

# 当输出数组数目过大时，只显示一部分
np.set_printoptions(threshold=np.inf)

# 导入手写数据集。训练集和测试集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 对数据集的像素点进行归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# 搭建训练模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),  # 拉直层
    # 全连接层：128个神经元，激活函数relu
    tf.keras.layers.Dense(128, activation='relu'),
    # 输出层：10个神经元，激活函数softmax
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
# 设置断点续训保存参数的路径
checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
# 给模型送入训练数据并分批次训练
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()  # 控制台输出模型信息
# 输出可训练参数
print(model.trainable_variables)
# 保存参数为文本
file = open('./mnist_weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
