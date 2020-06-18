from PIL import Image
import numpy as np
import tensorflow as tf
# 断点续训参数保存路径
model_save_path = './checkpoint/mnist.ckpt'
# 搭建模型，拉直层、128全连接、10输出层
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

# 载入之前训练的参数
model.load_weights(model_save_path)

# 控制台输入要验证手写数字图片的个数
preNum = int(input("input the number of test pictures:(p<=9)\n"))

for i in range(preNum):
    # 输入要验证手写图片的路径
    image_path = input("the path of test picture:\n")
    # 打开指定的手写数字图片
    img = Image.open(image_path)
    # 调整图片形状
    img = img.resize((28, 28), Image.ANTIALIAS)
    # 灰度化处理
    img_arr = np.array(img.convert('L'))
    # 图片黑白底变换
    img_arr = 255 - img_arr
    # 归一化
    img_arr = img_arr / 255.0
    # 输出调整后的形状
    print("img_arr:", img_arr.shape)
    # 插入一个图片数量的维度
    x_predict = img_arr[tf.newaxis, ...]
    print("x_predict:", x_predict.shape)
    # 模型对指定图片进行分类
    result = model.predict(x_predict)
    # 模型预测的结果中最大的值对应的索引就是0-9分类的结果
    pred = tf.argmax(result, axis=1)
    print('\n')
    tf.print(pred)  # 输出预测结果
