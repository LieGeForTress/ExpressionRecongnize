# PreWork
# 制作图片数据集，获得神经网络输入的 get_files 文件
# 同时为了方便网络的训练，对输入数据进行 batch 处理

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import *

angry = []
disgusted = []
fearful = []
happy = []
sadness = []
surprised = []
label_angry = []
label_disgusted = []
label_fearful = []
label_happy = []
label_sadness = []
label_surprised = []

def get_file(fdir):
    #---第一步，获取路径下所有的图片，存入list
    for file in os.listdir(fdir+'/angry'):
        angry.append(fdir+'/angry'+'/'+file)
        label_angry.append(0)
    for file in os.listdir(fdir+'/disgusted'):
        angry.append(fdir+'/disgusted'+'/'+file)
        label_angry.append(1)
    for file in os.listdir(fdir+'/fearful'):
        angry.append(fdir+'/fearful'+'/'+file)
        label_angry.append(2)
    for file in os.listdir(fdir+'/happy'):
        angry.append(fdir+'/happy'+'/'+file)
        label_angry.append(3)
    for file in os.listdir(fdir+'/sadness'):
        angry.append(fdir+'/sadness'+'/'+file)
        label_angry.append(4)
    for file in os.listdir(fdir+'/surprised'):
        angry.append(fdir+'/surprised'+'/'+file)
        label_angry.append(5)
    print("angry:%d disgusted:%d fearful:%d happy:%d sadness:%d surprised:%d" % (len(angry),len(disgusted),len(fearful),len(happy),len(sadness),len(surprised)))

    #---第2步 合并+打乱处理
    img_list = np.hstack((angry,disgusted,fearful,happy,sadness,surprised))
    img_label_list = np.hstack((label_angry,label_disgusted,label_fearful,label_happy,label_sadness,label_surprised))
    temp =np.array([img_list,img_label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    #将所有的 img 和label 转成 list
    all_img_list = list(temp[:,0]) #取出第0列，即图片路径
    all_label_list = list(temp[:,1]) #取出第1列，即标签
    label_list = [int(i) for i in img_label_list]

    # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * 0.8))  # 测试样本数, ratio是测试集的比例
    n_train = n_sample - n_val  # 训练样本数
    tra_images = all_img_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]  # 转换成int数据类型
    val_images = all_img_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]  # 转换成int数据类型
    return tra_images, tra_labels, val_images, val_labels

# 为了方便网络的训练，输入数据进行batch处理
# image_W, image_H, ：图像高度和宽度
# batch_size：每个batch要放多少张图片
# capacity：一个队列最大多少
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    #---第1步，将上面生成的List 传入 get_batch()，转换类型，产生一个输入队列
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    # tf.train.slice_input_producer是一个tensor生成器
    # 作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    #从队列中读取图像

    #---第2步,解码图像,使用相同的图像
    image = tf.image.decode_jpeg(image_contents,channels=3)
    # 0：使用JPEG编码图像中的通道数量.
    # 1：输出灰度图像.
    # 3：输出RGB图像.
    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档

    #---第3步，数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 对resize后的图片进行标准化处理
    image = tf.image.per_image_standardization(image)

    #---第4步，生成batch
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    # image_batch = tf.cast(image_batch, tf.uint8)    # 显示彩色图像
    image_batch = tf.cast(image_batch, tf.float32)  # 显示灰度图
    # print(label_batch) Tensor("Reshape:0", shape=(6,), dtype=int32)
    return image_batch, label_batch
    # 获取两个batch，两个batch即为传入神经网络的数据