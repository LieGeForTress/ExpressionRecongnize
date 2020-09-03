# CNN_Model
# 含3层池化层，2层全连接层
# 激活函数ReLU，采用dropout和softmax函数做分类器

import tensorflow as tf

def weight_variable(shape,n):
    # tf.truncated_normal(shape, mean, stddev)这个函数产生正态分布，均值和标准差自己设定。
    # shape表示生成张量的维度，mean是均值
    # stddev是标准差,，默认最大为1，最小为-1，均值为0
    init = tf.truncated_normal(shape,stddev=n,dtype=tf.float32)
    return init

#构建一个结构为shape矩阵，所有值初始化为0.1
def bias_variable(shape):
    init = tf.constant(0.1,shape=shape)
    return init

#卷积层，卷积遍历各个方位为1，SAME--边缘自动补0，遍历相乘
def conv2d(x,W):
    # x是卷积的图像，W是卷积核，两者都是张量
    # https://blog.csdn.net/zuolixiangfisher/article/details/80528989
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
#池化卷积结果(conv2d)，采用3X3kernel，步数为2
#SAME 周围补0，取最大值
def max_pool_2x2(x,name):
    # x是CNN第一步卷积,必须是张量
    # ksize 是池化窗口的大小， shape为[batch, height, weight, channels]
    # stride 步长，一般是[1，stride， stride，1]
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name=name)

'''
CNN Build
'''
# 含3层池化层，2层全连接层
# 激活函数ReLU，采用dropout和softmax函数做分类器
def deep_CNN(images,batch_size,n_classes):
    #---第一层卷积
    with tf.variable_scope('conv1') as scope:
        # [3,3,3,64] 一二参数指卷积核尺寸大小
        #三参数指通道数
        #最后一个指卷积核个数，即64个3X3的卷积核（3通道）
        # 1.0 即均值
        w_conv1 = tf.Variable(weight_variable([3,3,3,64],1.0),name='weights',dtype=tf.float32)
        b_conv1 = tf.Variable(bias_variable([64]),name='biases',dtype=tf.float32)
        h_conv1 = tf.nn.relu(conv2d(images,w_conv1) + b_conv1,name='conv1')

    #---第一层池化
    #3X3最大池化，步长strides为2，池化后执行lrn()操作
    #局部响应归一化，增强模型的泛化能力
    #对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元
    with tf.variable_scope('pooling_lrn') as scope:
        pool1 = max_pool_2x2(h_conv1,'pooling1')
        norm1 = tf.nn.lrn(pool1,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')

    #---第2层卷积
    with tf.variable_scope('conv2') as scope:
        w_conv2 = tf.Variable(weight_variable([3,3,64,32],0.1),name='weights',dtype=tf.float32)
        b_conv2 = tf.Variable(bias_variable([32]),name='biases',dtype=tf.float32)
        h_conv2 = tf.nn.relu(conv2d(norm1,w_conv2)+b_conv2,name='conv2')

    #---第2层池化
    with tf.variable_scope('pooling_lrn') as scope:
        pool2 = max_pool_2x2(h_conv2,'pooling2')
        norm2 = tf.nn.lrn(pool2,depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    #---第3层卷积
    with tf.variable_scope('conv3') as scope:
        w_conv3 = tf.Variable(weight_variable([3,3,32,16],0.1),name='weights',dtype=tf.float32)
        b_conv3 = tf.Variable(bias_variable([16]),name='biases',dtype=tf.float32)
        h_conv3 = tf.nn.relu(conv2d(norm2,w_conv3)+b_conv3,name='conv3')

    #---第3层池化
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = max_pool_2x2(h_conv3,'pooling3')
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

    #---第4层全连接层
    ## 128个神经元，将之前pool层的输出reshape成一行，激活函数relu()
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm3,shape=[batch_size,-1])
        dim = reshape.get_shape()[1].value
        w_fc1 = tf.Variable(weight_variable([dim,128],0.005),name='weights',dtype=tf.float32)
        b_fc1 = tf.Variable(bias_variable([128]),name='biases',dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape,w_fc1)+b_fc1,name='scope.name')

    #---第5层全连接层
    with tf.variable_scope('local4') as scope:
        w_fc2 = tf.Variable(weight_variable([128, 128], 0.005), name='weights', dtype=tf.float32)
        b_fc2 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc1, name=scope.name)

    # 对卷积结果进行dropout操作
    h_fc2_dropout = tf.nn.dropout(h_fc2,0.5)

    #Softmax回归层
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(weight_variable([128,n_classes],0.005),name='softmax_linear',dtype=tf.float32)
        biases = tf.Variable(bias_variable([n_classes]),name='biases',dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(h_fc2_dropout,weights),biases,name='softmax_linear')

    return softmax_linear

#loss计算
#传入参数：logits 网络计算输出值
# labels 真实值
def losses(logits,labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy,name='loss')
    return loss

#loss 损失值优化
def training(loss,learning_rate):
    with tf.variable_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0,name='global_step',trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op

#评价/准确率计算
#输入参数：logits---网络计算值
#标签 labels
#返回参数：accuracy 当前step的平均准确率
def evaluation(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        right = tf.nn.in_top_k(logits,labels,1)
        accuracy = tf.reduce_mean(tf.cast(right,tf.float16))
    return accuracy