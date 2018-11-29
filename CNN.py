# encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()  # 构建一个交互式session

# 初始化大量的W和b(避免大量初始化W和b,定义两个函数)
x = tf.placeholder('float', [None, 784])  # 原始数据维度
y_ = tf.placeholder('float', [None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 标准差为0.1的正太随机分布初始化,避免0梯度
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 避免加大量的卷积层和池化层,定义连个函数
def conv2d(x, W):  # tf.nn.conv2d
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 卷积层Filter的维度,输入图像是4d的,在每个维度步长都为1


# 2*2的池化层,池化层才用求最大值的方式
def max_pool_2x2(x):  # tf.nn.max_pool
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],  
                          strides=[1, 2, 2, 1], padding='SAME') # 步长在长和宽维度都是2


# 第一层 1个卷积+1个池化
W_conv1 = weight_variable([5, 5, 1, 32])  # 在5*5的小块上算出32个特征, 5*5patch大小,1 输入通道数目, 32输出通道个数, 不必关心在被卷积之后的矩阵大小,只关心有多少个通道被卷积,最后要输出多少个特征
b_conv1 = bias_variable([32])  # 每个输出通道一个偏执项

# 把输入编程4d向量  第2、第3维对应图片的宽、高 ,最后一维代表图片的颜色通道数
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层, 1024个神经元
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 输出层之前加入dropout, 训练过程中启用dropout，在测试过程中关闭dropout。
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 采用softmax作为激活函数得到输出值
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 损失函数,交叉熵减小来优化模型
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 计算准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
# 开始训练模型
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:  # 每训练100次查看一下训练的准确率
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 用测试集去验证模型
print(
    "test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# ----------训练结果------------ # 

# -------------------- # 
