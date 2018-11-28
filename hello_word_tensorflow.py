# encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder('float', [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)  # 预测值
y_ = tf.placeholder('float', [None, 10])  # 真实值

# 顺便查看预测的准确率
# argmax, 那个位置更接近1
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 下面是朝着预测更准确的地方优化
# 交叉殇计算损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 朝着减小交叉熵的方向优化模型
model = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 让模型训练1000次
for i in range(1000):
    # 训练数据
    x_trains, y_trains = mnist.train.next_batch(128) # 同样是200个批次,第一次训练0.96,第二次却只有0.11?
    sess.run(model, feed_dict={
        x: x_trains,
        y_: y_trains
    })
    if i % 50 == 0:
        print(sess.run(accuracy, feed_dict={
            x: x_trains,
            y_: y_trains
        }))
        # print(sess.run(W, feed_dict={
        #     x: x_trains,
        #     y_: y_trains
        # }))
