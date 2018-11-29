# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import jieba
import re
from string import punctuation

#文章全部词语分布
words=''
corpus={}
#语料实体
reviews = []
#语料正负项
label=[]
corpus_len=0
#lstm 个数
lstm_size=128
#lstm 层数
lstm_layers = 2
batch_size= 200
learning_rate = 0.01
LR = 0.006

def get_batches(x, y, batch_size=100): # 每次训练100个样本
    n_batches = len(x) // batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]

    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


def dataInit (corpusFn):
    global words,reviews,neg_len,pos_len,corpus,label,corpus_len
    fileTrainRead = []
    with open(corpusFn,'r') as nfn:
        for line in nfn.readlines():
            corpus_tab_arr = line.split('\t')
            corpus_body = ''.join(corpus_tab_arr[:len(corpus_tab_arr)])
            corpus_label = corpus_tab_arr[-1]
            corpus[corpus_body] = corpus_label
        all_text = ''
        for body in corpus:
            body_drop_punctuation = re.sub("[$\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",body.strip())
            wordseg = list(jieba.cut(body_drop_punctuation, cut_all=False, HMM=True))
            if len(wordseg) != 0:
                corpus_len += 1
                str_tmp = ' '.join(wordseg)
                all_text += str_tmp + ' '
                reviews.append(str_tmp)
                label.append(corpus[body].replace("\n",""))
            else:
                corpus.pop(body)
        words = all_text.split()

from collections import Counter

# 语料的所有单词,和单词的出现的个数
def get_vocab_to_int(words):
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get,reverse=True)
    vocab_to_int = {word : i for i,word in enumerate(vocab,0)}
    return vocab_to_int

# 每段语料中的词语在总文本中有多少数量
def get_reviews_ints(vocab_to_int, reviews):
    reviews_ints = []
    for each in reviews:
        reviews_ints.append([vocab_to_int[word] for word in each.split()]) 
    return reviews_ints # [[1,2,3,4,....],[2,2,2,3,4...]]

def lstm_cell():
    global lstm_size
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size,forget_bias=1.0)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob)
    return drop

if __name__ == '__main__':
    #dataInit('/Users/lvyifu/Desktop/reviews.csv')
    # for test we use corpus_test.csv
    dataInit('/Users/lvyifu/Desktop/corpus_test.csv')
    vocab_to_int = get_vocab_to_int(words)
    reviews_ints = get_reviews_ints(vocab_to_int, reviews)
    print("the vocab size is :" + str(len(vocab_to_int))) # 输出有多少个词
    #print get_reviews_ints(vocab_to_int, [u"非常 好？"])
    labels= np.array(label,dtype=np.int8)

    seq_len= 200 
    features = np.zeros((len(reviews_ints),seq_len), dtype=np.int) # 200维度的矩阵,行数为语料样本数
    for i, review in enumerate(reviews_ints):
        if i % 1000 == 0 :
            print (i)
        features[i, -len(review):] = np.array(review)[:seq_len]

    split_frac = 0.8 # 测试集和训练集比例划分
    split_index = int(len(features) * split_frac)
    train_x, val_x = features[:split_index], features[split_index:]
    train_y, val_y = labels[:split_index], labels[split_index:]

    test_index = int(len(val_x) * 0.5)
    val_x, test_x = val_x[:test_index], val_x[test_index:]
    val_y, test_y = val_y[:test_index], val_y[test_index:]

    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    n_words = len(vocab_to_int)
    #定义输入输出
    graph = tf.Graph()
    with graph.as_default():
        inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
        labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    #定义嵌入层，防止过拟合
    embed_size = 300
    with graph.as_default():
        embedding = tf.Variable(tf.truncated_normal((n_words, embed_size), stddev=0.01))
        embed = tf.nn.embedding_lookup(embedding, inputs_)
    #定义lstm
    #there is a bug in there
    #https://github.com/tensorflow/tensorflow/issues/16186
    # 定义两层的LSTM
    with graph.as_default():
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)]) # 定义多层LSTM
        initial_state = cell.zero_state(batch_size, tf.float32) # 用0初始化Cell的state_size


    with graph.as_default(): # 通过制定的RNN Cell 开始计算神经网络
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state) 
        # finnal_state: cell1 [batch_size, lstm_size]
        # outputs : [batch_size, lstm_size]

    with graph.as_default(): #  得到预测值和误差,并用优化器向误差减小的方向训练
        # cell处理后的数据经过全连接层, 输出维度 1
        predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid) 
        cost = tf.losses.mean_squared_error(labels_, predictions) # 损失函数, 采用均方误差, 分类只有两个,可以当做是一个回归问题
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    with graph.as_default(): # 计算预测的准确率
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    epochs = 10

    # 持久化，保存训练的模型
    # 制作流程图, 以及打印在训练过程中误差情况
    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        iteration = 1

        for e in range(epochs):
            state = sess.run(initial_state)

            for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob: 0.5,
                        initial_state: state}

                loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

                if iteration % 5 == 0:
                    print('Epoch: {}/{}'.format(e, epochs),
                          'Iteration: {}'.format(iteration),
                          'Train loss: {}'.format(loss))

                if iteration % 25 == 0:
                    val_acc = []
                    val_state = sess.run(cell.zero_state(batch_size, tf.float32))

                    for x, y in get_batches(val_x, val_y, batch_size):
                        feed = {inputs_: x,
                                labels_: y[:, None],
                                keep_prob: 1,
                                initial_state: val_state}

                        batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                        val_acc.append(batch_acc)

                    print('Val acc: {:.3f}'.format(np.mean(val_acc)))

                iteration += 1

        saver.save(sess, "checkpoints/sentiment.ckpt")
        writer = tf.summary.FileWriter(r"./path/to/log", tf.get_default_graph())
        writer.close()

        test_acc = []
        with tf.Session(graph=graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            test_state = sess.run(cell.zero_state(batch_size, tf.float32))
            for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob: 1,
                        initial_state: test_state}
                batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.3f}".format(np.mean(test_acc)))





