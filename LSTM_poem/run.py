import tensorflow as tf
import collections
import numpy as np
import collections
# lstm训练生成唐诗

poetry_file = 'data/poetry.txt'
rnn_size = 128
num_layers = 2

# 诗集
poetrys = []
with open(poetry_file, "r", encoding='utf-8') as f:
    for line in f:
        if len(line.strip().split(':')) > 2:
            continue
        title, content = line.strip().split(':')
        content = content.replace(' ', '')
        if '_' in content or '(' in content or u'（' in content or '《' in content or '[' in content:  # 包含特殊字符的不要
            continue
        if len(content) < 5 or len(content) > 79:  # 太长的不要,太短的不要
            continue
        content = '[' + content + ']'  # 只要诗的内容
        poetrys.append(content)

# -----------
poetrys = sorted(poetrys, key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))

# 统计每个字出现次数
all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]
all_words.append(' ')  # 加上这个字符
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)

# 取前多少个常用字
# words = words[:len(words)] + (' ',)
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]

# -----------
# 每次取64首诗进行训练
batch_size = 64
n_chunk = len(poetrys_vector) // batch_size


class DataSet(object):
    def __init__(self, data_size):
        self._data_size = data_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(data_size)

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if start + batch_size > self._data_size:
            np.random.shuffle(self._data_index)
            self._epochs_completed = self._epochs_completed + 1
            self._index_in_epoch = batch_size
            full_batch_features, full_batch_labels = self.data_batch(0, batch_size)
            return full_batch_features, full_batch_labels
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            full_batch_features, full_batch_labels = self.data_batch(start, end)
            if self._index_in_epoch == self._data_size:
                self._index_in_epoch = 0
                self._epochs_completed = self._epochs_completed + 1
                np.random.shuffle(self._data_index)
            return full_batch_features, full_batch_labels

    def data_batch(self, start, end):
        batches = []
        for i in range(start, end):
            batches.append(poetrys_vector[self._data_index[i]])

        length = max(map(len, batches))

        xdata = np.full((end - start, length), word_num_map[' '], np.int32)
        for row in range(end - start):
            xdata[row, :len(batches[row])] = batches[row]
        ydata = np.copy(xdata)
        ydata[:, :-1] = xdata[:, 1:]
        return xdata, ydata


input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])  # 诗词的长度不设置限制, ?? 不是

# 定义RNN
#     cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True) contrib中没有函数了??
# cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size, state_is_tuple=True)
# cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0)
cell = tf.contrib.rnn.MultiRNNCell([lstm] * 2)

# initial_state = cell.zero_state(batch_size, tf.float32)
initial_state = cell.zero_state(batch_size, tf.float32)
embedding = tf.Variable(tf.random_uniform(shape=[len(words), rnn_size], minval=-1.0, maxval=1.0))
# embedding = tf.get_variable("embedding", [len(words), rnn_size])# rnn_size 词向量的维度
inputs = tf.nn.embedding_lookup(embedding, input_data)

outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
output = tf.reshape(outputs, [-1, rnn_size])  # 这里的含义是, 最后得到多少个词,

# 全连接层
#     softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)])# 映射到词汇表
softmax_w = tf.Variable(tf.truncated_normal(shape=[rnn_size, len(words)]))
#     softmax_b = tf.get_variable("softmax_b", [len(words)])
softmax_b = tf.Variable(tf.zeros(shape=[len(words)]))
logits = tf.nn.bias_add(tf.matmul(output, softmax_w), softmax_b)  # 映射到词汇表
targets = tf.one_hot(tf.reshape(output_targets, [-1]), depth=len(words))
# tf.nn.seq2seq.sequence_loss_by_example ??
loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
# loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))
cost = tf.reduce_mean(loss)
#     learning_rate = tf.Variable(0.0, trainable=False)
#     tvars = tf.trainable_variables()
#     grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     train_op = optimizer.apply_gradients(zip(grads, tvars))
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

trainds = DataSet(len(poetrys_vector))
epoch = 1000 # 训练1千次
iter_size = len(poetrys) // batch_size # 每次epoch需要多少次batch_size
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # sess.run(tf.assign(learning_rate, 0.01))
    for e in range(epoch):
        for batche in range(iter_size):
            x, y = trainds.next_batch(batch_size)
            # print('x.shape:', x.shape,' y.shap:', y.shape)
            cost_look, _ = sess.run([cost, train_op], feed_dict={input_data: x, output_targets: y}) # 不要使用相同的变量名
            # print(cost_look)
        if e % 100 == 0:
            print('{}次数, cost:{}'.format(e, cost_look))
