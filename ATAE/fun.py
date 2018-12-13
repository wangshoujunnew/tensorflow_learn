import jieba
import datetime
import os
import re
from gensim.models import word2vec
import numpy as np


def word2vec_train(file=''):
    stopWords, sentence, corpus_save_path = [], [], 'corpus_jieba_after.txt'
    mode_save_path = datetime.datetime.now().strftime('%Y%m%d') + 'word2vec.model'
    if os.path.exists(mode_save_path):
        return word2vec.Word2Vec.load(mode_save_path), corpus_save_path

    print('train word2vec begin ... ')
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line) < 5 or len(line) > 500:
                print('句子太长或者太短:' + line)
                continue
            sentence.append(' '.join([tmp for tmp in \
                                      list(jieba.cut(re.sub(',|!|\.|\?|，|、', '', line.strip()), cut_all=False)) \
                                      if tmp not in stopWords]))

    with open(corpus_save_path, 'w', encoding='utf-8') as f:  # 分词后存入一个文件中
        for line in sentence:
            f.writelines(line.strip() + '\n')

    del sentence

    sentence = word2vec.LineSentence(corpus_save_path)
    model = word2vec.Word2Vec(sentences=sentence, size=500, min_count=1)
    model.save(mode_save_path)
    return model, corpus_save_path


def get_data(model, corpus_data):
    word2vec_sentence, max_length = [], 0

    with open(corpus_data, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word2vec_sentence.append([model.wv[tmp] for tmp in line.strip().split(' ')])
            max_length = max(max_length, len(line.strip().split(' ')))

    # 语料矩阵
    corpus_matrix = np.zeros([len(word2vec_sentence), max_length, 500], dtype=np.float32)

    for row, line in enumerate(word2vec_sentence):
        corpus_matrix[row, :len(line), :] = np.array(line)
    return corpus_matrix, max_length


# 得到下一个批次的矩阵, batch_size 为length
def next_batch(mertric, length):
    indexs = np.random.randint(0, mertric.shape[0], length)
    return np.array([mertric[i, :] for i in indexs])

# x = np.arange(0,27).reshape((3,3,3))
# print(x)
# print(x[0,:2,:])
