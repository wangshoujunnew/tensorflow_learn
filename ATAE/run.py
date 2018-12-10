from fun import *

if __name__ == '__main__':
    model, corpus_data = word2vec_train(file='corpus.txt')
    # print(model.wv['酒店'])

    input_data, max_length = get_data(model, corpus_data)
    print(input_data[2, :, :1], max_length)
