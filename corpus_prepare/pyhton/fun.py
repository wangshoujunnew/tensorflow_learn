# encoding=utf-8
import re
import jieba
import copy
import sys

reload(sys)
sys.setdefaultencoding("utf-8")


def corpus_prepare_save_split(file):
    corpus_save_path, sentence = str(file) + '_jieba', []

    with open(corpus_save_path, 'w') as ff:  # 分词后存入一个文件中
        with open(file, 'r') as f:
            for line in f.readlines():
                replace_str = re.sub("[’!\"#$%&\'()*+,-./:;<=>?@，。?：★、…【】《》？“”‘’！[\\]^_`{|}~i ……。，；！？,.;?!！（）()～~\n\r|、'——]+".decode(
                    "utf8"), "".decode("utf8"),
                       re.sub("[！!,，。\.？\?、]".decode("utf8"), '\t', line.strip().decode('utf-8', 'ignore')).decode(
                           'utf-8', 'ignore'))
                replace_str = re.sub('\t+', '\t', replace_str)
                sentence.append(' '.join([tmp for tmp in \
                    list(jieba.cut(replace_str, cut_all=False,HMM=True))]))
                if len(sentence) % 10 == 0:
                    for l in sentence[-10:]:
                        ff.writelines(l + '\n')


def get_chat_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            mg, fromuserid, createtime, touserid = line.strip().split('#%')
            data.append({'mg': mg, 'fromuserid': fromuserid, 'createtime': createtime, 'touserid': touserid})
    return data


def merge_chat_split(user_chat, merchant_chat):
    user_chat_next = copy.deepcopy(user_chat)[1:]
    user_chat_next.append({'mg': '', 'fromuserid': '', 'createtime': '', 'touserid': ''})

    # 从user_chat开始出发,找merchat_chat
    log_line, one_record, now_fromuser, now_touser, chat_date = [], [], '', '', ''
    with open('merge_chat_log.txt', 'w') as f:

        for row, uc in enumerate(user_chat):
            one_record.append(uc['mg'])
            if now_fromuser == '' and now_touser == '' and chat_date == '':
                now_fromuser, now_touser, chat_date = uc['fromuserid'], uc['touserid'], uc['createtime'][:10]

            for mc in merchant_chat:
                if mc['fromuserid'] == uc['touserid'] and \
                        uc['createtime'] < mc['createtime'] and \
                        user_chat_next[row]['fromuserid'] == now_fromuser and \
                        user_chat_next[row]['touserid'] == now_touser and \
                        user_chat_next[row]['createtime'][:10] == chat_date and \
                        user_chat_next[row]['createtime'] > mc['createtime']:
                    one_record.append(mc['mg'])
                elif mc['fromuserid'] == uc['touserid'] and \
                        uc['createtime'] < mc['createtime'] and \
                        user_chat_next[row]['createtime'][:10] == chat_date and \
                        (user_chat_next[row]['fromuserid'] != now_fromuser or user_chat_next[row][
                            'touserid'] != now_touser):
                    one_record.append(mc['mg'])
                else:  # 进行fromuserid 的下一条说话记录
                    pass
            if user_chat_next[row]['fromuserid'] != now_fromuser or user_chat_next[row]['touserid'] != now_touser or \
                    user_chat_next[row]['createtime'][:10] != chat_date:
                now_fromuser = user_chat_next[row]['fromuserid']
                now_touser = user_chat_next[row]['touserid']
                chat_date = user_chat_next[row]['createtime'][:10]
                print('fromuserid 重新赋值 ... ')
                log_line.append('|'.join(one_record))
                one_record = []
            if len(log_line) % 10 == 0:
                for l in log_line[-10:]:
                    f.writelines(l + '\n')
