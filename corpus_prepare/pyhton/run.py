#encoding=utf-8
import sys
from fun import *
reload(sys)
sys.setdefaultencoding("utf-8")





if __name__ == '__main__':
    # 合并 用户和商户的聊天记录到一个文件
    # user_chat, merchant_chat = get_chat_data('../chat_log_user.data'), get_chat_data('../chat_log_merchant.data')
    # all_chat_log = merge_chat(user_chat, merchant_chat)
    # with open('chat_log_merge.data','w') as f:
    #     for line in all_chat_log:
    #         f.writelines(line+'\n')

    # 切词 途家有效评论数据

    #corpus_prepare_save_split('../fromComment.data')
    corpus_prepare_save_split('merge_chat_log.data')
    #user_chat, merchant_chat = get_chat_data('../chat_log_user.data'), get_chat_data('../chat_log_merchant.data')
    #merge_chat_split(user_chat, merchant_chat)

