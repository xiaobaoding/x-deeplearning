#coding=utf-8
#constants
import datetime
EMBEDDING_SIZE = 1658384 #2019-02-11号样本的item_embedding_size
EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
max_sequence_length = 30
reduce_sequence_length = 20
max_order_sequence_length = 15
best_auc = 0.0
batch_size = 128
is_train = True
def logger(msg):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print (time + ":" + str(msg))
