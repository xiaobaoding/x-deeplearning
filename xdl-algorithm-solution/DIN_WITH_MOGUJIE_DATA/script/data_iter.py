#coding=utf-8
import tensorflow as tf
import xdl
from utils import *
import numpy as np
import tensorflow as tf
class DataIterator:
    def __init__(self, train_files, batch_size):
        self.train_files = train_files
        self.batch_size = batch_size

        training_data = self.tf_batch_inputs()
        self.iterator = training_data.make_one_shot_iterator()
        logger("init the data input at DataIterator")

    def tf_batch_inputs(self):
        with tf.name_scope('input'):
            files = tf.data.Dataset.list_files(self.train_files)
            dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=2))
            #dataset = tf.data.Dataset.from_tensor_slices(self.train_files).interleave(
            #    lambda x: tf.data.TFRecordDataset(x).prefetch(10), cycle_length=2)
            dataset = dataset.shuffle(buffer_size=self.batch_size * 10)
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
            dataset = dataset.map(lambda x: self._decode(x), num_parallel_calls=2)
            dataset = dataset.prefetch(buffer_size=1000)
            return dataset

    def _decode(self, example):
        features_config = {}
        base_feature = [
            'tradeitemid_qid',
            'tradeitemid', 'uuid_did', 'item_click_rate_search',
            'item_gmv_rate_search',
            'item_order_rate_search',
            'item_click_pv_search_smoothed',
            'item_gmv_count_search_smoothed',
            'item_gmv_sum_search_smoothed'
        ]
        features_config['user_click_seq_3day_all'] = tf.FixedLenFeature([max_sequence_length], tf.int64)
        features_config['item_id'] = tf.FixedLenFeature([1], tf.int64)
        features_config['tradeitemid_user_click_seq_3day_all'] = tf.FixedLenFeature([max_sequence_length], tf.int64)
        features_config['tradeitemid_user_order_seq_3day_all'] = tf.FixedLenFeature([max_order_sequence_length],
                                                                                    tf.int64)
        for fea_name in base_feature:
            features_config[fea_name] = tf.FixedLenFeature([], tf.int64)

        features_config['label'] = tf.FixedLenFeature([], tf.int64)
        features = tf.parse_example(example, features=features_config)

        label = tf.cast(features['label'], tf.int32)
        click_seq = tf.cast(features['user_click_seq_3day_all'], tf.int32)
        itemid = tf.cast(features['item_id'], tf.int32)
        item_click_seq = tf.cast(features['tradeitemid_user_click_seq_3day_all'], tf.int32)
        item_order_seq = tf.cast(features['tradeitemid_user_order_seq_3day_all'], tf.int32)
        item_query = tf.cast(features['tradeitemid_qid'], tf.int32)
        wide_item = tf.cast(features['tradeitemid'], tf.int32)
        uuid_did = tf.cast(features['uuid_did'], tf.int32)
        # cid = tf.cast(features['cid'], tf.int32)
        # shopid = tf.cast(features['shopid'], tf.int32)
        click_rate = tf.cast(features['item_click_rate_search'], tf.int32)
        gmv_rate = tf.cast(features['item_gmv_rate_search'], tf.int32)
        order_rate = tf.cast(features['item_order_rate_search'], tf.int32)
        click_pv = tf.cast(features['item_click_pv_search_smoothed'], tf.int32)
        gmv_count = tf.cast(features['item_gmv_count_search_smoothed'], tf.int32)
        gmv_sum = tf.cast(features['item_gmv_sum_search_smoothed'], tf.int32)
        #return click_seq, itemid, item_click_seq, item_order_seq, item_query, wide_item, uuid_did, click_rate, gmv_rate, order_rate, click_pv, gmv_count, gmv_sum, label

        #cast to np array /it's black magic...
        sess = tf.Session()
        click_seq_np = np.array(click_seq.eval(session=sess),dtype=np.int32)
        itemid_np = np.array(itemid.eval(session=sess),dtype=np.int32)
        label_np = np.array(label.eval(session=sess),dtype=np.float32)

        return click_seq_np, itemid_np, label_np

    def read_and_parse_data(self):
        """
        :return:
        """
        try:
            click_seq, itemid, label = self.iterator.get_next()
        except tf.errors.OutOfRangeError:
            raise Exception("train file ends !")
        #parse data
        results = []
        # 构建xdl SparseTensor
        #click_seq
        results.append(np.reshape(click_seq, -1)) #click seq
        results.append(np.ones([batch_size * max_sequence_length], np.float32)) #default values
        results.append(np.array([i + 1 for i in range(batch_size * max_sequence_length)], dtype=np.int32))#segments

        #itemid
        results.append(np.reshape(itemid, -1))  #  why reshape？
        results.append(np.ones([batch_size], np.float32))  # default values
        results.append(np.array([i + 1 for i in range(batch_size )], dtype=np.int32))  # segments
        #label
        results.append(np.array(label, dtype=np.float32))
        return results

    def next_batch(self):
        """
        data[0] -> click_seq
        data[1] -> itemid
        data[2] -> label
        :return:
        """
        types = []
        for _ in range(max_sequence_length): #for click_seqs
            types.append(np.int64)
        types.append(np.int64)  #for itemid
        types.append(np.int64)  # for label
        datas =xdl.py_func(self.read_and_parse_data, [], output_type=types)
        sparse_cnt = 2  #only click_seq and itemid is sparse
        sparse_tensors = []
        for i in range(sparse_cnt):
            sparse_tensors.append(xdl.SparseTensor(
                datas[3 * i], datas[3 * i + 1],
                datas[3 * i + 2]))  # a batch of sparse examples .  ids/values/segments_index
        return sparse_tensors + datas[sparse_cnt * 3:]  #
