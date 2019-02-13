#coding=utf-8
# Copyright (C) 2016-2018 Alibaba Group Holding Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
import time
import math
import random
import argparse
import tensorflow as tf
import numpy

from model import *
from utils import *
import xdl
from xdl.python.training.train_session import QpsMetricsHook, MetricsPrinterHook
from utils import *



#config here
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="random seed", default=3)
parser.add_argument("-jt", "--job_type", help="'train' or 'test'", default='train')
parser.add_argument("-m", "--model", help="'din' or 'dien'", default='din')
parser.add_argument("-si", "--save_interval", help="checkpoint save interval steps", default=20000)
parser.add_argument("-dr", "--data_dir", help="data dir")
args, unknown = parser.parse_known_args()

seed = args.seed
job_type = args.job_type
model_type = args.model
save_interval = args.save_interval


def get_data_prefix():
    #/user/dixin/deep/wide_deep_base/2019-02-11/data/tfrecord/train/xxx
    return args.data_dir

train_file = os.path.join(get_data_prefix(), "local_train_splitByUser")

def train(train_file=train_file,
          batch_size=128,
          maxlen=100,
          test_iter=700):
    if xdl.get_config('model') == 'din_mogujie':
        model = Model_DIN_MOGUJIE(
            EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
    else:
        raise Exception('only support din_mogujie and dien')

    #data set

    with xdl.model_scope('train'):

        train_ops = model.build_network(EMBEDDING_DIM)
        lr = 0.001
        # Adam Adagrad
        train_ops.append(xdl.Adam(lr).optimize())
        hooks = []
        log_format = "[%(time)s] lstep[%(lstep)s] gstep[%(gstep)s] lqps[%(lqps)s] gqps[%(gqps)s] loss[%(loss)s]"
        hooks = [QpsMetricsHook(), MetricsPrinterHook(log_format)]
        if xdl.get_task_index() == 0:
            hooks.append(xdl.CheckpointHook(xdl.get_config('checkpoint', 'save_interval')))
        train_sess = xdl.TrainSession(hooks=hooks)

    with xdl.model_scope('test'):
        test_ops = model.build_network(
            EMBEDDING_DIM, is_train=False)
        test_sess = xdl.TrainSession()
    model.run(train_ops, train_sess, test_ops, test_sess, test_iter=test_iter)

def test():
    pass

if __name__ == '__main__':
    SEED = xdl.get_config("seed")
    if SEED is None:
        SEED = 3
    tf.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

    if job_type == 'train':
        train()
    elif job_type == 'test':
        test()
    else:
        print('job type must be train or test, do nothing...')
