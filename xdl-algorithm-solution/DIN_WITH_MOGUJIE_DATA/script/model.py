#coding=utf-8
#
# model structure same as:
# http://gitlab.mogujie.org/yangming/rank_tf/blob/distribute_dixin_tfrank_adam_v1/models/wd/wide_deep_attention_loss.py
import xdl
from xdl.python.utils.metrics import add_metrics
import tensorflow as tf
from data_iter import DataIterator
from operator import mul
from utils import *
class Model_DIN_MOGUJIE(object):
    def __init__(self, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        """

        :param EMBEDDING_DIM:
        :param HIDDEN_SIZE:
        :param ATTENTION_SIZE:
        :param use_negsampling:
        """
        self.embedding_dim = EMBEDDING_DIM
        self.hidden_size = HIDDEN_SIZE
        self.attention_size = ATTENTION_SIZE
        self.use_negsampling = use_negsampling
        self.data_iterator = DataIterator()

    def build_inputs(self):
        """
        iterator the training data and parse to xdl format
        :return:
        """

        datas = self.data_iterator.next_batch()

        return datas

    def build_network(self):
        """
        build model structure
        :return:
        """
        @xdl.tf_wrapper(is_training=True)
        def tf_train_model(*inputs):
            with tf.variable_scope("deep_layer", reuse=tf.AUTO_REUSE):
                #reshape sparse tensor back??

                sequence_emb = inputs[0]
                sequence_emb = tf.reshape(inputs[0], [-1, max_sequence_length, EMBEDDING_DIM])
                sequence_emb.set_shape([None, None, EMBEDDING_DIM])
                itemid_emb = inputs[1]
                itemid_emb = tf.reshape(inputs[1], [-1, EMBEDDING_DIM])
                itemid_emb.set_shape([None, EMBEDDING_DIM])

                itemid_token_mask = tf.cast(itemid_emb, tf.bool)
                sequence_token_mask = tf.cast(tf.slice(sequence_emb, [0, 0], [batch_size, reduce_sequence_length]), tf.bool)
                self.logits_deep = self.bulid_attention_layers(sequence_emb, sequence_token_mask, itemid_emb,
                                                                   itemid_token_mask, 'attention')
                self.logits = self.logits_deep

            train_ops = self.train_ops()  #

            return train_ops[0], train_ops[1:]
        #get data batch
        datas = self.build_inputs()
        #
        train_ops = tf_train_model(*self.xdl_embedding(datas))
        return train_ops

    def xdl_embedding(self, datas): #data[0]=;data[1]=item_seq
        """
        稀疏部分的定义
        :param datas:
        :return:
        """
        results = []
        seq_emb = xdl.embedding("item_embedding", datas[0],
                                xdl.VarianceScaling(scale=1.0, mode="fan_avg", distribution="normal", seed=3),
                                EMBEDDING_DIM, EMBEDDING_SIZE, 'sum')
        item_emb = xdl.embedding("item_embedding", datas[1], xdl.VarianceScaling(scale=1.0, mode="fan_avg", distribution="normal", seed=3),
                                    EMBEDDING_DIM, EMBEDDING_SIZE, 'sum')

        results.append(seq_emb)
        results.append(item_emb)
        #return results + datas[7:]
        return results+datas[2:]

    def bulid_attention_layers(self, u_emb, all_token_mask, itemid_emb, temid_token_mask, scope):
        """
        :param u_emb:
        :param all_token_mask:
        :param itemid_emb:
        :param temid_token_mask:
        :param scope:
        :return:
        """
        all_u_emb = tf.reduce_sum(u_emb, 1)
        itemid_emb = tf.squeeze(itemid_emb, 1)
        logger('item squeeze shape:{0}'.format(itemid_emb.get_shape()))
        with tf.name_scope('linear_layer'):
            outputs = bn_dense_layer(u_emb, 128, True, 0.01, 'multi_logits', 'elu',
                                     False, 5e-5, 0.75, is_train)
            """
            outputs= directional_attention_with_dense(
                self.item_his_eb, all_token_mask,None, 'dir_attn',
                0.75, self.is_train,5e-5, 'elu' , None,name='s1_attention')

            tf.summary.histogram('GRU_outputs', outputs)
            """
            self.aux_loss = 0.

            """
            #do not use aux_loss 
            self.aux_loss = auxiliary_loss(outputs[:, :-1, :], u_emb[:, 1:, :],
                                           self.noclk_item_his_eb[:, 1:, :],
                                           all_token_mask[:, 1:], stag="gru")
            """
        # Attention layer
        with tf.name_scope('cross_attention_layer'):
            att_outputs, alphas = din_fcn_attention(itemid_emb, outputs, 128, all_token_mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('att_outputs', att_outputs)

        with tf.name_scope('single_attention_layer'):
            final_state = single_attention(
                att_outputs, all_token_mask, 'disan_sequence', 0.75, is_train, 5e-5,
                'elu', None, 's1_attention', is_multi_att=False, attention_dim=None
            )
        inp = tf.concat(
            [itemid_emb, all_u_emb, itemid_emb * all_u_emb, all_u_emb - itemid_emb, final_state], 1)

        with tf.variable_scope('output_layer_last'):
            logits_deep = self.build_fcn_net(inp, use_dice=True)

        return logits_deep

    def build_fcn_net(self, inp, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')
        dnn1 = tf.nn.dropout(dnn1, 0.75)
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn2 = tf.nn.dropout(dnn2, 0.9)
        dnn3 = tf.layers.dense(dnn2, 1, activation=None, name='f3')[:, 0]
        return dnn3

    def run(self, train_ops, train_sess, test_ops=None, test_sess=None, test_iter=100):
        iter = 0
        for epoch in range(1):
            while not train_sess.should_stop():
                values = train_sess.run(train_ops)
                if values is None:
                    break
                loss, acc, aux_loss, _ = values
                add_metrics("loss", loss)
                add_metrics("time", datetime.datetime.now(
                ).strftime('%Y-%m-%d %H:%M:%S'))

                iter += 1
                if (iter % test_iter) == 0:
                    self.run_test(test_ops, test_sess)
                if (iter % test_iter) == 0:  # add timeline
                    run_option = xdl.RunOption()
                    run_option.perf = True
                    run_statistic = xdl.RunStatistic()
                    _ = train_sess.run(train_ops, run_option, run_statistic)
                    xdl.Timeline(run_statistic.perf_result).save('../ckpt/timeline.json-' + str(iter))
                    print ('======print the timeline =====')
                    iter += 1
            train_sess._finish = False

# copied func
def bn_dense_layer(input_tensor, hn, bias, bias_start=0.0, scope=None,
                   activation='relu', enable_bn=True,
                   wd=0., keep_prob=1.0, is_train=None):
    if is_train is None:
        is_train = False

    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map',
                            False, wd, keep_prob, is_train)
        if enable_bn:
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True, is_training=is_train, scope='bn')
        return activation_func(linear_map)

def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args]  # for dense layer [(-1, d)]
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                     # for dense layer [(-1, d)]
                     for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope)  # dense
    out = reconstruct(flat_out, args[0], 1)  # ()
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list()) - 1])

    if wd:
        add_reg_without_bias()

    return out

def _linear(xs, output_size, bias, bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs, -1)
        input_size = x.get_shape()[-1]
        W = tf.get_variable('W', shape=[input_size, output_size], dtype=tf.float32, )
        if bias:
            bias = tf.get_variable('bias', shape=[output_size], dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        return out

def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat

def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list()  # original shape
    tensor_shape = tensor.get_shape().as_list()  # current shape
    ref_stop = len(ref_shape) - keep  # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]  #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]  #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

def auxiliary_loss(h_states, click_seq, noclick_seq, mask, stag=None):
    mask = tf.cast(mask, tf.float32)
    click_input_ = tf.concat([h_states, click_seq], -1)
    noclick_input_ = tf.concat([h_states, noclick_seq], -1)
    click_prop_ = auxiliary_net(click_input_, stag=stag)[:, :, 0]
    noclick_prop_ = auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
    click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
    noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
    loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
    return loss_

def auxiliary_net(in_, stag='auxiliary_net'):
    bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
    dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
    dnn1 = tf.nn.sigmoid(dnn1)
    dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
    dnn2 = tf.nn.sigmoid(dnn2)
    dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
    y_hat = tf.nn.softmax(dnn3) + 0.00000001
    return y_hat

def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                      return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        raw_scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(raw_scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(raw_scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output

def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def single_attention(rep_tensor, rep_mask, scope=None,
                     keep_prob=1., is_train=None, wd=0., activation='elu',
                     tensor_dict=None, name='', is_multi_att=False, attention_dim=None):
    """single attention: contain two traditional attention struct.

    Args:
      rep_tensor: list tensor,shape is [batch_size,list_size,embedding_dim].
      rep_mask: whether the marker Tensor is 0,bool value, shape is [batch_size,list_size]
    Returns:
      Attention representation of tensor,shape is [batch_size,embedding_dim]
    Raises:
      TypeError: If the input dimension is incorrect.
    """

    with tf.variable_scope(scope):
        attention_rep_first_layer = traditional_attention(
            rep_tensor, rep_mask, 'traditional_attention', 'linear',
            keep_prob, is_train, wd, activation,
            tensor_dict=tensor_dict, name=name + '_attn')

        # attention_rep_final = traditional_attention(
        #    attention_rep_first_layer, rep_mask, 'traditional_attention',
        #    keep_prob, is_train, wd, activation,
        #    tensor_dict=tensor_dict, name=name + '_attn')

        return attention_rep_first_layer

def traditional_attention(rep_tensor, rep_mask, scope=None, func='linear',
                          keep_prob=1., is_train=None, wd=0., activation='elu',
                          tensor_dict=None, name=None, output_dim=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    if output_dim is None:
        ivec = rep_tensor.get_shape()[2]
    else:
        ivec = output_dim
    with tf.variable_scope(scope):
        rep_tensor_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                        False, wd, keep_prob, is_train)
        rep_tensor_logits = get_logits([rep_tensor_map], None, False, scope='self_attn_logits', func=func,
                                       mask=rep_mask, input_keep_prob=keep_prob, is_train=is_train)  # bs,sl
        print ('rep_tensor_logits_shape:', rep_tensor_logits.get_shape())
        attn_res = softsel(rep_tensor, rep_tensor_logits, rep_mask)  # bs,vec

        # save attn
        # if tensor_dict is not None and name is not None:
        #    tensor_dict[name] = tf.nn.softmax(rep_tensor_logits)

        return attn_res
def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        out = tf.nn.softmax(logits, -1)
        return out

def softsel(target, logits, mask=None, scope=None):
    """
    :param target: [ ..., J, d] dtype=float #(b,sn,sl,ql,d)
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out
def get_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0,
               input_keep_prob=1.0, is_train=None, func=None):
    if func is None:
        func = "linear"
    if func == 'sum':
        return sum_logits(args, mask=mask, name=scope)
    elif func == 'linear':
        return linear_logits(args, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd,
                             input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'dot':
        assert len(args) == 2
        arg = args[0] * args[1]
        return sum_logits([arg], mask=mask, name=scope)
    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits([arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd,
                             input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'proj':
        assert len(args) == 2
        d = args[1].get_shape()[-1]
        proj = linear([args[0]], d, False, bias_start=bias_start, scope=scope, wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        return sum_logits([proj * args[1]], mask=mask)
    elif func == 'tri_linear':
        assert len(args) == 2
        new_arg = args[0] * args[1]
        return linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd,
                             input_keep_prob=input_keep_prob,
                             is_train=is_train)
    else:
        raise Exception()

def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "Linear_Logits"):
        logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def sum_logits(args, mask=None, name=None):
    with tf.name_scope(name or "sum_logits"):
        if args is None or (isinstance(args, (tuple, list)) and not args):
            raise ValueError("`args` must be specified")
        if not isinstance(args, (tuple, list)):
            args = [args]
        rank = len(args[0].get_shape())
        logits = sum(tf.reduce_sum(arg, rank - 1) for arg in args)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits

def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * (-1e30), name=name)

def dice(_x, axis=-1, epsilon=0.0000001, name=''):
    alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)

    input_shape = list(_x.get_shape())
    reduction_axes = list(range(len(input_shape)))

    del reduction_axes[axis]  # [0]

    broadcast_shape = [1] * len(input_shape)  # [1,1]
    broadcast_shape[axis] = input_shape[axis]  # [1 * hidden_unit_size]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)  # [1 * hidden_unit_size]
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)  # [1 * hidden_unit_size]
    # x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)  # a simple way to use BN to calculate x_p
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x