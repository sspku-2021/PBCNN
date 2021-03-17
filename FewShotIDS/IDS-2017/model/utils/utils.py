# -*- coding: utf-8 -*-

import tensorflow as tf

def convolution_1d(x, filter_size, num_filters, name_scope, stddev = 0.1):
    """
    :param x: [batch_size * channels, length, width] or [batch_size, channels, vec_dim]
    :param vector_dim:
    :param filter_size:
    :param num_filters:
    :param name_scope:
    :return:
           conv: [batch_size * channels, length - filter_size + 1, num_filters] or
                 [batch_size, channels - filter_size + 1, num_filters]
    """
    with tf.variable_scope(name_scope + str(filter_size)):
        conv = tf.layers.conv1d(
            inputs = x,
            filters = num_filters,
            kernel_size = filter_size,
            strides = 1,
            activation = tf.nn.relu,
            padding = "VALID",
            kernel_initializer = tf.truncated_normal_initializer(stddev=stddev),
            bias_initializer = tf.constant_initializer(),
            reuse=tf.AUTO_REUSE
        )
    return conv

def max_pool_1d(x, filter_size, name_scope):
    """
    :param x: [batch_size * channels, length - filter_size + 1, num_filters] or
              [batch_size, channels - filter_size + 1, num_filters]
    :param filter_size:
    :param name_scope:
    :return:
           pooled: [batch_size * channels, num_filter] or [batch_size, num_filters]
    """

    pool_width = int(x.shape[1])
    with tf.name_scope(name_scope + str(filter_size)):
        pooled = tf.layers.max_pooling1d(inputs = x,
                                         pool_size = pool_width,
                                         strides = 1,
                                         padding = "VALID")
        pooled = tf.squeeze(pooled, axis=-2)
        return pooled

def dropout(cell, keep_prob):
    cell_dropout = tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=keep_prob,
        variational_recurrent=True, dtype=tf.float32)
    return cell_dropout


def create_lstm_cell(hidden_size, keep_prob, num_layers, reuse=False):
    def single_rnn_cell():
        cell = tf.contrib.rnn.LSTMCell(
            hidden_size, use_peepholes=True,  # peephole: allow implementation of LSTMP
            initializer=tf.contrib.layers.xavier_initializer(),
            forget_bias=1.0, reuse=reuse)
        cell = dropout(cell, keep_prob)
        return cell

    cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(num_layers)])
    return cell

def dynamic_origin_bilstm_layer(lstm_fw_cell, lstm_bw_cell, inputs, seq_length):
    '''
    output is the last layer hidden state for every time step...
    state is the last cell and hidden state for every layers...
    '''
    bilstm_output, bilstm_state = tf.nn.bidirectional_dynamic_rnn(
        lstm_fw_cell, lstm_bw_cell,
        inputs=inputs,
        sequence_length=seq_length,
        dtype=tf.float32,
        swap_memory=True)
    output = tf.concat(bilstm_output, axis=2)
    # 0 left->right
    # 1 right->left
    h_state = tf.concat([bilstm_state[0][-1].h, bilstm_state[1][-1].h], axis=1)
    # c_state = tf.concat([bilstm_state[0][-1].c, bilstm_state[1][-1].c], axis=1)
    # state = tf.nn.rnn_cell.LSTMStateTuple(c=c_state,
    #                                       h=h_state)
    return output, h_state

def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def position_embeded(max_seq_length, embedding_dim, name):
    position_embedding = tf.get_variable(
        name = name,
        shape = [max_seq_length, embedding_dim],
        initializer = tf.truncated_normal_initializer(stddev=0.02)
    )
    return position_embedding


def compute_grads(loss, optimizer, var_list=None):
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    valid_grads = [
        (grad, var)
        for (grad, var) in grads
        if grad is not None]
    if len(valid_grads) != len(var_list):
        print("Warning: some grads are None.")
    return valid_grads


def average_gradients(tower_grads):
    """
    Copied from: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples.
            The outer list is over individual gradients.
            The inner list is over the gradient calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been
        averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, axis=0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]  # [0]: first tower [1]: ref to var
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def apply_grads(optimizer, tower_grads, clipping_threshold=5.0, global_step=None):
    """
    Compute average gradients, perform gradient clipping and apply gradients
    Args:
        tower_grad: gradients collected from all GPUs
    Returns:
        the op of apply_gradients
    """
    # averaging over all gradients
    avg_grads = average_gradients(tower_grads)

    # Perform gradient clipping
    (gradients, variables) = zip(*avg_grads)
    (clipped_gradients, _) = tf.clip_by_global_norm(gradients, clipping_threshold)

    # Apply the gradients to adjust the shared variables.
    apply_gradients_op = optimizer.apply_gradients(
        zip(clipped_gradients, variables), global_step=global_step)

    return apply_gradients_op

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    '''
    type(model_params):dict
    '''
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)