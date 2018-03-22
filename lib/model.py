""" model implementation """

import logging

import tensorflow as tf

__all__ = ['RNN']

def read_one_sequence(file_queue):
    """Read one sequence from .tfrecords"""
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(file_queue)

    feature = tf.parse_single_example(serialized_example, features={
        'rnn_input': tf.VarLenFeature(tf.int64),
        'rnn_input_len': tf.FixedLenFeature([1], tf.int64),
        'target_output': tf.FixedLenFeature([1], tf.int64),
    })

    return feature['rnn_input'], feature['rnn_input_len'], feature['target_output']

def get_sampled_ids(outputs):
    """Get sampled ids from outputs"""
    # outputs: [batch_size, vocab_size]
    ids = tf.multinomial(outputs, num_samples=1)
    return ids

class RNN(): # pylint: disable=too-many-instance-attributes
    """Implementation of a RNN-LM model"""

    def __init__(self, para):
        self.para = para
        self.dtype = tf.float32
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if self.para.mode == 'pretrain':
            self.set_input()
            self.build_graph()
            self.build_optimizer()

        elif self.para.mode == 'rl':
            self.set_input()
            self.build_graph()
            self.build_rl_optimizer()

        elif self.para.mode == 'test':
            self.set_input()
            self.build_graph()

        self.saver = tf.train.Saver(max_to_keep=1)

    def set_input(self):
        """Construct input nodes for the graph
            raw_rnn_inputs: [batch_size, max_len]
            raw_rnn_inputs_len: [batch_size]
            raw_target_outputs: [batch_size]
            rnn_inputs_embedded: [batch_size, max_len, embedding_size]
            rnn_inputs_len: [batch_size]
        """
        logging.debug('set input...')
        self.rnn_inputs_embedded = tf.placeholder(
            dtype=self.dtype,
            shape=(None, self.para.max_len, self.para.embedding_size)
        )
        self.rnn_inputs_len = tf.placeholder(
            dtype=tf.int32,
            shape=(None,)
        )
        if self.para.mode == 'pretrain':
            self.raw_rnn_inputs, self.raw_rnn_inputs_len, self.raw_target_outputs = \
                self.read_batch_sequences()
            self.target_outputs = tf.placeholder(
                dtype=tf.int32,
                shape=(None,)
            )
        elif self.para.mode == 'rl':
            self.raw_rnn_inputs, self.raw_rnn_inputs_len, self.raw_target_outputs = \
                self.read_batch_sequences()
            self.sampled_ids_inputs = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='sampled_ids_inputs'
            )
            self.rewards = tf.placeholder(
                dtype=self.dtype, shape=(None,), name='rewards'
            )

    def build_graph(self):
        """Build graph"""
        logging.debug('build graph...')

        self.rnn_cell = self.build_rnn_cell()
        self.all_rnn_outputs, self.next_state = tf.nn.dynamic_rnn(
            cell=self.rnn_cell,
            inputs=self.rnn_inputs_embedded,
            sequence_length=self.rnn_inputs_len,
            dtype=self.dtype,
        )
        if self.para.concatenate == 1:
            self.final_rnn_outputs = tf.reshape(
                self.all_rnn_outputs,
                [self.para.batch_size, self.para.max_len * self.para.num_units]
            )
        else:
            self.final_rnn_outputs = tf.concat(
                [self.next_state[i][1] for i in range(self.para.num_layers)], 1
            )

        # rnn_outputs: [batch_size, vocab_size]
        self.rnn_outputs = tf.layers.dense(self.final_rnn_outputs, self.para.vocab_size)
        self.probs = tf.nn.softmax(self.rnn_outputs)

        self.sampled_ids = get_sampled_ids(self.rnn_outputs)
        self.predicted_ids = tf.argmax(self.rnn_outputs, axis=1)

        if self.para.mode == 'pretrain':
            self.loss = self.compute_loss(
                labels=self.target_outputs,
                logits=self.rnn_outputs
            )
        elif self.para.mode == 'rl':
            self.loss = self.compute_rl_loss(
                labels=self.sampled_ids_inputs,
                logits=self.rnn_outputs
            )

    def build_optimizer(self):
        """Build optimizer"""
        logging.debug('build optimizer...')
        trainable_variables = tf.trainable_variables()

        if self.para.optimizer == 'sgd':
            self.learning_rate = tf.cond(
                self.global_step < self.para.start_decay_step,
                lambda: tf.constant(self.para.learning_rate),
                lambda: tf.train.exponential_decay(
                    self.para.learning_rate,
                    (self.global_step - self.para.start_decay_step),
                    self.para.decay_steps,
                    self.para.decay_factor,
                    staircase=True),
                name="learning_rate"
            )
            self.opt = tf.train.GradientDescentOptimizer(self.para.learning_rate)
        elif self.para.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer()
        elif self.para.optimizer == 'rms':
            self.opt = tf.train.RMSPropOptimizer(self.para.learning_rate)

        gradients = tf.gradients(self.loss, trainable_variables)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.para.max_gradient_norm)
        self.update = self.opt.apply_gradients(
            zip(clip_gradients, trainable_variables),
            global_step=self.global_step
        )

    def build_rl_optimizer(self):
        """Build optimizer for rl mode"""
        trainable_variables = tf.trainable_variables()
        self.rl_opt = tf.train.AdamOptimizer(self.para.rl_learning_rate)
        gradients = tf.gradients(self.loss, trainable_variables)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.para.max_gradient_norm)
        self.rl_update = self.rl_opt.apply_gradients(
            zip(clip_gradients, trainable_variables),
            global_step=self.global_step
        )

    def compute_loss(self, labels, logits):
        """
            rnn_outputs: [batch_size, vocab_size]
            labels: [batch_size]
        """

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits
        )
        loss = tf.reduce_sum(cross_entropy) / tf.to_float(self.para.batch_size)
        return loss

    def compute_rl_loss(self, labels, logits):
        """
            logits: [batch_size, vocab_size]
            labels: [batch_size]
            self.rewards: [batch_size]
        """
        # log_p: [batch_size, vocab_size]
        log_p = -tf.log(
            tf.add(tf.nn.softmax(logits), tf.constant(1e-8, dtype=self.dtype))
        )
        # labels: [batch_size, vocab_size]
        labels = tf.one_hot(
            indices=labels,
            depth=self.para.vocab_size
        )
        # loss: [batch_size]
        loss = tf.reduce_sum(tf.multiply(log_p, labels), [1])
        loss = tf.reduce_sum(tf.multiply(loss, self.rewards)) / \
               tf.to_float(self.para.batch_size)
        return loss

    def build_single_cell(self):
        """Build a single RNN cell"""
        cell = tf.contrib.rnn.LSTMCell(self.para.num_units)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell,
            input_keep_prob=(1.0 - self.para.dropout)
        )
        if self.para.attention == 1:
            cell = tf.contrib.rnn.AttentionCellWrapper(cell, self.para.attention_window)
        return cell

    def build_rnn_cell(self):
        """Build multiple RNN layers"""
        return tf.contrib.rnn.MultiRNNCell([self.build_single_cell()] * self.para.num_layers)

    def read_batch_sequences(self):
        """ read a batch from .tfrecords """
        logging.debug('read data from traing.tfrecords...')

        file_queue = tf.train.string_input_producer(['./data/train.tfrecords'])

        r_i, r_i_l, r_o = read_one_sequence(file_queue)

        min_after_dequeue = 3000
        capacity = min_after_dequeue + 3 * self.para.batch_size

        rnn_inputs, rnn_inputs_len, target_outputs = tf.train.shuffle_batch(
            [r_i, r_i_l, r_o],
            batch_size=self.para.batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue
        )

        rnn_inputs = tf.sparse_tensor_to_dense(rnn_inputs)
        rnn_inputs_len = tf.reshape(rnn_inputs_len, [self.para.batch_size])
        target_outputs = tf.reshape(target_outputs, [self.para.batch_size])

        return rnn_inputs, rnn_inputs_len, target_outputs
