import tensorflow as tf

class Model:

  def __init__(self,
               class_num,
               input_height,
               input_width):
    self.class_num = class_num
    self.input_height = input_height
    self.input_width = input_width

  def placeholders(self):
    with tf.name_scope('input'):
      input_pl = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, 3], name="input")
      label_pl = tf.placeholder(tf.int32, [1], name="label")

    return input_pl, label_pl

  def loss(self, pred, labels):
    with tf.name_scope('loss'):
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=pred)
      tf.summary.scalar('loss', loss)
    return loss

  def optimizer(self, loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    return optimizer.minimize(loss)

  def accuracy(self, pred, label):
    with tf.name_scope('metric'):
      acc, acc_op = tf.metrics.accuracy(label, tf.argmax(pred, 1))

    tf.summary.scalar('accuracy', acc)
    return acc, acc_op

  def build(self, inputs, is_training):

    x = self._conv2d(inputs, 128, "conv1")
    x = self._conv2d(x, 128, "conv2")

    x = self._flatten(x)
    x = tf.layers.dense(x, 10)

    new_batch_size = 1
    sequence_len = 3 # == batch_size for cnn input
    hidden_num = 20
    x = tf.reshape(x, [1, 3, 10])

    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_num)
    initial_state = cell.zero_state(new_batch_size, dtype=tf.float32)
    xs, _ = tf.nn.dynamic_rnn(cell=cell,
                              inputs=x,
                              initial_state=initial_state,
                              dtype=tf.float32,
                              time_major=False)
    xs = tf.transpose(xs, [1,0,2])
    # print(output.get_shape())
    # x = tf.gather(xs, int(xs.get_shape()[0] -1))
    x = tf.slice(xs, [int(xs.get_shape()[0] - 1), 0, 0], [1, new_batch_size, hidden_num])

    x = tf.layers.dense(x,
                        1,
                        use_bias=False,
                        activation=None)
    x = tf.identity(x, name="output")

    return x

  def _conv2d(self, x, out_ch, name, activate=True):
    x = tf.layers.conv2d(x,
                         out_ch,
                         kernel_size=3,
                         strides=(1, 1),
                         padding="same",
                         name=name,
                         kernel_initializer=tf.glorot_normal_initializer())
    if activate:
      x = tf.nn.relu(x)

    return x

  def _flatten(self, x):
    shape = x.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
      dim *= d

    return tf.reshape(x, [-1, dim])
