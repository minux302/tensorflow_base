import os
import click
import tensorflow as tf
from tqdm import tqdm

import config
from model import Model
from loader import Loader


def train(id, reset):

  # set log and ckpt dir
  log_dir = config.LOG_DIR
  save_dir = config.SAVE_DIR
  if not(os.path.exists(log_dir)):
    os.system('mkdir ' + log_dir)
  if not(os.path.exists(save_dir)):
    os.system('mkdir ' + save_dir)

  log_id_dir  = os.path.join(log_dir,  id)
  save_id_dir = os.path.join(save_dir, id)
  if reset:
    if os.path.exists(log_id_dir):
      os.system('rm -rf ' + log_id_dir)
    if not(os.path.exists(save_id_dir)):
      os.system('rm -rf ' + save_id_dir)

  loader = Loader(data_dir=config.DATA_DIR,
                  input_height=config.INPUT_HEIGHT,
                  input_width=config.INPUT_WIDTH,
                  batch_size=config.BATCH_SIZE)

  with tf.Graph().as_default():

    # build_model
    model = Model(class_num=config.CLASS_NUM,
                  input_height=config.INPUT_HEIGHT,
                  input_width=config.INPUT_WIDTH)

    input_pl, label_pl = model.placeholders()
    is_training_pl = tf.placeholder(tf.bool, name="is_training")
    pred = model.build(input_pl, is_training_pl)

    # loss
    loss = model.loss(pred, label_pl)

    # optimizer
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    opt = model.optimizer(loss)

    # accuracy
    acc, acc_op = model.accuracy(pred, label_pl)

    # saver
    saver = tf.train.Saver()

    # training
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
      # init
      init_op = tf.global_variables_initializer()
      reset_metric_op = tf.local_variables_initializer()
      sess.run([init_op, reset_metric_op])

      for epoch in range(config.MAX_EPOCH):
        loader.on_epoch_end()
        sess.run([reset_metric_op])
        print("epoch: {} / {}".format(epoch, config.MAX_EPOCH))

        for j in tqdm(range(len(loader))):
          image_batch, label_batch = loader[j]

          feed_dict = {
            input_pl: image_batch,
            label_pl: label_batch,
          }
          _, _, _loss = sess.run([opt, acc_op, loss], feed_dict)

      save_path = saver.save(sess, os.path.join(config.SAVE_DIR, id + '/' + id + '.ckpt'))
      saver.save(sess, save_path)

  print("finished!!")


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
  "-i",
  "--id",
  help="training id",
  default="0",
  required=False
)
@click.option(
  "-r",
  "--reset",
  help="remove directory for ckpt and tensorboard",
  default="True",
  required=False
)

def main(id, reset):
    train(id, reset)


if __name__ == '__main__':
  main()

