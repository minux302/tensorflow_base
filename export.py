import os
import click
import tensorflow as tf
from tqdm import tqdm

import config
from model import Model
from loader import Loader


def export(id, save_dir):

  with tf.Graph().as_default():

    # build_model
    model = Model(class_num=config.CLASS_NUM,
                  input_height=config.INPUT_HEIGHT,
                  input_width=config.INPUT_WIDTH)

    input_pl, _ = model.placeholders()
    is_training_pl = tf.placeholder(tf.bool, name="is_training")
    _ = model.build(input_pl, is_training_pl)

    # saver
    saver = tf.train.Saver()

    # training
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
      # init
      init_op = tf.global_variables_initializer()
      sess.run([init_op])
      ckpt = tf.train.get_checkpoint_state(save_dir)
      if ckpt and ckpt.model_checkpoint_path:
        print("--------- Restore last checkpoint -------------")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("recovered.")


      minimal_graph = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(add_shapes=True),
        ["output"],
      )
      ckpt_name = "hoge"
      pb_name = ckpt_name + ".pb"
      pbtxt_name = ckpt_name + ".pbtxt"
      tf.train.write_graph(minimal_graph, save_dir, pb_name, as_text=False)
      tf.train.write_graph(minimal_graph, save_dir, pbtxt_name, as_text=True)


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
  "-sd",
  "--save_dir",
  help="path to ckpt dir",
  default="save/0",
  required=True
)

def main(id, save_dir):
    export(id, save_dir)


if __name__ == '__main__':
  main()

