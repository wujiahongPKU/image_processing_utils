#coding=utf-8

import os
import sys
import numpy as np

import tensorflow as tf
from argparse import ArgumentParser



def build_parser():
    parser=ArgumentParser()

    # parser.add_argument('input')
    return parser



def generator_graph(checkpoint_dir,output_dir, graph_name):
    g=tf.Graph()
    with g.as_default():
        img_placeholder = tf.placeholder(tf.float32, shape=None,
                                         name='img_placeholder')
        sess=tf.Session()
        saver=tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess,checkpoint_dir)

        # output_path_name=os.path.join(output_graph,graph_name)
        print("the saver is:",saver)
        tf.train.write_graph(sess.graph_def,output_dir,graph_name)




def generate_no_value_graph(checkpoint_dir,output_dir,graph_name):
    ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)



if __name__=="__main__":
    checkpoint_dir='./checkpoint_test'
    output_dir="./checkpoint_test"
    graph_name="freeze_graph.pb"

    generator_graph(checkpoint_dir,output_dir,graph_name)