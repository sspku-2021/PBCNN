
import sys
sys.path.append("../")

import tensorflow as tf

import utils
from model.siamese_network import Siamese



def test(model: Siamese, K,
         test_file_path: str,
         model_name: str, restore_ckpt: str,
         file_out: str):

    print("starting testing %s model" % model_name)

    gpu_config = utils.gpu_config()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        model.build_model()
        saver = tf.train.Saver(tf.global_variables())