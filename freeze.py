import tensorflow as tf
import os, sys

checkpoint_dir_path = sys.argv[1]
meta_path = os.path.join(checkpoint_dir_path, 'fns.ckpt.meta')
output_node_names = ['add_37']

with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph(meta_path)
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint(checkpoint_dir_path))

    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    with open(os.path.join(checkpoint_dir_path, 'frozen_graph.pb'), 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
