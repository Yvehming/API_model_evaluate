# https://www.cnblogs.com/CheeseZH/p/12620404.html
# https://www.cnblogs.com/hellcat/p/6925757.html
# https://www.cnblogs.com/weizhen/p/8451514.html
# from tensorflow.python.tools import inspect_checkpoint as chkp
# chkp.print_tensors_in_checkpoint_file('pepper_inference_graph_v2.pb\model.ckpt', tensor_name=None, all_tensors=True, all_tensor_names=True)
# Total number of params: 3152979
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
ckpt = tf.compat.v1.train.get_checkpoint_state('pepper_inference_graph_v2.pb')
print(ckpt.model_checkpoint_path)
saver = tf.compat.v1.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
with tf.compat.v1.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    graph = tf.compat.v1.get_default_graph()
    variable_name = [v.name for v in tf.compat.v1.trainable_variables()]
    print(variable_name)