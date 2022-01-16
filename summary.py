# https://www.cnblogs.com/CheeseZH/p/12620404.html

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
node = []
saved_model_dir = "pepper_inference_graph_v2.pb/saved_model"
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], saved_model_dir)
    graph = tf.get_default_graph()
    for n in tf.get_default_graph().as_graph_def().node:
        node.append(n.name)
    print(node)
    # t2 = tf.get_default_graph().get_tensor_by_name(name='FeatureExtractor/MobilenetV1/Conv2d_0/weights:0')
    variable_name = [c.name for c in tf.trainable_variables()]
    print(variable_name)
