import tensorflow as tf
import tensorflow_hub as hub

model = tf.compat.v1.keras.models.load_model("pepper_output_inference_graph_v1.pb/saved_model", custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
# model = tf.compat.v1.keras.models.load_model("pepper_output_inference_graph_v1.pb/saved_model")
# model = tf.keras.models.load_model("pepper_output_inference_graph_v1.pb/saved_model")
model.summary()
