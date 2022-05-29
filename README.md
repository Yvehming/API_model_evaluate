# Tensorflow Object Detection API生成模型指标的评估
## saved_model模型的评估
利用export_inference_graph.py输出检查点对应的静态图模型文件(saved_model格式)。

saved_model格式文件类型是TensorFlow推荐的模型格式，包含权值和计算。

由于Tensorflow Object Detection API的config文件中规定了模型输入图像的尺寸调整，因此可以不用对输入图片进行resize。

运行get_map.py计算模型的各项指标，见map_out文件夹。

## tflite模型的评估

.tflite文件没有规定模型尺寸调整，因此需要将图片送入模型前resize。

运行get_map_lite.py计算模型的各项指标，见lite_detection-results文件夹。