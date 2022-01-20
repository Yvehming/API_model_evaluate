# https://www.cnblogs.com/CheeseZH/p/12620404.html
# https://www.cnblogs.com/hellcat/p/6925757.html
# https://www.cnblogs.com/weizhen/p/8451514.html
from tensorflow.python.tools import inspect_checkpoint as chkp
chkp.print_tensors_in_checkpoint_file('pepper_output_inference_graph_v1.pb\model.ckpt', tensor_name=None, all_tensors=True, all_tensor_names=True)
# Total number of params: 3152979
# Total number of params: 4648891