# tensorflow 组件
. tensor  # 数据
. Variable # 变量"维护状态"
. feed fetch # 任意操作赋值和获取数据
. session 
. graph

# python的作用: 只是构建计算图,真正的计算在python外部

# 构建图
# 构件图的形式
创建常量     # matrix1 = tf.constant([[3., 3.]])    op1
加减乘除操作 # product = tf.matmul(matrix1, matrix2) op2

# GPU如何参与运算->指明GPU
with tf.Session() as sess:
  with tf.device("/gpu:1"): # /cpu:0 机器的CPU, /gpu:0 机器的第一个GPU,如果有的话
    matrix1 = ...
    matrix2 = ...
    product = tf.matmul(matrix1, matrix2)
    
 
# 交互式Session
可以使用 InteractiveSession 代替 Session 类
用 Tensor.eval() 和 Operation.run() 方法代替 Session.run(). 这样可以避免使用一个变量来持有会话.
# 使用初始化器 initializer op 的 run() 方法初始化 'x' 
x.initializer.run()
print x.eval()

# 数据的输入 
. 数据类型 
. shape (包含batch_size)
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                       IMAGE_PIXELS))

# TensorFlow构建图表3步
1.inference(推理) —— 尽可能地构建好图表，做到返回包含结果的Tensor
2.loss() —— 往inference图表中添加生成损失（loss）所需要的操作（ops）。
3.training() —— 往损失图表中添加计算并应用梯度（gradients）所需的操作。

 每一层都创建于一个唯一的tf.name_scope
 with tf.name_scope('hidden1') as scope:
  
# 状态可视化
summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, # 实例化SummaryWriter, 写入包含图表本身和即时数据具体值的数据文件
                                        graph_def=sess.graph_def)
