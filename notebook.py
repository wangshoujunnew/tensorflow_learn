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


# 彩色图像 CNN
1: 读取二进制文件
** tf的数据读取
. 文件名列表交给 tf.train.string_input_producer 函数 # string_input_producer来生成一个先入先出的队列， 文件阅读器会需要它来读取数据。
A: 文件名shuff + 最大迭代次数epoch limits
** 文件阅读器
CSV:  TextLineReader和decode_csv

# 每次read的执行都会从文件中读取一行内容， decode_csv 操作会解析这一行内容并将其转为张量列表
# --------------------------   # 
filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])

reader = tf.TextLineReader()
 ----------------------------- #
key, value = reader.read(filename_queue)  # key 输入的文件

# 读取二进制 
tf.FixedLengthRecordReader的tf.decode_raw # decode_raw操作可以讲一个字符串转换为一个uint8的张量。
the CIFAR-10 dataset的文件格式 :  一个字节的标签，后面是3072字节的图像数据
