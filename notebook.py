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
