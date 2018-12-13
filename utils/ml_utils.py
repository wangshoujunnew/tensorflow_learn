import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report  # 分类报告


# 画图函数: 只画两种点类型 + 折线图 回归问题
def my_scatter_line(points, colors, zhexian_points=None, line_colors=None):
    '''
    [[1,2],[1,2]]
    plt.scatter 分成两部分 [x],[y],mark 标志是什么
    例子:
    arr = np.array([[1, 2], [2, 3], [3, 3]])
    my_scatter_line([arr],['green'])
    :return:
    '''

    for point, color in zip(points, colors):
        plt.scatter(list(point[:, 0]), list(point[:, 1]), edgecolors=color)

    if zhexian_points:
        for zhexian_point, color in zip(zhexian_points, line_colors):
            plt.plot(zhexian_point[:, 0], zhexian_point[:, 1], color=color)
            plt.legend()

    plt.show()


# 分类画图
def my_cluster():
    pass


# 定义画图函数
def plot_classifier(classifier, X, y):
    """

    :param classifier: 分类器
    :param X: 数据点
    :param y: 该点对应的分类
    :return:
    """
    # 图形的取值范围
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0  # 第0列的所有行
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0  # 第1列的所有行

    # 设置网格数据的步长
    step_size = 0.01
    # 网格（grid）数据求出方程的值，画边界
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size),
                                     np.arange(y_min, y_max, step_size))

    # 计算分类器输出结果 ** # 对网格中的每个点进行预测分类 x_values.ravel() x_values的所有值(包括各个维度)
    # classifier已经训练好了,然后对网格的每个点进行训练
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # print(len(mesh_output)) 675000 np.c_[x_values.ravel(), y_values.ravel()] shape = (67500,2) 产生了67500这么多个样本

    # 数组维度变形
    mesh_output = mesh_output.reshape(x_values.shape)

    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1,
                cmap=plt.cm.Paired)

    # 设置图形的取值范围
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    # 设置X轴与Y轴
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))
    plt.show()


def add_type(func):
    def wrapper(something, name=''):  # 指定一毛一样的参数
        print("#[DEBUG]:name:{name}, type: {type}".format(name=name, type=type(something)))
        print('#function:', end='')
        print([x for x in something.__dir__() if (not x.startswith('_') and not x.endswith('_'))])
        print('#doc:', end='')
        print(something.__doc__)
        print('#value:', end='')
        return func(something)

    return wrapper  # 返回包装过函数


@add_type
def my_print(someting, name=''):
    print(someting)


# 对预测结果进行评估
def predict_score(y_pred, y_true, class_name):
    """
    精度: 分类正确的样本数65/总分类的样本数73
    召回率: 数据集中我们感兴趣的样本数量65 / 分类正确的样本数量82
    F1得分: 2 * 精度 * 召回率 / (精度 + 召回率)
    :param y_pred: 预测值
    :param y_true: 真实值
    :param class_name: 分类名称
    :return:
    """
    max_value = max(y_true) if max(y_true) > max(y_pred) else max(y_pred)
    max_value = max_value + 1

    confusion_mat = confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(confusion_mat, max_value):
        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Paired)

        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(max_value)  # 4 和输入预测的最大值有关
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    print(classification_report(y_true, y_pred, target_names=class_name))  # 打印报告
    plot_confusion_matrix(confusion_mat, max_value)


