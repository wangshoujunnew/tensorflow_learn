# LSTM
## 1. 忘记门Forget Gate Layer 激活函数:sigmoid(cita)
> ![name][01]{:target="_blank"}
[01]: https://img-blog.csdn.net/20171116154539468 "忘记门矩阵"

> 上一层的输出信息ht-1和当前信息Xt进行线性组合后，利用激活函数，将其函数值进行压缩，得到一个大小在0和1之间的阈值。当函数值越接近1,表示记忆体保留的信息越多。当函数值接近0，表示记忆体丢失的信息越多

## 2. 输入门input Gate Layer(当前时刻的输入信息Xt,有多少信息被加到信息流里面) 激活函数: sigmoid(cita)
> ![name][02]
[02]: https://img-blog.csdn.net/20171116154601824 "输入门矩阵"

## 3. 候选门Candidate Layer(当前的输入和过去的记忆所具有的信息的总和)  公式是图二中的Ct
>记忆的更新由两个部分组成 :第一部分是通过忘记门过滤之前的记忆，大小为ft x Ct-1.第二部分是添加当前的新增的数据信息，添加的比比例由输入门控制，大小为it xCt..得到更新后的记忆信息如下：Ct
> ![name][03]
[03]:https://img-blog.csdn.net/20171116154651890  "候选们矩阵"

## 4. 输出门output gate Layer(输出门控制着有多少记忆信息被用于下一个阶段的更新中) 激活函数sigmoid(cita)
>输出门控制着有多少记忆信息被用于下一个阶段的更新中，输出门依然使用sigmoid激活函数，公式如下Ot.
Ot是一个大小在0和1之间的权重值，传递给下一个阶段信息是上图的ht.
上图是LSTM网络几个基本概念。隐藏层几个门的设计，可能根据实际的需求来分析。
公式总结：
>![name][04]
[04]: https://img-blog.csdn.net/20171116154704063 "输出门矩阵"
