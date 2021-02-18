## MPC - Model Predictive Control

该文件夹包含MPC算法的实现及其评估。

我们在三个 Quanser Robots 环境 Qube、CartPoleSwing 和 Ballbalancer 上进行了实验。如果你想对其中一个环境有更多的了解，请进入相应的文件夹

### 如何运行

1. 选择你想要进行实验的环境并进入相应的文件夹
2. 跟随文件夹中的指引运行相关程序。

### 每个环境文件夹中包含：

```controller.py```

MPC controller 的实现。如果想要改变 controller 的优化方法， 请在此文件中进行改动。

```dynamics.py```

model 的构建以及训练数据的生成。model 为四层的神经网络，用来对真实环境进行模拟。

```utils.py```

包含对环境参数的分析和目前实验超参数设置的分析。

```run.py```

主程序，可以直接使用python运行进入训练。训练出来的模型和训练使用的数据默认保存在```storage``` 文件夹中。

```configuration.yml```

对于实验超参数的设置。如果想要改变超参数，请在本文件中进行修改。

### data文件夹

包含了我们针对不同环境和不同超参数训练出的数据和模型，其中还有我们用于画图的相关程序。



