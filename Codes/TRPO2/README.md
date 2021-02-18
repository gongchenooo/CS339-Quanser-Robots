# Trust Region Policy Gradient (TRPO)
```train.py``` ：

主函数，可以直接运行 python3 train.py进行训练。模型会保存在save_models文件夹中，训练数据会保存在logs文件夹中

```arguments.py```：

参数文件，里面包含要训练的环境名称、折扣因子、批数据量、模型存放地址等参数，训练不同模型和不同超参可直接修改此文件然后运行 python3 train.py

```models.py```：

包含两个具有三层的神经网络Value_net和Policy_net，前者用来估计state的价值，后者用来选择state所对应的策略

```trpo_agent.py```：

包含类trpo_agent。类中learn()函数是整个模型的训练过程,包含_update_network()、_get_surrogate_loss()、_fisher_vector_product()、_get_kl()等函数用来进行神经网络Value_net优化、获取surrogate函数的loss用来对Policy_net进行优化、获取费舍尔矩阵（二阶导数）、获取两个策略分布的kl散度

```utils.py```：

包含trpo_agent所用到的一些函数如select_actions()、 eval_actions()、 conjugated_gradient()、line_search()等

```demo.py```：

用来运行测试我们训练好存下来的模型，并将环境render出来查看效果，可以修改```arguments.py```文件里面的env_id来测试不同环境的训练效果

```draw.py```：

用来对保存下来的数据处理并作图

```log文件```：记录各个环境下，用不同超参时的reward和policy loss

```saved_models文件```：包含各个环境下不同超参训练出来的模型

