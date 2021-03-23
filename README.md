# Botzone国标麻将深度学习&强化学习

**北京大学强化学习课程2020秋季学期课程大作业**


## 文件结构简介
- 深度学习部分文件为`filereader.py`和`DL_agent.py`，前者用于将初始训练数据转化为与botzone相同的输入输出，后者用于训练DL模型，并且可以调整参数变更为可以在botzone运行的bot。
- 强化学习部分文件为`PolicyGradient.py`，`A3C.py`以及`PolicyGradient_naive.py`，前者为主要研究的部分，后两者为初期尝试的代码，没有经过很好的debug和维护。
- 工具文件为`model_handler.py`，用于将模型数据缩小以在botzone运行。
- 数据文件中包括人类对局原始数据，深度学习的训练数据（即filereader.py得到的结果），训练得到的模型三部分。深度学习得到的初始模型为models文件夹中的super_model_2，经过强化学习训练得到的模型为rl_pg_new，用于botzone的模型为super_model_small。
  + 文件连接：[百度网盘](https://pan.baidu.com/s/1wpPBHq3MRngMQx9EAS6-aw )
  + 提取码：agmm 


## 代码运行方式
**所有代码均支持命令行运行，使用`python xxx.py`即可在默认参数下运行，具体参数解释如下**

#### filereader.py
-  `-h, --help` 展示运行帮助信息
- `-lp, --load_path` 原始训练数据所在文件夹
- `-sp, --save_path` 处理后的训练数据输出文件夹，需要事先创建
- `-tn, --thread_number, default=32` 线程总数，线程i处理的文件编号为 i + k * thread_number, k = 0, 1, 2...
- `-tr, --thread_round, default=4` 每个线程运行多少轮, 实际线程数量为tn/tr， 推荐该值和cpu实际核数相同

#### DL_agent.py
该文件可以上传botzone作为bot使用
- 试运行推荐`python DL_agent.py -t -l -lp path_to_pretrained_model`
-  `-h, --help` 展示运行帮助信息
-  `-t, --train, default=False` 是否训练模型
-  `-s, --save, default=False` 是否保存模型
-  `-l, --load, default=False` 是否加载预训练的模型
- `-lp, --load_path` 加载模型的路径
- `-sp, --save_path` 保存模型的路径
-  `-b, --batch_size, default=1000` 训练用的batch大小
-  `--training_data` 训练数据所在文件夹，和filereader.py中的输出文件夹相同
-  `-pi, --print_interval, default=2000` 每隔这么多局输出总体预测正确率
- `-si, --save_interval, default=5000` 每隔这么多局保存模型
- `--botzone, default=False` 该选项不能在命令行设置。如果将该文件传到botzone作为bot，需要在程序中设置`parser.add_argument('--botzone', action='store_true', default=True, help='whether to run the model on botzone')`

#### PolicyGradient.py
- 试运行推荐`python PolicyGradient.py -p 1 -o path_to_pretrained_model -n path_to_pretrained_model -S path_to_rl_model -s -lp none -bs 500 -lr 8e-6`
- 由于其余两种强化学习的尝试均效果不佳且速度很慢，故没有在后续进行维护。其运行参数基本上被该方法包含，不再在此列出，详情可以通过`python A3C.py -h`以及`python PolicyGradient_naive.py`进行查看。
-  `-h, --help` 展示运行帮助信息
-  `-lr, --learning_rate, default=1e-5` 学习率
-  `-s, --save, default=False` 是否保存模型
- `-o, --old_path` 预训练模型路径，使用这个模型的bot用于作为对手，不会被训练
- `-n, --new_path` 预训练模型路径，使用这个模型的bot用于训练
-  `-S, --save_path` 保存模型路径，设置-s后才有效果
-  `-p, --num_process_per_gpu, default=1` 每个GPU上运行多少进程，如果使用CPU则为总进程数
-  `-rn, --round_number, default=10` 每个episode运行多少个不同的对局
-  `-rt, --repeated_times, default=10` 每个对局重复多少次，即每个episode并发运行rn * rt场游戏
-  `-ti, --train_interval, default=2` 每隔多少episodes进行训练，即更新全局模型。如果设置过大的ti，会导致manager线程保存数据过多而引发错误，具体原因未知
-  `-ji, --join_interval, default=2` 训练多少次后将全局模型更新到所有本地模型，即经过ti * ji个episodes
-  `-pi, --print_interval, default=10` 每隔多少episodes进行输出
- `-si, --save_interval, default=20` 每隔多少episodes保存模型
- `--eps_clip, default=0.3` 重要性采样中的比例截断
- `--entropy_weight, default=1e-3` 初始的entropy loss权重
- `--entropy_target, default=1e-4` 平均entropy的目标值，用于动态更新entropy loss权重
- `--entropy_step, default=0.01` 动态更新的步长
- `-lp, --log_path` 保存输出的文件路径，设为none则不保存输出。不影响标准输出中的输出
- `-e, --epochs, default=1` 训练时重复利用训练数据的轮次。各种资料表明不重复利用效果最好，为了效率考虑，目前的实现中设置不为1的数仅仅相当于提高学习率。
- `-bs, --batch_size, default=1000` 训练时训练数据的最大batch size，用于防止显存溢出

#### model_handler.py
- 用于将训练好的模型转换为botzone上可以运行的、较小的模型，去除模型中的轮次、优化器信息
- 运行方式：`python model_handler.py -o path_to_original_model -n path_to_smaller_model`
