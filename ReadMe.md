## 对抗训练在 TextCNN 中的应用
### 复现流程   
1. **任务分析**: 将论文中的对抗训练策略应用于TextCNN中    
    1.1 TextCNN 是什么：利用CNN实现NLP的神经网络算法   
    1.2 对抗训练是什么： 通过对训练样本添加扰动，提高模型鲁棒性
2. **复现 TextCNN**：下载源码，复现Baseline训练流程，得到Baseline结果
3. **对抗训练实现**：根据文中提供的对抗训练算法表，编写 adversarial_strategy.py，其中编写方式借鉴知乎专栏文章[https://zhuanlan.zhihu.com/p/91269728]   ，以脚本方式编写
4. **对抗训练**: 根据对应策略改写 TextCNN 源码中的 train_eval(), 提供新的训练方式
5. **代码整理**：删除非必要文件，整理 main.py，定义网络关键参数，定义运行flow
6. **输出结果**：分别在命令行内输入可得Baseline，PGD，Free，FGSM (random initial)的结果
    ```
    # 训练并测试：
    # TextCNN Baseline
    python main.py --adversarial_model Baseline

    # TextRNN with PGD
    python main.py --adversarial_model PGD --pgd_steps 7 #默认为7

    # TextRNN with Free
    python main.py --adversarial_model Free --minibatch_replays 10 #默认为10

    # TextRCNN with FGSM
    python main.py --adversarial_model FGSM #alpha设为2
    ```
7. **输出报告**：详细见当前目录<总结报告.docx>
