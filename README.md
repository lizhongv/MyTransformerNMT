>>>>>>> HEAD
## MyTransformer
======
这是一个利用transformer进行机器翻译的简单例子。

## 论文来源

[Attention is all you need](https://arxiv.org/abs/1706.03762)

## 代码参考

[harvardnlp](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
[]()

## 项目结构

- data `源数据目录`
    - dev.txt `测试数据集`
    - train.txt `训练数据集` 
    - vocab_en `英文词汇表`
    - vocab_zh `中文词汇表`
- log  `日志存放目录` 
    - log-timestamp.txt `每次预测产生一个此类文件`
    - train.log `训练记录`
    - eval.log `测试记录`
- save `模型存放目录`
    - model.pt `训练好模型` 
- model `模型目录`
    - attention.py
    - embedding.py
    - encoder.py
    - decoder.py
    - generator.py
    - sublayer.py
    - position_wise_feedforward.py
    - transformer.py
- lib  `损失函数、优化器等存放位置`
    - criterion.py `损失函数`
    - optimizer.py `优化器`
    - loss.py `优化器 + 损失函数封装类`
- evaluate.py `预测.py`
- train.py `训练.py`
- parser.py `参数.py`
- utils.py `工具类.py`
- run.py `入口文件.py`
- README.md `readme`


## 训练
```
python3 run.py
nohup python ./run.py >> ./log/train.log 2>&1 &
```

## 预测 (前提：训练过)

```
python3 run.py --type evaluate
nohup python3 run.py --type evaluate >> ./log/eval.log 2>&1 &
```
=======
## MyTrasformer
>>>>>>> origin/main
