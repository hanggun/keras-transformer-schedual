# keras-transformer-schedual
keras implementation of transformer schedual

## 使用方法
```python
from bert4keras.optimizers import Adam
from optimizer import extend_with_transformer_schedule
Adamt = extend_with_transformer_schedule(Adam)
optimizer=Adamt(learning_rate=1,
                start_step=0,
                warmup_steps=4000,
                d_model=512,
                beta_1=0.9,
                beta_2=0.98
                )
```

通过将learning_rate设置为1，模型梯度下降的学习等于transformer schedule给出的学习率，start_step可以方便中断后继续训练
其他介绍可以在[基于keras的transformer learning rate schedule](https://blog.csdn.net/qiongyaoxinpo/article/details/126744099?spm=1001.2014.3001.5501)参考
