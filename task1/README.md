## 任务:基于机器学习的文本分类

实现基于logistic regression的文本分类

 
### 数据

清华大学自然语言处理实验室推出的中文文本分类工具包THUCTC
http://thuctc.thunlp.org<br />

### 运行

训练：python main.py

单条语句测试：python predict.py

### 输入与输出

输入：描述语句

输出：10个候选分类类别：体育、娱乐、家居、房产、教育、时尚、时政、游戏、科技、财经

### 评价指标

交叉熵损失函数

### 数据划分

留出法

### 数据预处理

特征值：jieba库分词+tf-idf算法（词袋模型）

标签：one-hot算法

### 模型

单隐藏层神经网络+梯度降优化

| 文件                | 功能
|:--------------:  |:-------------:|
|config/lr_config.py|配置路径与参数|
|data/cnews.train.txt|数据集|
|data/stopwords.txt|停用词|
|data/categories.txt|分类类别|
|datahelper/data_process.py|数据处理|
|lr_model.py|网络模型|
|main.py|训练程序|
|predict.py|预测程序|

### 缺点

1.词袋模型(BOW)中： 
* 无法反映词之间的关联关系
* 需要深层分析的场合效果太差

2.数据集只划分训练集和验证集，且验证集仅做测试数据使用，未用之调参