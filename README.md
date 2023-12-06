# Introduction
This is the code for the paper

Sequential Recommendation for Long-Tail Users based on Knowledge-Enhanced Contrastive Learning(KECL-SR)

The preprocessed datasets are included in the ‘data’ folder, where each line contains an user id and item ids user interactedthat is sorted by timestamp (starting from 1) .

# Implementation
## Requirements

Python >= 3.7  
Pytorch >= 1.2.0  
tqdm == 4.26.0
hnswlib==0.7.0
numba==0.53.1
numpy==1.25.2
pandas==2.0.3
scikit-learn==1.3.0
scipy==1.6.2

## Datasets

two prepared datasets are included in `data` folder. 

| datasets               | ML-1M  | ML-10M   |
| ---------------------- | ------ | -------- |
| number of users        | 6040   | 69878    |
| number of items        | 3698   | 10677    |
| number of interactions | 854573 | 10000054 |
| number of triples      | 707654 | 1492617  |



## Train Model

To train KECL-SR on 'ml-1m' dataset, change to the `src` folder and run following command: 

```
python main.py --data_name ml-1m 
```


## Evaluate Model

You can directly evaluate a trained model on test set by running:

```
python main.py --data_name ml-1m  --do_eval
```

If you want the test set to only include long-tail users, then run the following command:

```
python main.py --data_name ml-1m  --do_eval --tail_test
```



# 引言

这是论文《基于知识增强对比学习的长尾用户序列推荐(KECL-SR)》的代码

预处理的数据集包含在data文件夹中，每一行包含一个用户ID和用户按时间戳（从1开始）交互过的物品ID。

# 实现

## 要求

Python >= 3.7
Pytorch >= 1.2.0
tqdm == 4.26.0 
hnswlib==0.7.0 
numba==0.53.1 
numpy==1.25.2 
pandas==2.0.3 
scikit-learn==1.3.0 
scipy==1.6.2

## 数据集

`data`文件夹中包含两个准备好的数据集。

| 数据集     | ML-1M  | ML-10M   |
| ---------- | ------ | -------- |
| 用户数量   | 6040   | 69878    |
| 物品数量   | 3698   | 10677    |
| 交互数量   | 854573 | 10000054 |
| 三元组数量 | 707654 | 1492617  |

## 训练模型

要在'ml-1m'数据集上训练KECL-SR，切换到`src`文件夹并运行以下命令：

```
python main.py --data_name ml-1m 
```

## 评估模型

你可以通过运行以下命令直接在测试集上评估已训练的模型：

```
python main.py --data_name ml-1m  --do_eval
```

如果你希望测试集只包含长尾用户，那么运行以下命令：

```
python main.py --data_name ml-1m  --do_eval --tail_test
```
