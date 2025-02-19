# Introduction
This is the code for the paper

Knowledge Graph-Augmented Contrastive Learning for Long-Tail User Sequential Recommendation

The preprocessed dataset is contained in the "data" folder, where each line includes a user ID and the corresponding item IDs that the user interacted with, along with timestamps (starting from 1).

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

To train LTCL on 'ML-1M' and 'ML-10M' datasets, change to the `src` folder and run following command: 

```
python main.py -- data_name ML-1M
python main.py -- data_name ML-10M
```


## Evaluate Model (ML-1M as example)

You can directly evaluate a trained model on test set by running:

```
python main.py --data_name ml-1m  --do_eval
```

If you want the test set to only include long-tail users, then run the following command:

```
python main.py --data_name ml-1m  --do_eval --tail_test
```
