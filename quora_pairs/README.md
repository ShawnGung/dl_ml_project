# quora-question-pairs

Kaggle比赛 : https://www.kaggle.com/c/quora-question-pairs/overview
主要目的是了解文本相似度的深度模型

## Run
- download the in [.data](./data) folder
- run the [data_helper.py](data_helper.py), 清理文本数据
- run the [feature_extract.py](feature_extract.py), 提取NLP文本特征
- run the [leaky_feature.py](data_helper.py), 提取本比赛的特殊特征(magic feature)
- run the [training_stacking.ipynb](training_stacking.ipynb), 提取LSTM模型的中间特征
- run the [xgboost.ipynb](xgboost.ipynb.ipynb), 把特殊特征和LSTM的中间特征融合,利用xgboost训练融合特征


## 模型:
1. 利用LSTM模型进行训练,中间添加了NLP文本特征,单个模型LB ~0.25
2. 在训练好的LSTM模型中取中间隐藏特征 concate到 magic feature上
3. 利用xgboost再融合特征上训练, 最终LB ~0.17



## reference:

1. magic features : https://www.kaggle.com/jturkewitz/magic-features-0-03-gain. 

