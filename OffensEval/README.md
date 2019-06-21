# the OffensEval 2019 challenge

该项目为 [the OffensEval 2019 challenge](https://competitions.codalab.org/competitions/20011) 的代码和模型


## 模型:
1. [TF-IDF BASELINE](./offEval/Tfidf.ipynb)
2. [SVM](./offEval/SVMGlove.ipynb)
3. [TextCNN](./offEval/testCNNGlove.ipynb)
4. [Bert](./BERT/Bert-TextClassification-master)


## 各指标对比:
### Task_A
Model | Micro F1 | Macro F1 | Acc |
----|---- |---- |---- |
TF-IDF baseline | 0.73 | 0.63 | 0.73 | 
SVM(GLOVE BOW) | 0.75 | 0.71 | 0.75 | 
TextCNN(GLOVE) | **0.81** | 0.77 | **0.81** | 
BERT(base-uncased) | - | **0.803** | 0.80 | 



### Task_B
Model | Micro F1 | Macro F1 | Acc |
----|---- |---- |---- |
TF-IDF baseline | 0.77 | 0.58 | 0.77 | 
SVM(GLOVE BOW) | 0.53 | 0.46 | 0.53 | 
TextCNN(GLOVE) | **0.89** | 0.55 | 0.88 | 
BERT(base-uncased) | - | **0.60** | **0.90** | 



### Task_C
Model | Micro F1 | Macro F1 | Acc |
----|---- |---- |---- |
TF-IDF baseline | 0.64 | 0.48 | 0.64 | 
SVM(GLOVE BOW) | **0.65** | 0.49 | 0.65 | 
TextCNN(GLOVE) | 0.63 | 0.52 | 0.63 | 
BERT(base-uncased) | - | **0.58** | **0.65** | 

