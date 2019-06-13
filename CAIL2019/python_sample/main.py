import random
import json
import jieba
import numpy as np
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


from collections import Counter
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *
import os
import torch 
from torch.autograd import Variable 
import torch.nn.functional as F

total_nlp_features = 16


#对短信中的用户名前缀和内部的url链接进行过滤删除
def filter(line):
    #剔除日期
    data_regex = re.compile(u"""        #utf-8编码
        年 |
        月 |
        日 |
        (周一) |
        (周二) | 
        (周三) | 
        (周四) | 
        (周五) | 
        (周六)
    """, re.VERBOSE)
    #剔除所有数字
    decimal_regex = re.compile(r"[^a-zA-Z]\d+")

    line = data_regex.sub(r"", line)
    line = decimal_regex.sub(r"", line)

    return line

def trans(sentence,cutwordslist = None):
    texts_cut = [word for word in jieba.lcut(filter(sentence)) if len(word) > 1]
    outstr = ' '.join(texts_cut)
    return outstr


se = set()
f = open("data/input.txt", "r", encoding="utf8")
for line in f:
    x = json.loads(line)
    se.add(x["A"])
    se.add(x["B"])
    se.add(x["C"])
    
cutwordslist = []
data = list(se)
for a in range(0, len(data)):
    data[a] = trans(data[a],cutwordslist)
    
tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",max_features = 300).fit(data)
sparse_result = tfidf_model.transform(data)

f = open("../input/input.txt", "r", encoding="utf8")
ouf = open("../output/output.txt", "w", encoding="utf8")
model = Doc2Vec.load(os.path.join(MODEL_PATH,DOC2VEC))  # you can continue training with the loaded model!




X_test = []
for line in f:
    x = json.loads(line)
    y = [
        model.infer_vector(trans(x["A"]).split()),
        model.infer_vector(trans(x["B"]).split()),
        model.infer_vector(trans(x["C"]).split())
    ]
    
    X_test.append(y)
X_test = np.array(X_test)

        
        
import torch 
from torch.autograd import Variable 
import torch.nn.functional as F

# 定义一个构建神经网络的类 
class Net(torch.nn.Module): # 继承torch.nn.Module类 
    def __init__(self, n_feature = 4492, n_hidden = 256, n_output = 300): 
        super(Net, self).__init__() # 获得Net类的超类（父类）的构造方法 
        # 定义神经网络的每层结构形式 
        # 各个层的信息都是Net类对象的属性 
        self.hidden1 = torch.nn.Linear(n_feature, 4*n_hidden) # 隐藏层线性输出 
        self.dropout1 = torch.nn.Dropout(0.8)
        self.hidden2 = torch.nn.Linear(4*n_hidden, 4*n_hidden) # 隐藏层线性输出 
        self.dropout2 = torch.nn.Dropout(0.5)
        self.predict = torch.nn.Linear(4*n_hidden, n_output) # 输出层线性输出 
        self.margin = 0.5
        
        
    def doc_encoding(self, d):
        d = F.relu(self.hidden1(d)) # 对隐藏层的输出进行relu激活 
        d = self.dropout1(d)
        d = F.relu(self.hidden2(d))
        d = self.dropout2(d)
        d = self.predict(d)  # batch_size x 5
        return d

    # 将各层的神经元搭建成完整的神经网络的前向通路 
    def forward(self,d1,d2,d3): 
        d1 = self.doc_encoding(d1)
        
        d2 = self.doc_encoding(d2)
        
        d3 = self.doc_encoding(d3)
        
        pos_sim=F.cosine_similarity(d1, d2)
        neg_sm=F.cosine_similarity(d1, d3)
        
        loss=(self.margin-pos_sim+neg_sm).clamp(min=1e-6).mean()
        return loss


model_dict = torch.load(os.path.join(MODEL_PATH, MODEL_NAME))
net = Net(X_test.shape[2],256,300)
net.eval()
net.load_state_dict(model_dict)

test_y = []

count= 0 
for each in tqdm(X_test):
    d1 = each[0]
    d2 = each[1]
    d3 = each[2]
    
    d1 = torch.tensor(d1).view(1,-1).float()
    d2 = torch.tensor(d2).view(1,-1).float()
    d3 = torch.tensor(d3).view(1,-1).float()


    d1_encoding = net.doc_encoding(d1)
    d2_encoding = net.doc_encoding(d2)
    d3_encoding = net.doc_encoding(d3)


    d1_2 = F.cosine_similarity(d1_encoding,d2_encoding)

    d1_3 = F.cosine_similarity(d1_encoding,d3_encoding)
    
    
    
    if d1_2 > d1_3:
        print("B", file=ouf)
        count+=1
    else:
        print("C", file=ouf)
        
print('X_test :', X_test.shape[0])
print('B acc :', count/X_test.shape[0])

