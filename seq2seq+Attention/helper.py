from config import *
import math
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import io
import torchvision
from PIL import Image
import matplotlib.ticker as ticker
plt.rcParams[u'font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False


# 把句子变成index的list，最后加上EOS
def indexes_from_sentence(lang, sentence):
    idxs=[]
    for word in sentence.split(' '):
        if word in lang.word2index:
            idxs.append(lang.word2index[word])
    idxs.append(EOS_token)
    return idxs

# 把一个序列padding到长度为max_length。
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
