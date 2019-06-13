import re
import pandas as pd
import distance
import json
from fuzzywuzzy import fuzz

from string import punctuation
import jieba
from config import *
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from pandas import Series,DataFrame

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)


def stopwordslist(filepath = SW_PATH):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

STOP_WORDS = stopwordslist('data/stopwords.txt')  # 这里加载停用词的路径     
SAFE_DIV = 0.0001

def preprocess(line, remove_stop_words=True):
   
    sentence_seged = jieba.cut(line,cut_all=False)

    text = " ".join(sentence_seged)
    return text

def get_token_features(q1, q2):
    token_features = [0.0]*11

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    
    
    
    token_features[10] = jaccard_similarity(q1_tokens, q2_tokens)
    return token_features


def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)
    
    
    
def extract_features(df):
    df["d1"] = df["d1"].fillna("").apply(preprocess)
    df["d2"] = df["d2"].fillna("").apply(preprocess)
    df["d3"] = df["d3"].fillna("").apply(preprocess)
    df_1 = df.copy()
    df_2 = df.copy()

    print("d1 and d2 token features...")
    token_features_1 = df_1.apply(lambda x: get_token_features(x["d1"], x["d2"]), axis=1)
    df_1["cwc_min"]       = list(map(lambda x: x[0], token_features_1))
    df_1["cwc_max"]       = list(map(lambda x: x[1], token_features_1))
    df_1["csc_min"]       = list(map(lambda x: x[2], token_features_1))
    df_1["csc_max"]       = list(map(lambda x: x[3], token_features_1))
    df_1["ctc_min"]       = list(map(lambda x: x[4], token_features_1))
    df_1["ctc_max"]       = list(map(lambda x: x[5], token_features_1))
    df_1["last_word_eq"]  = list(map(lambda x: x[6], token_features_1))
    df_1["first_word_eq"] = list(map(lambda x: x[7], token_features_1))
    df_1["abs_len_diff"]  = list(map(lambda x: x[8], token_features_1))
    df_1["mean_len"]      = list(map(lambda x: x[9], token_features_1))
    df_1["jaccard"]      = list(map(lambda x: x[10], token_features_1))
    
    print("d1 and d2 fuzzy features..")
    df_1["token_set_ratio"]       = df_1.apply(lambda x: fuzz.token_set_ratio(x["d1"], x["d2"]), axis=1)
    df_1["token_sort_ratio"]      = df_1.apply(lambda x: fuzz.token_sort_ratio(x["d1"], x["d2"]), axis=1)
    df_1["fuzz_ratio"]            = df_1.apply(lambda x: fuzz.QRatio(x["d1"], x["d2"]), axis=1)
    df_1["fuzz_partial_ratio"]    = df_1.apply(lambda x: fuzz.partial_ratio(x["d1"], x["d2"]), axis=1)
    df_1["longest_substr_ratio"]  = df_1.apply(lambda x: get_longest_substr_ratio(x["d1"], x["d2"]), axis=1)
    
    
    
    
    
    print("d1 and d3 token features...")
    token_features_2 = df_2.apply(lambda x: get_token_features(x["d1"], x["d3"]), axis=1)
    df_2["cwc_min"]       = list(map(lambda x: x[0], token_features_2))
    df_2["cwc_max"]       = list(map(lambda x: x[1], token_features_2))
    df_2["csc_min"]       = list(map(lambda x: x[2], token_features_2))
    df_2["csc_max"]       = list(map(lambda x: x[3], token_features_2))
    df_2["ctc_min"]       = list(map(lambda x: x[4], token_features_2))
    df_2["ctc_max"]       = list(map(lambda x: x[5], token_features_2))
    df_2["last_word_eq"]  = list(map(lambda x: x[6], token_features_2))
    df_2["first_word_eq"] = list(map(lambda x: x[7], token_features_2))
    df_2["abs_len_diff"]  = list(map(lambda x: x[8], token_features_2))
    df_2["mean_len"]      = list(map(lambda x: x[9], token_features_2))
    df_2["jaccard"]      = list(map(lambda x: x[10], token_features_2))    
    
    print("d1 and d3 fuzzy features..")
    df_2["token_set_ratio"]       = df_2.apply(lambda x: fuzz.token_set_ratio(x["d1"], x["d3"]), axis=1)
    df_2["token_sort_ratio"]      = df_2.apply(lambda x: fuzz.token_sort_ratio(x["d1"], x["d3"]), axis=1)
    df_2["fuzz_ratio"]            = df_2.apply(lambda x: fuzz.QRatio(x["d1"], x["d3"]), axis=1)
    df_2["fuzz_partial_ratio"]    = df_2.apply(lambda x: fuzz.partial_ratio(x["d1"], x["d3"]), axis=1)
    df_2["longest_substr_ratio"]  = df_2.apply(lambda x: get_longest_substr_ratio(x["d1"], x["d3"]), axis=1)
    
    
    
    
    
    
    
    return df_1,df_2
    
    
    

def changeIntoDataframe():
    f = open(INPUT_DATA, "r", encoding="utf8")
    train_d1 = []
    train_d2 = []
    train_d3 = []

    for line in f:
        x = json.loads(line)
        train_d1.append(x["A"])
        train_d2.append(x["B"])
        train_d3.append(x["C"])

    data = {
        'd1':Series(train_d1),
        'd2':Series(train_d2),
        'd3':Series(train_d3)
        }
    df = DataFrame(data)
    return df



print("Loading Dataframe from dataset:")
train_df = changeIntoDataframe()
print("Extracting features for train:")
train_df_1, train_df_2 = extract_features(train_df)

train_df_1.drop(["d1", "d2", "d3"], axis=1, inplace=True)
train_df_1.to_csv(FEATURE_TRAIN_1, index=False)

train_df_2.drop(["d1", "d2", "d3"], axis=1, inplace=True)
print(train_df_2)


train_df_2.to_csv(FEATURE_TRAIN_2, index=False)

# print("Extracting features for test:")
# test_df = pd.read_csv("data/test.csv")
# test_df = extract_features(test_df)
# test_df.drop(["test_id", "question1", "question2"], axis=1, inplace=True)
# test_df.to_csv("data/nlp_features_test.csv", index=False)
