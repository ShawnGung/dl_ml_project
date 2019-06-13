import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import json
from string import punctuation
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from config import *
import jieba


def stopwordslist(filepath = SW_PATH):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def text_to_wordlist(line, remove_stop_words=True):
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
    
    texts_cut = [word for word in jieba.lcut(line) if len(word) > 1]
    outstr = ' '.join(texts_cut)

    return outstr



def process_questions(question_list, questions, question_list_name):
    '''transform questions and display progress'''
    print("==="+question_list_name+' is being processed')
    for question in tqdm(questions):
        question_list.append(text_to_wordlist(question))
        

       
def clean_dataset():
    f = open(INPUT_DATA, "r", encoding="utf8")
    train_d1 = []
    train_d2 = []
    train_d3 = []
    
    for line in f:
        x = json.loads(line)
        train_d1.append(x["A"])
        train_d2.append(x["B"])
        train_d3.append(x["C"])
        
        # Preview some transformed pairs of questions
    print('===questions sample before cleaning')
    print(train_d1[0])
    print('length:',len(train_d1[0]))
    print()
    
    processed_train_d1 = []
    process_questions(processed_train_d1,train_d1, 'train_d1')
    
    processed_train_d2 = []
    process_questions(processed_train_d2, train_d2, 'train_d2')
    
    processed_train_d3 = []
    process_questions(processed_train_d3,train_d3, 'train_d3')
    
    
        # Preview some transformed pairs of questions
    print('===questions sample after cleaning')
    print(processed_train_d1[0])
    print('length:',len(processed_train_d1[0]))
    print()
    return processed_train_d1,processed_train_d2,processed_train_d3


def tokenization(train_d1,train_d2,train_d3):
    #进行切分词,然后得到word_to_index字典,以及换成index的句子数据集
    # make word index
    questions = train_d1 + train_d2 + train_d3
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)
    train_d1_word_sequences = tokenizer.texts_to_sequences(train_d1)
    train_d2_word_sequences = tokenizer.texts_to_sequences(train_d2)
    train_d3_word_sequences = tokenizer.texts_to_sequences(train_d3)
    word_index = tokenizer.word_index
    print('words in index:', len(word_index))
    return word_index,train_d1_word_sequences,train_d2_word_sequences,train_d3_word_sequences


def get_word2vec_embedding2():
    #加载glove的embedding的look up table
    print("Processing", WORD_FILE)

    embeddings_index = {}
    with open(WORD_FILE, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings: %d' % len(embeddings_index))
    return embeddings_index

def get_word2vec_embedding(path = WORD_FILE, topn = MAX_NB_WORDS):  # read top n word vectors, i.e. top is 10000
    #加载glove的embedding的look up table
    print("Processing", WORD_FILE)
    
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    
    print('Word embeddings: %d' % len(vectors))
    return vectors, iw, wi, dim


def get_word_embedding_matrix(word_index,embeddings_index):
    #得到我们当前字典中的look_up_table,bn_words是当前字典的大小
    nb_words = min(MAX_NB_WORDS, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector

    print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))
    return word_embedding_matrix,nb_words


def pad_sequence(train_d1_seq,train_d2_seq,train_d3_seq):
    #对每个以index标记的句子,填充pad到MAX_SEQUENCE_LENGTH
    train_d1_data = pad_sequences(train_d1_seq, maxlen=MAX_SEQUENCE_LENGTH)
    train_d2_data = pad_sequences(train_d2_seq, maxlen=MAX_SEQUENCE_LENGTH)
    train_d3_data = pad_sequences(train_d3_seq, maxlen=MAX_SEQUENCE_LENGTH)
    
    print('Shape of train question1 data tensor:', train_d1_data.shape)
    print('Shape of train question2 data tensor:', train_d2_data.shape)
    print('Shape of train question3 data tensor:', train_d3_data.shape)
    return train_d1_data,train_d2_data,train_d3_data

def save_tmpfiles(train_d1_data,train_d2_data,train_d3_data,word_embedding_matrix,nb_words):
    #把所有文件先保存好
    np.save(open(D1_TRAINING_DATA_FILE, 'wb'), train_d1_data)
    np.save(open(D2_TRAINING_DATA_FILE, 'wb'), train_d2_data)
    np.save(open(D3_TRAINING_DATA_FILE, 'wb'), train_d3_data)
    np.save(open(WORD_EMBEDDING_MATRIX_FILE, 'wb'), word_embedding_matrix)
    with open(NB_WORDS_DATA_FILE, 'w') as f:
        json.dump({'nb_words': nb_words}, f)
        
        
def calculate_averageLength(train_set):
    data = []
    total_length = 0
    for each in train_set:
        data += each
    
    for d in data:
        total_length += len(d)
    print('avg_lenght : ',total_length // len(data))
    return total_length // len(data)
    
       
if __name__ == '__main__':
    train_d1,train_d2,train_d3 = clean_dataset()
    word_index,train_d1_seq,train_d2_seq,train_d3_seq = tokenization(train_d1,train_d2,train_d3)
#     averageLength = calculate_averageLength([train_d1_seq,train_d2_seq,train_d3_seq])
    vectors, iw, embedding_index, dim = get_word2vec_embedding()
    word_embedding_matrix,nb_words = get_word_embedding_matrix(word_index,embedding_index)
    train_d1_data,train_d2_data,train_d3_data = pad_sequence(train_d1_seq,train_d2_seq,train_d3_seq)
    save_tmpfiles(train_d1_data,train_d2_data,train_d3_data,word_embedding_matrix,nb_words)

