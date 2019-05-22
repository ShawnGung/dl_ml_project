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


# clean dataset, ref : https://www.kaggle.com/currie32/the-importance-of-cleaning-text

def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # text: string
    # return : string
    # remove_stop_words : list of string
    # stem_Wrods: list of string
    # Clean the text, with the option to remove stop_words and to stem words.
    

    stop_words = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
                  'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
                  'Is','If','While','This']

    
    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


def process_questions(question_list, questions, question_list_name, dataframe):
    '''transform questions and display progress'''
    print("==="+question_list_name+' is being processed')
    for question in tqdm(questions):
        question_list.append(text_to_wordlist(question))
            

def clean_dataset():
    
    print("===loading csv data")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print("===done loading csv data")
    
    # Check for any null values
    print('null values in train set:',train.isnull().sum())
    print('null values in train set:',test.isnull().sum())


    # Add the string 'empty' to empty strings
    train = train.fillna('empty')
    test = test.fillna('empty')

    # Preview some of the pairs of questions
    a = 0 
    print('===questions sample before cleaning')
    for i in range(a,a+3):
        print(train.question1[i])
        print(train.question2[i])
        print()
        
    is_duplicate = train.is_duplicate
        
    train_question1 = []
    process_questions(train_question1, train.question1, 'train_question1', train)



    train_question2 = []
    process_questions(train_question2, train.question2, 'train_question2', train)


    test_question1 = []
    process_questions(test_question1, test.question1, 'test_question1', test)


    test_question2 = []
    process_questions(test_question2, test.question2, 'test_question2', test)


    # Preview some transformed pairs of questions
    print('===questions sample after cleaning')
    a = 0 
    for i in range(a,a+3):
        print(train_question1[i])
        print(train_question2[i])
        print()
        
        
#     return train_question1,train_question2,test_question1,test_question2
    return train_question1,train_question2,test_question1,test_question2,is_duplicate


def tokenization(train_q1,train_q2,test_q1,test_q2):
    #进行切分词,然后得到word_to_index字典,以及换成index的句子数据集
    # make word index
    questions = train_q1 + train_q2 + test_q1 + test_q2
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)
    train_q1_word_sequences = tokenizer.texts_to_sequences(train_q1)
    train_q2_word_sequences = tokenizer.texts_to_sequences(train_q2)
    test_q1_word_sequences = tokenizer.texts_to_sequences(test_q1)
    test_q2_word_sequences = tokenizer.texts_to_sequences(test_q2)
    word_index = tokenizer.word_index
    print('words in index:', len(word_index))
    return word_index,train_q1_word_sequences,train_q2_word_sequences,test_q1_word_sequences,test_q2_word_sequences


def get_glove_embedding():
    #加载glove的embedding的look up table
    print("Processing", GLOVE_FILE)

    embeddings_index = {}
    with open(GLOVE_FILE, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings: %d' % len(embeddings_index))
    return embeddings_index


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



def pad_sequence(train_q1_seq,train_q2_seq,test_q1_seq,test_q2_seq,is_duplicate):
    #对每个以index标记的句子,填充pad到MAX_SEQUENCE_LENGTH
    train_q1_data = pad_sequences(train_q1_seq, maxlen=MAX_SEQUENCE_LENGTH)
    train_q2_data = pad_sequences(train_q2_seq, maxlen=MAX_SEQUENCE_LENGTH)
    test_q1_data = pad_sequences(test_q1_seq, maxlen=MAX_SEQUENCE_LENGTH)
    test_q2_data = pad_sequences(test_q2_seq, maxlen=MAX_SEQUENCE_LENGTH)
    
    labels = np.array(is_duplicate, dtype=int)
    print('Shape of train question1 data tensor:', train_q1_data.shape)
    print('Shape of train question2 data tensor:', train_q2_data.shape)
    print('Shape of test question1 data tensor:', test_q1_data.shape)
    print('Shape of test question2 data tensor:', test_q2_data.shape)
    print('Shape of label tensor:', labels.shape)
    return train_q1_data,train_q2_data,test_q1_data,test_q2_data,labels




def save_tmpfiles(train_q1_data,train_q2_data,test_q1_data,test_q2_data,labels,word_embedding_matrix,nb_words):
    #把所有文件先保存好
    np.save(open(Q1_TRAINING_DATA_FILE, 'wb'), train_q1_data)
    np.save(open(Q2_TRAINING_DATA_FILE, 'wb'), train_q2_data)
    np.save(open(Q1_TEST_DATA_FILE, 'wb'), test_q1_data)
    np.save(open(Q2_TEST_DATA_FILE, 'wb'), test_q2_data)
    np.save(open(LABEL_TRAINING_DATA_FILE, 'wb'), labels)
    np.save(open(WORD_EMBEDDING_MATRIX_FILE, 'wb'), word_embedding_matrix)
    with open(NB_WORDS_DATA_FILE, 'w') as f:
        json.dump({'nb_words': nb_words}, f)


if __name__ == '__main__':
    train_question1,train_question2,test_question1,test_question2,is_duplicate = clean_dataset()
    word_index,train_q1_seq,train_q2_seq,test_q1_seq,test_q2_seq = tokenization(train_question1,train_question2,test_question1,test_question2)
    embedding_index = get_glove_embedding()
    word_embedding_matrix,nb_words = get_word_embedding_matrix(word_index,embedding_index)
    train_q1_data,train_q2_data,test_q1_data,test_q2_data,labels = pad_sequence(train_q1_seq,train_q2_seq,test_q1_seq,test_q2_seq,is_duplicate)
    save_tmpfiles(train_q1_data,train_q2_data,test_q1_data,test_q2_data,labels,word_embedding_matrix,nb_words)

