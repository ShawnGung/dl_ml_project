import os
from config import *
import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer


class Data_Loader():
    def __init__(self):
        file = os.path.join(DATA_DIR,ALL_DATA_FILE)
        self.dataframe = pd.DataFrame.from_csv(file, sep='\t', header=0)
        self.contract_map = {"ain't": "is not", "aren't": "are not", "arent": "are not", "can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "dont": "do not", "hadn't": "had not", "hadnt": "had not",
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have",  
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", "shouldvetaken": "should have taken", 
                   "this's": "this is", "doesnt": "does not",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have", "youre": "you are"}



        nltk.download('punkt')
        nltk.download('stopwords')
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords += ['url', 'http', 'rt', 'a', 'youre', 'your', 're', 'you']
        self.stopwords.append('@USER')
        self.stopwords.append('#')


    def get_data(self):
        self.dataframe['cleaned_tweet'] = self.dataframe.tweet.apply(self.clean_sentence)
        self.dataframe['stemmed_tweet'] = self.dataframe.cleaned_tweet.apply(self.stemming)
        self.dataframe['cleaned_s'] = self.dataframe.cleaned_tweet.map(lambda s: ' '.join(s))
        return self.dataframe
    
    
    def save_csv(self):
        self.dataframe.to_csv(os.path.join(DATA_DIR,PROCESSED_DATA_FILE), index=None)


    def expand_contractions(self,sentence, contraction_mapping): 

        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),  
                                          flags=re.IGNORECASE|re.DOTALL) 
        def expand_match(contraction): 
            match = contraction.group(0) 
            first_char = match[0] 
            expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())                        
            expanded_contraction = first_char+expanded_contraction[1:] 
            return expanded_contraction 

        expanded_sentence = contractions_pattern.sub(expand_match, sentence) 
        return expanded_sentence 


    def clean_sentence(self,s):
        s = self.expand_contractions(s, self.contract_map)
        s = s.strip()
        s = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", s)
        tokens = nltk.word_tokenize(s)
        return [token.lower() for token in tokens if token.isalpha() and token.lower() not in self.stopwords]

    def stemming(self,s):
        ps = PorterStemmer()
        return [ps.stem(word) for word in s]