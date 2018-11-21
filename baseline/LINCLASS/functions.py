import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import os
import pandas as pd
import numpy as np



def load_data(data_path, data_path_reddit, tags_path):
    
    #load twitter
    df = pd.DataFrame(columns = ['text'])
    folder_names = next(os.walk(data_path))[1]
    for name in folder_names:
        theme = data_path + name + '/'
        twit_names = next(os.walk(theme))[1]
        for twit in twit_names:
            repl_names = os.listdir(theme+twit+'/replies')
            for repl in repl_names:
                path = theme+twit+'/replies/'+repl
                df.loc[repl[:-5],'text'] = pd.read_json(path).loc['name','text']
    
    #load reddit
    folder_names = next(os.walk(data_path_reddit))[1]
    for name in folder_names:
        theme = data_path_reddit + name + '/'
        repl_names = os.listdir(theme + 'replies/')
        for repl in repl_names:
            path = theme + 'replies/'+repl
            if 'body' in pd.read_json(path).index:
                df.loc[repl[:-5],'text'] = pd.read_json(path).loc['body','data']
    
    #add tags
    tags = pd.read_json(tags_path).drop('subtaskbenglish',axis = 1)
    df_merged = df.merge(tags, left_index = True, right_index = True)
        
    
    
    return df_merged


def load_test_data(path_test, path_tag):
    
    test = pd.DataFrame(columns = ['text'])
    folder_names = next(os.walk(path_test))[1]
    for name in folder_names:
        theme = path_test + name + '/'
        repl_names = os.listdir(theme + 'replies/')
        for repl in repl_names:
            path = theme + 'replies/'+repl
            if 'body' in pd.read_json(path).index:
                test.loc[repl[:-5],'text'] = pd.read_json(path).loc['body','data']
    
    
    tags = pd.read_json(path_tag).drop('subtaskbenglish',axis = 1)
    df_merged = test.merge(tags, left_index = True, right_index = True)
    
    return df_merged



def clean_str(string):
    string = string.replace('#', ' ')
    string = " ".join(filter(lambda x:x[0]not in ['@','#','&'] and x[:4] != 'http', 
                             string.split()))
    string = string.replace('\\', ' ')
    string = string.replace('w/', ' ')
    string = string.replace('(', ' ')
    string = string.replace(')', ' ')
    string = string.replace('.', ' ')
    string = string.replace(',', ' ')
    string = string.replace(';', ' ')
    string = string.replace('-', ' ')
    string = string.replace("'s", ' ')
    string = string.replace(' 2 ', ' to ')
    string = string.replace(' 4 ', ' for ')
    string = string.replace(':', ' ')
    string = string.replace('/', ' ')
    string = re.sub("[^a-z,\s, A-Z,']", '', string)
    
    return string.lower()



def clean_text(df):
    df['clean_text'] = [None for i in df.index]    
    
    for i in df.index:
        df.loc[i,'clean_text'] = clean_str(df.loc[i,'text'])
    
    return df


def make_vocabulary(inp, n):
    bag = ' '.join(inp).split(' ')
    d = Counter(bag)
    vocabulary = {k: v for k, v in d.items() if v < n }
    return vocabulary


def tfidf(df, vocabulary):
    vectorizer = TfidfVectorizer(vocabulary = vocabulary.keys())
    X = vectorizer.fit_transform(df.loc[:,'text'])
    tfidf = pd.DataFrame(X.todense(), columns = vectorizer.get_feature_names(), 
                         index = [i for i in range(len(df.index))])
    return tfidf


def emb_sent(df, tfidf, emb, dim):
    nw = pd.DataFrame(columns = ['embed'])
    for ind in tfidf.index:
        em = np.zeros(dim)
        for col in tfidf.columns:
            if tfidf.loc[ind,col] != 0:
                if col in emb:
                    em = em + tfidf.loc[ind,col]*emb[col]
        nw.loc[ind,'embed'] = em
        
    df = df.merge(nw,right_index = True, left_on = 'id')
    return df
