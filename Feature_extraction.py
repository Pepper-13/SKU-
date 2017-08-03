#Assumption of getting data to this format


import os #can be used for python eggs
import pandas as pd
import ast
import re
import numpy as np
import sys
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

def model(inputdata):
    df = pd.DataFrame(inputdata, columns = ['class_id','dept_id','item_id','item_descrp','item_flag'])
    df = df.reset_index()
    old_items = df[df['item_flag'].str.strp()=='OLD'].item_i.tolist()
    new_items = df[df['item_flag'].str.strp()=='NEW'].item_i.tolist()
    old_ids =  df[df['item_flag'].str.strp()=='OLD'].index.tolist()
    new_ids = df[df['item_flag'].str.strp()=='NEW'].index.tolist()
    
    def token(text):
        a = text.split(":")
        return a 
    
    vec = TfidfVectorizer(ngram_range = (1,1), stop_words = 'english', tokenizer = token)
    tfid_mat = vec.fit_transform(df.item_descrp)
    feature_mat = tfid_mat.toarray()
    
    feature_mat_new = feature_mat[new_ids]
    feature_mat_old = feature_mat[old_ids]
    
    ans =[]
    
    for i in range(0,len(feature_mat_new)):
        ans1 =[]
        for j in range(0,len(feature_mat_old)):
            ans1[old_items[j]] = (cosine(feature_mat_new[i], feature_mat_old[j]))
            ans.append(ans1)
            
        ans1=[]
        for i in ans:
            ans1.append(min(i,key=i.get))
            
        ans2= zip(new_items,ans1)
        
        for it in ans2:
            print ('\t'.join(['%s' %(i,) for i in it]))
            
        ids = None
        itemdata =[]
        counter = 0
        
        for line in sys.stdin:
            if len(line.strip()) > 0:
                if counter >0:
                    if line.split('\t')[0] != ids:
                        model(itemdata)
                        itemdata = []
                    ids = line.split('\t')[0]
                    itemdata.append([line.split('\t')[0]],(line.split('\t')[1]))
                    counter +=1
                    
                    
        if len(itemdata)>0:
            model(itemdata)
