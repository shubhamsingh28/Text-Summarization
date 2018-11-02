# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:55:42 2018

@author: Shubham
"""

import bs4 as bs  
import urllib.request  
import re
import nltk
import heapq
from nltk import PorterStemmer
import math
import rbm
import numpy as np
#nltk.download()

scraped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Golghar')  
article = scraped_data.read()

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""
par_arr=[];


for p in paragraphs:
    article_text += p.text
    par_arr.append(p.text)

article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)  
article_text = re.sub(r'\s+', ' ', article_text)
sentence_list = nltk.sent_tokenize(article_text)
article_text = re.sub('[^a-zA-Z]', ' ', article_text)  
article_text = re.sub(r'\s+', ' ', article_text)

ps=PorterStemmer()

stopwords = nltk.corpus.stopwords.words('english')

word_frequencies = {}  
formatted_sentence=''
for c_word, word in zip(nltk.word_tokenize(article_text), nltk.word_tokenize(article_text.lower())):  
    if word not in stopwords:
        formatted_sentence+= c_word
        formatted_sentence+=" "
        if ps.stem(word) not in word_frequencies.keys():
            word_frequencies[ps.stem(word)] = 1
        else:
            word_frequencies[ps.stem(word)] += 1

tag_word=nltk.pos_tag(nltk.word_tokenize(formatted_sentence))

thematic_words = heapq.nlargest(10, word_frequencies, key=word_frequencies.get)
sent={}
sentence_thematic=[]
for sentence in sentence_list:
    c=0
    for word in nltk.word_tokenize(sentence.lower()):
        if word in thematic_words:
            c=c+1
    tw=len(nltk.word_tokenize(sentence.lower()))
    sentence_thematic.append(c/tw)
    
sentence_position=[]
N=len(sentence_list)
th=N*0.2
min=th*N
max=th*2*N

for i in range(N):
    if i==0 or i==N-1:
        sentence_position.append(1)
    else:
        sentence_position.append(math.cos((i+1-min)/((1/max)-min)))
  
sentence_length=[]
for sentence in sentence_list:
    if len(nltk.word_tokenize(sentence))>=3:
        sentence_length.append(len(nltk.word_tokenize(sentence)))
    else:
        sentence_length.append(0)

'''position_in_para=[]
for sentence in sentence_list:
    for i in range(len(par_arr)):
        if sentence==nltk.sent_tokenize(par_arr[i])[0] or sentence==nltk.sent_tokenize(par_arr[i])[len(nltk.sent_tokenize(par_arr[i]))-1]:
            position_in_para.append(1)
        else:
            position_in_para.append(0)'''
   
sentence_proper_noun=[]
proper_noun= [w for w,pos in tag_word if pos=='NNP']
for sentence in sentence_list:
    c=0
    for word in nltk.word_tokenize(sentence):
        if word in proper_noun:
            c=c+1
    sentence_proper_noun.append(c)
            
sentence_numerals=[]
for sentence in sentence_list:
    c=0
    numbers=[int(word) for word in nltk.word_tokenize(sentence) if word.isdigit()]
    c=len(numbers)
    sentence_numerals.append(c/len(nltk.word_tokenize(sentence)))
#Tf-sdf
Tfsdf=[]
for sentence in sentence_list :
    score=0
    for words in nltk.word_tokenize(sentence):
        count_word=0
        count_word1=0
        for w in nltk.word_tokenize(sentence):
            if ps.stem(words) == ps.stem(w):
                count_word+=1
        for w1 in nltk.word_tokenize(formatted_sentence):
            if ps.stem(words) == ps.stem(w1):
                count_word1+=1
        score+=(count_word1*count_word)
    Tfsdf.append(math.log(score)/len(nltk.word_tokenize(sentence)))

featureMatrix = []
featureMatrix.append(sentence_thematic)
featureMatrix.append(sentence_position)
featureMatrix.append(sentence_length)
featureMatrix.append(sentence_proper_noun)
featureMatrix.append(sentence_numerals)
featureMatrix.append(Tfsdf)

featureMat = np.zeros((len(sentence_list),5))
for i in range(5) :
    for j in range(len(sentence_list)):
        featureMat[j][i] = featureMatrix[i][j]


featureMat_normed = featureMat

feature_sum = []

for i in range(len(np.sum(featureMat,axis=1))) :
    feature_sum.append(np.sum(featureMat,axis=1)[i])
    
temp = rbm.test_rbm(dataset = featureMat_normed,learning_rate=0.1, training_epochs=14, batch_size=5,n_chains=5,n_hidden=5)

enhanced_feature_sum = []
enhanced_feature_sum2 = []

for i in range(len(np.sum(temp,axis=1))) :
    enhanced_feature_sum.append([np.sum(temp,axis=1)[i],i])
    enhanced_feature_sum2.append(np.sum(temp,axis=1)[i])

enhanced_feature_sum.sort(key=lambda x: x[0])
length_to_be_extracted = len(enhanced_feature_sum)/2

for x in range(len(sentence_list)):
        print(sentence_list[x])

print("\n\n\nExtracted sentences : \n\n\n")
extracted_sentences = []
extracted_sentences.append([sentence_list[0], 0])

indeces_extracted = []
indeces_extracted.append(0)

for x in range(length_to_be_extracted) :
    if(enhanced_feature_sum[x][1] != 0) :
        extracted_sentences.append([sentence_list[enhanced_feature_sum[x][1]], enhanced_feature_sum[x][1]])
        indeces_extracted.append(enhanced_feature_sum[x][1])


extracted_sentences.sort(key=lambda x: x[1])

finalText = ""
print("\n\n\nExtracted Final Text : \n\n\n")
for i in range(len(extracted_sentences)):
    print("\n"+extracted_sentences[i][0])
    finalText = finalText + extracted_sentences[i][0]

print(finalText)



    





