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
import numpy as np
#nltk.download()

scraped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/National_Institute_of_Technology,_Patna')  
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
print('1st feature=thematic words')
thematic_words = heapq.nlargest(10, word_frequencies, key=word_frequencies.get)
print(thematic_words)
sent={}
sentence_thematic=[]
for sentence in sentence_list:
    c=0
    for word in nltk.word_tokenize(sentence.lower()):
        if word in thematic_words:
            c=c+1
    tw=len(nltk.word_tokenize(sentence.lower()))
    sentence_thematic.append(c/tw)
print(sentence_thematic)
print("2nd feature=sentence position")
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
print(sentence_position)
print('3rd feature:sentence length')  
sentence_length=[]
for sentence in sentence_list:
    if len(nltk.word_tokenize(sentence))>=3:
        sentence_length.append(len(nltk.word_tokenize(sentence)))
    else:
        sentence_length.append(0)
print(sentence_length)
'''position_in_para=[]
for sentence in sentence_list:
    for i in range(len(par_arr)):
        if sentence==nltk.sent_tokenize(par_arr[i])[0] or sentence==nltk.sent_tokenize(par_arr[i])[len(nltk.sent_tokenize(par_arr[i]))-1]:
            position_in_para.append(1)
        else:
            position_in_para.append(0)'''
print('4th feature: proper noun')
sentence_proper_noun=[]
proper_noun= [w for w,pos in tag_word if pos=='NNP']
for sentence in sentence_list:
    c=0
    for word in nltk.word_tokenize(sentence):
        if word in proper_noun:
            c=c+1
    sentence_proper_noun.append(c)
print(sentence_proper_noun)
print('5th feature : sentence numeral')            
sentence_numerals=[]
for sentence in sentence_list:
    c=0
    numbers=[int(word) for word in nltk.word_tokenize(sentence) if word.isdigit()]
    c=len(numbers)
    sentence_numerals.append(c/len(nltk.word_tokenize(sentence)))
print(sentence_numerals)
#Tf-sdf
print('6th feature=Tfsdf')
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
    if score !=0:
        Tfsdf.append(math.log(score)/len(nltk.word_tokenize(sentence)))
    else:
        Tfsdf.append(0)
print(Tfsdf)
featureMatrix = []
featureMatrix.append(sentence_thematic)
featureMatrix.append(sentence_position)
featureMatrix.append(sentence_length)
featureMatrix.append(sentence_proper_noun)
featureMatrix.append(sentence_numerals)
featureMatrix.append(Tfsdf)
featureMat = np.zeros((len(sentence_list),6))
for i in range(6) :
    for j in range(len(sentence_list)):
        featureMat[j][i] = featureMatrix[i][j]

sentence_scores = {}  
for sent in sentence_list:
    sum=0
    for j in range(0,6):
        sum+=sentence_thematic[j]
        sum+=sentence_position[j]
        sum+=sentence_length[j]
        sum+=sentence_proper_noun[j]
        sum+=sentence_numerals[j]
        sum+=Tfsdf[j]
    sentence_scores[sent]=sum
                    

summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)  

print(summary)



    





