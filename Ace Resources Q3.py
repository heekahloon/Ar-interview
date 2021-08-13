# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 19:21:20 2021

@author: Klun
"""

from nltk import FreqDist
import pandas as pd
import seaborn as sns


paragraph=open('paragraph.txt').read()

###############################################################################

#Q3a

#Let the paragraph become lists of each line
lines=paragraph.split(' \n')

#Removing symbols from each line
symbol=['?','!',',','.','--','(',')',':']
for i in range(len(lines)):
    for s in symbol:
        lines[i]=lines[i].replace(s,'')

#Setting every words into lowercase
for i in range(len(lines)):
    lines[i]=lines[i].lower()       
        
#Getting the list of words in each line
wordinline=[]
for line in lines:
    wordinline.append(line.split(' '))

#Double check if there is any spacebar or blank str
pre=[]
for i in wordinline:
    pre.append(len(i))
    
for i in range(len(wordinline)):
    if '' in wordinline[i]:
        wordinline[i].remove('')
    if ' ' in wordinline[i]:
        wordinline[i].remove(' ')
    
for i in range(len(wordinline)):
    if len(wordinline[i]) != pre[i]:
        print("From row {}, we removed {} blank or spacebar string(s).".format(i,pre[i]-len(wordinline[i])))
    
    
freq=[]
for i in range(len(wordinline)):
    freq.append(FreqDist(wordinline[i]))

#Here we get the probability of  word "data" occuring in each of the lines
datap_inline=[freq[i]['data']/len(wordinline[i]) for i in range(len(wordinline))]

###############################################################################

#Q3b

#Removing symbols from each line and also the '\n' which is "Enter"
symbol=['?','!',',','.','--','(',')',':']
b=paragraph
for s in symbol:
    b=b.replace(s,'')
b=b.replace('\n','')

#Setting the words to lowercase    
b=b.lower()

#Making the paragraph in to list of the words
b=b.split(' ')

#Remove any extra blanks
while '' in b:
    b.remove('')

#Getting the ten most frequently used words
freqb10=FreqDist(b).most_common(10)
freqb10 = pd.Series(dict(freqb10))

#Plotting the distribution of the 10 words into a graph
sns.barplot(x=freqb10.index, y=freqb10.values)

#Here is the version of all words but very hard to see
freqb=FreqDist(b)
freqb=pd.Series(dict(freqb))
sns.barplot(x=freqb.index, y=freqb.values)




###############################################################################

#3c

from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, ELEProbDist

#Making list of tuples with the format of (word A, preceeding word after the word A)
pre=None
l=[]
for word in b:
    outcome=word
    l.append(tuple([pre,outcome]))
    pre=word

#Fitting the list of tuples into the the frequency distribution function
cfdist = ConditionalFreqDist(l)

#this is will tell us the probability distribution
cpdist = ConditionalProbDist(cfdist, ELEProbDist, 11)

#Here is the probability of the word "analytics" occuring after word "data"
cpdist['data'].prob('analytics')



