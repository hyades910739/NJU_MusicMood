import os
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from cnn_data_process import read_song

def clean_lyrics(lyrics,stop,abbr,symbol,n_remove=2):
    lyrics = lyrics.split()
    lyrics = [abbr.sub("",word.lower()) for word in lyrics]
    lyrics = [symbol.subn("",word)[0] for word in lyrics]
    lyrics = [word for word in lyrics if word not in stop and len(word)>2] 
    return(lyrics)

def main():
	#read words
	lyrics_dic = read_song()
	#clean words
	stop = set(stopwords.words('english'))
	porter_stemmer = PorterStemmer()
	abbr = re.compile("[n't]|['re]|['s']")
	symbol = re.compile("[?!,.\"\'\(\-\)\*\+/]|^\(|\)^|（|）|－|[0-9<>]|[^a-zA-Z]")
	word_dic = {}
	for k,v in lyrics_dic.items():
	    words = clean_lyrics(v,stop=stop,symbol=symbol,abbr=abbr)
	    word_dic[k] = Counter(words)
	#vsm model
	vsm = pd.DataFrame(word_dic).transpose()
	vsm = vsm.iloc[:,np.where(vsm.sum(0)<3)[0]]
	vsm = vsm.fillna(0)

	# create y df
	df_label = pd.DataFrame(index=vsm.index,columns=["label","split"])
	re_test = re.compile("Test")
	re_happy = re.compile("Happy")
	re_angry = re.compile("Angry")
	re_relax =  re.compile("Relaxed")
	re_sad =  re.compile("Sad")
	df_label.label = [df_label.index[i].split("_")[1] for i in range(len(df_label))]
	df_label.split = [df_label.index[i].split("_")[2] for i in range(len(df_label))]

	#TFIDF transform	
	transformer = TfidfTransformer(smooth_idf=False)
	tfidf = transformer.fit_transform(vsm)
	tfidf = pd.DataFrame(tfidf.toarray(),index=vsm.index,columns=vsm.columns)

	#split
	train_x = tfidf.loc[df_label.split=="Train",:]
	train_y = df_label.label.loc[df_label.split=="Train"]
	test_x = tfidf.loc[df_label.split=="Test",:]
	test_y = df_label.label.loc[df_label.split=="Test"]

	#SVM
	clf = LinearSVC()
	clf.fit(train_x, train_y)
	print("train acc : {}".format(clf.score(train_x,train_y)))
	print("test acc : {}".format(clf.score(test_x,test_y)))


if __name__ == '__main__':
	main()