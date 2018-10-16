'''
data processing before cnn is fitted.
Beware your RAM useage if your RAM is less than 8G.
'''
import os
import re
import itertools
import pickle
from collections import Counter,defaultdict
try:
	import pandas as pd
	import numpy as np
	from nltk.corpus import stopwords
	from nltk.stem import PorterStemmer
	import gensim
	from gensim.models import Word2Vec
	from nltk.stem import WordNetLemmatizer
	from keras.utils import np_utils
except:
	print("require modules: keras,gensim,nltk.stem,nltk.corpus,nltk.stem, please install it.")
	exit()

def read_song():
	'''
	read all txt lyrics file
	output is a dic with file name as key, lyrics as values
	'''
	reg1 = re.compile("\.txt$")
	reg2 = re.compile("([0-9]+)\.txt")
	reg3 = re.compile(".*_([0-9])\.txt")
	reg4 = re.compile("\[.+\]")
	reg5 = re.compile("info\.txt")
	lyrics_dic = {}
	#iter all directory and load all song(txt file)
	for i in os.listdir():
	    if os.path.isdir(i):
	        for path,sub,items in os.walk(i):
	            if any([reg1.findall(item) for item in items]):
	                for item in items:
	                    if reg5.findall(item):
	                        continue
	                    if reg3.findall(item):
	                        num = ["0"+reg3.findall(item)[0]]
	                        name = "_".join(path.split("/") + num)
	                    else:
	                        name = "_".join(path.split("/") + reg2.findall(item))
	                    
	                    with open(os.path.join(path,item),"r") as f:
	                        lyrics = "".join(f.readlines())
	                        lyrics = reg4.subn("",lyrics)[0]
	                        lyrics_dic[name] = lyrics
	return lyrics_dic

def lyric_2w(lyrics_dic,balance=True): 
    '''
	change lyrics to word-sequence, where word is represented as integer
	to identity word:id, id:w2v vectors, the word_id,id_vec dic is returned.
	wordcount dict is also return if you want to filter words that less occured.	
    '''
    
    train_dic = {}
    test_dic = {}
    wordcount = defaultdict(int)
    wordset = set()
    word_id = dict()
    id_vec = dict()
    #catch words not in word2vec
    no_vec_set = set()
    punctuation = re.compile("[\[\.\:\,\]\?\\\/\~!\)#\(\*’\|;\"\$\%\^\&\*\_\-]|＇t|＇ll|\'[a-z]+")
    end_with_in = re.compile(".+in$")
    test = re.compile("Test")
    wordnet_lemmatizer = WordNetLemmatizer()
    w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #iter over all songs:
    for title,song in lyrics_dic.items():        
        lyrics_seq = []
        #iter over one song's words:
        for word in punctuation.subn("",song)[0].split():
            word = wordnet_lemmatizer.lemmatize(word.lower())
            if word not in wordset:
                try:
                    cur = w2v.get_vector(word)  
                    wordset.add(word)
                    wordcount[word] += 1
                    word_id[word] = len(word_id)+1
                    id_vec[word_id[word]] = cur 
                    lyrics_seq.append(word_id[word])                    
                except KeyError as e:
                    #if word is end with 'in', add g as 'ing'
                    if end_with_in.findall(word):
                        word = word +"g"
                    try:
                        cur = w2v.get_vector(word)
                        wordset.add(word)
                        wordcount[word] += 1
                        word_id[word] = len(word_id)+1
                        id_vec[word_id[word]] = cur
                        lyrics_seq.append(word_id[word])                                            
                    except KeyError as e:
                        no_vec_set.add(word)                              
            else:
                wordcount[word] += 1
                lyrics_seq.append(word_id[word])
        if test.findall(title):
            test_dic[title] = lyrics_seq
        else:
            train_dic[title] = lyrics_seq

    if balance:
        ly_max = max(max([len(val) for val in train_dic.values()]),max([len(val) for val in test_dic.values()]))
        for key,value in train_dic.items():    
            zero = [0]*(ly_max-len(value))
            train_dic[key] = value+zero
        for key,value in test_dic.items():
            zero = [0]*(ly_max-len(value))
            test_dic[key] = value+zero
    return (train_dic,test_dic,wordcount,word_id,id_vec,no_vec_set)

'''
replaced by lyric_2W, DONOT USE IT
'''
def lyric_2v(lyrics_dic,balance=True):
	'''
	convert the lyricss to w2v vector tensor and 
	split it into train and test set(3 dim)
	if balance=True, all lyrics metrics will be same dim(fill with 0)
	output is
	(train_dic,test_dic,no_vec_set)	
	no_vec_set is the words set that doesnt include in GoogleNews pre-trained model.
	'''
	train_dic = {}
	test_dic = {}
	#catch words not in word2vec
	no_vec_set = set()
	punctuation = re.compile("[\[\.\:\,\]\?\\\/\~!\)#\(\*’\|;\"\'\$\%\^\&\*\_\-]|＇t|＇ll")
	end_with_in = re.compile(".+in$")
	test = re.compile("Test")		
	wordnet_lemmatizer = WordNetLemmatizer()
	w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
	#iter over all songs:
	for title,song in lyrics_dic.items():
	    if test.findall(title):
	        lyricsvec = []
	        #iter over one song's words:
	        for word in punctuation.subn("",song)[0].split():
	            try:
	                word = wordnet_lemmatizer.lemmatize(word)
	                cur = w2v.get_vector(word)        
	                lyricsvec.append(cur)
	            except KeyError as e:
	                #if word is end with in, add g as ing
	                if end_with_in.findall(word):
	                    word = word +"g"
	                try:
	                    cur = w2v.get_vector(word)
	                    lyricsvec.append(cur)
	                except KeyError as e:
	                    no_vec_set.add(word)
	            test_dic[title] = np.array(lyricsvec)
	    else:
	        lyricsvec = []
	        #iter over one song's words:
	        for word in punctuation.subn("",song)[0].split():
	            try:
	                word = wordnet_lemmatizer.lemmatize(word)
	                cur = w2v.get_vector(word)        
	                lyricsvec.append(cur)
	            except KeyError as e:
	                #if word is end with in, add g as ing
	                if end_with_in.findall(word):
	                    word = word +"g"
	                try:
	                    cur = w2v.get_vector(word)
	                    lyricsvec.append(cur)
	                except KeyError as e:
	                    no_vec_set.add(word)
	            train_dic[title] = np.array(lyricsvec) 
	if balance:
		ly_max = max(max([val.shape[0] for val in train_dic.values()]),max([val.shape[0] for val in test_dic.values()]))
		for key,value in train_dic.items():    
		    zero = np.zeros((ly_max-value.shape[0],300))
		    train_dic[key] = np.concatenate([value,zero])

		for key,value in test_dic.items():    
		    zero = np.zeros((ly_max-value.shape[0],300))
		    test_dic[key] = np.concatenate([value,zero])	
	return (train_dic,test_dic,no_vec_set)	

def get_train_test_set(train_dic,test_dic):
	# get label by dic's key,return one-hot-key format y-label	
	x_train = np.array(list(train_dic.values()))
	x_test = np.array(list(test_dic.values()))
	y_train = []
	y_test = []
	for title in train_dic.keys():
	    y_train.append(title.split("_")[1])

	for title in test_dic.keys():
	    y_test.append(title.split("_")[1])

	labels = set(y_train)
	label_mapping = {label:no for no,label in enumerate(labels)}
	print(label_mapping)
	for i in range(len(y_train)):
	    y_train[i] = label_mapping[y_train[i]]
	    
	for i in range(len(y_test)):
	    y_test[i] = label_mapping[y_test[i]]
	    
	y_train = np_utils.to_categorical(y_train, 4)
	y_test = np_utils.to_categorical(y_test, 4)
	return (x_train,x_test,y_train,y_test)

def save_pickles(x_train,x_test,y_train,y_test,wordcount,word_id,id_vec,no_vec_set):
	'''
	save objects in pickle format
	'''	
	if not os.path.isdir("pkls"):
		os.mkdir("pkls")
	for name in ["x_train","x_test","y_train","y_test","wordcount","word_id","id_vec","no_vec_set"]:    
	    path = os.path.join("pkls",name+".pkl")
	    output = open(path, 'wb')
	    pickle.dump(eval(name), output)
	    output.close()
	return 0

if __name__ == '__main__':
	print("...")
	print("Now reading songs...")
	lyrics_dic = read_song()
	print("...")	
	print("Now converting lyrics to sequence...")
	train_dic,test_dic,wordcount,word_id,id_vec,no_vec_set = lyric_2w(lyrics_dic)
	print("...")
	print("Getting train test sets...")
	x_train,x_test,y_train,y_test = get_train_test_set(train_dic,test_dic)
	print("...")
	print("Saving to pickles...")
	save_pickles(x_train,x_test,y_train,y_test,wordcount,word_id,id_vec,no_vec_set)	
	print("completed!")	


