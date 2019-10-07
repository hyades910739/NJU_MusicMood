# 歌詞情緒分析
last update : Oct.8.2019

### Abstract : 
This repo implement some lyrics sentiment classification method, including TFIDF-based, CNN, RNN, and BERT classifier.

### Dataset :
NJU_MusicMood V1.0, contains 777 songs with lyrics, and each song comes with one label, representing its sentiment : angry, happy, relaxed, or sad. Please check `data/NJU-MusicMood-v1.0.htm` for more information. 

Reference : Multimodal Music Mood Classification by Fusion of Audio and Lyrics.  

### files and usage :
+ `data/` : Original datasets.
+ `pkls/` : Processed data (in pickle format) to use in CNN and RNN. Check `cnn_data_process.py`
+ To use pre-trained GoogleNews word2vec model, please download from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).


### Method : 

* new: BERT classifier(see `bert_utli.py ` and `train_bert.py`), now training and tuning...

<table>
　<tr>
　	<td>Method</td>
	<td>Description</td>
　	<td>Train_Acc</td>
　	<td>Test_Acc</td>
　</tr>
　<tr>
　	<td>TFIDF(Baseline)</td>
　	<td>Regular bag-of-word TFIDF with SVM and Random Forest<br>
		P.S. SVD DOESNOT help 
	</td>
　	<td>SVM:0.8625<br>RF :0.8625</td>
　	<td>SVM:0.292<br>RF :0.284</td>
　</tr>

　<tr>
　	<td>tfidf based classifer</td>
　	<td>Ref:
        <ul>
            <li>AUTOMATIC MOOD CLASSIFICATIONUSING TF*IDF BASED ON LYRICS
        </ul>
    </td>
　	<td>0.845</td>
　	<td>0.297</td>
　</tr>

　<tr>
　	<td>Convolutional Neural Network</td>
　	<td>Ref:
		<ul>
    		<li>Convolutional Neural Networks for Sentence Classification</li>
    		<li>A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification</li>
   		</ul>
	</td>
　	<td>0.95-0.98</td>
　	<td>0.40-0.47</td>
　</tr>

　<tr>
　	<td>RNN(bidirectional LSTM)</td>
　	<td>Fucked up.
	</td>
　	<td>over 0.95</td>
　	<td>about 0.3</td>
　</tr>
</table>
