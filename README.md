# 歌詞情緒分析
last update : Oct.16.2018
### Dataset :
NJU_MusicMood V1.0, contains 777 songs with lyrics, and each song comes with one label, representing its sentiment : angry, happy, relaxed, or sad. Please check `data/NJU-MusicMood-v1.0.htm` for more information. 

Reference : Multimodal Music Mood Classification by Fusion of Audio and Lyrics.  

### files and usage :
+ `data/` : Original datasets.
+ `pkls/` : Processed data (in pickle format) to use in CNN and RNN. Check `cnn_data_process.py`
+ To use pre-trained GoogleNews word2vec model, please download from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).


### Method : 
<table>
　<tr>
　	<td>Method</td>
	<td>Description</td>
　	<td>Train_Acc</td>
　	<td>Test_Acc</td>
　</tr>
　<tr>
　	<td>TFIDF</td>
　	<td>Regular bag-of-word TFIDF with SVM<br>
		P.S. SVD DOESNOT help 
	</td>
　	<td></td>
　	<td></td>
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
　	<td>RNN(LSTM)</td>
　	<td>Fucked up now.
	</td>
　	<td></td>
　	<td></td>
　</tr>
</table>
