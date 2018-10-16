import pickle
import os
import numpy as np
try:
    from keras.preprocessing import sequence
    from keras import regularizers,optimizers
    from keras.models import Sequential, Model
    from keras.layers import Dense, Embedding,LSTM,Input,Bidirectional,Dropout,Flatten
except:
    print("Require numpy,keras and tensorflow, install it.")

def rnn():
    #tuning parameters:
    return_sequences = False
    trainable = True
    epochs = 20
    #
    print("loading data...")
    data = {}
    for name in ["x_train","x_test","y_train","y_test",
                 "wordcount","word_id","id_vec","no_vec_set"]:
        path = os.path.join("pkls",name+".pkl")
        with open(path,"rb") as f:
            try:
                data[name] = pickle.load(f)
            except:
                print("failed at loading data:{}".format(name))
                exit()

    w2v_weight = np.array(list(data['id_vec'].values()))
    ## add zeros in first dim:
    zero = np.zeros((1,w2v_weight.shape[1]))
    w2v_weight = np.concatenate((zero,w2v_weight),axis=0) 
    embedding_layer = Embedding(len(data["word_id"]) + 1,
                                300,
                                weights= [w2v_weight],
                                input_length=data['x_test'].shape[1],
                                trainable=trainable)

    model_input = Input(shape=(data["x_train"].shape[1],),dtype='int32')
    embedded_sequences = embedding_layer(model_input)
    z = Bidirectional(LSTM(150,activation="sigmoid", return_sequences=return_sequences, dropout_U = 0.2, dropout_W = 0.2))(embedded_sequences)        
    if return_sequences:
        z = Flatten()(z)
    z = Dropout(0.5)(z)
    
    model_output = Dense(4, activation="softmax")(z)
    model = Model(model_input, model_output)
    adam = optimizers.Adam(lr=0.01)

    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['binary_accuracy', 'categorical_accuracy'])
    model.summary()
    model.fit(data["x_train"], data["y_train"], batch_size=50, epochs=epochs)   
    pred = np.array([np.argmax(i) for i in model.predict(data['x_test'])])
    y_true = np.array([np.argmax(i) for i in data['y_test']])           
    print("Testing Accuracy : {}".format(np.mean(pred == y_true)))

if __name__ == '__main__':
    rnn()
