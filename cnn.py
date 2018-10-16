import pickle
import numpy as np
try:
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, Activation
    from keras.layers.merge import Concatenate
    from keras.datasets import imdb
    from keras.layers import Embedding
    from keras.optimizers import SGD
    from keras.preprocessing import sequence
    from keras import regularizers
except:
    print("Require numpy,keras and tensorflow, install it.")

def cnn():
    #tuning parameters:
    trainable = True
    filter_sizes = [5,10,15,30]
    filters = 200
    pool_size = 1
    epochs = 30
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
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             use_bias=True,
                             strides=1)(embedded_sequences)
        conv = MaxPooling1D(pool_size=pool_size)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) 
    z = Dropout(0.5)(z)
    z = Dense(50, activation="relu")(z)
    z = Dropout(0.5)(z)
    z = Dense(40, activation="relu")(z)          
    model_output = Dense(4, activation="softmax",kernel_regularizer=regularizers.l2(0.001))(z)
    model = Model(model_input, model_output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['binary_accuracy', 'categorical_accuracy'])
    model.summary()
    model.fit(data["x_train"], data["y_train"], batch_size=50, epochs=epochs)   
    pred = np.array([np.argmax(i) for i in model.predict(data['x_test'])])
    y_true = np.array([np.argmax(i) for i in data['y_test']])           
    print("Testing Accuracy : {}".format(np.mean(pred == y_true)))

if __name__ == '__main__':
    cnn()
