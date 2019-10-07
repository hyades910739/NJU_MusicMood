from cnn_data_process import *
from bert_utli import *
import transformers
import time
import torch.nn as nn
import torch

BATCH_SIZE = 8
LR = 0.03
NUM_EPOCH = 15

def init_model():
    config = transformers.BertConfig.from_pretrained('bert-base-uncased',num_labels=4,hidden_size = 768)  
    model  = transformers.BertForSequenceClassification(config)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    return config,model,tokenizer

def get_data():
    lyrics_dic = read_song()
    train_dataset = LyricsDataset(
        [BertSong(k,v,tokenizer) for k,v in lyrics_dic.items() if 'Train' in k],True
        )
    test_dataset = LyricsDataset(
        [BertSong(k,v,tokenizer) for k,v in lyrics_dic.items() if 'Train' in k],False
        ) 
    trainloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, 
        collate_fn=my_collate,shuffle=True
        )   
    testloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, 
        collate_fn=my_collate,shuffle=False
        )
    return train_dataset,test_dataset,trainloader,testloader


if __name__ == '__main__':
    _,model,tokenizer = init_model()

    train_dataset,test_dataset,trainloader,testloader = get_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # training
    model.train()
    for epoch in range(NUM_EPOCH):
        epoch_loss = 0
        total = 0
        tps = 0 #num of true postive        
        init_time = time.time()
        for no,batch in enumerate(trainloader):
            res = model(batch[0],batch[1],batch[2])
            loss = criterion(res[0], batch[3])
            epoch_loss += loss.item()

            #acc
            val,indice = torch.max(res[0],dim=1)
            total += len(indice)
            tps += (indice==batch[3]).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch [{:2}/{}] ,Time:{} sec ,Loss: {:.4f}, acc: {:.3f}'.format(
              epoch+1,NUM_EPOCH,int(time.time()-init_time),epoch_loss,tps/total)
             )
    # testing:
    print('now testing')
    model.eval()
    total = 0
    tps = 0 #num of true postive
    with torch.no_grad():   
        for batch in testloader:
            res = model(batch[0],batch[1],batch[2])
            val,indice = torch.max(res[0],dim=1)
            total += len(indice)
            tps += (indice==batch[3]).sum().item()

    print('test result:\n acc: {}/{}'.format(tps,total))

        

