'''
   this file contains class to handle lyrics, and torch Dataset,Dataloader
   last modify : 0ct,8,2oi9
'''

import torch
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence

class Song():
    '''class save name/lyrics info'''
    def __init__(self,name,lyrics):
        _,label,train,num = name.split("_")
        self.name = name
        self.is_train = True if train=='Train' else False
        self.label = label
        self.num = int(num)
        self.lyrics = lyrics
        
    def __repr__(self):
        return self.name
    

class BertSong(Song):
    '''Song + preprocess for BERT'''
    __label_mapping = {
        label:index for index,label in enumerate(['Angry','Happy','Relaxed','Sad'])
        }
    
    def __init__(self,name,lyrics,tokenizer):
        super().__init__(name,lyrics)
        self._preprocess(tokenizer)
        
    def _preprocess(self,tokenizer):
        new_lyrics = self._remove_repeated()
        self.tokens = tokenizer.encode(new_lyrics,add_special_tokens=True)
        ### remove tokens when num of tokens >512
        if len(self)>512:
            self.tokens = self.tokens[:510]
            self.tokens.append(102)
        ###
        self.segments = [0]*len(self)
        self.label = BertSong.__label_mapping[self.label]
    
    def _remove_repeated(self)->str:
        'remove the repeated lines to shorten lyrics,return shortened string'
        lines = self.lyrics.split("\n")
        new_lines = []
        line_set = set()
        for line in lines:
            if line not in line_set:
                new_lines.append(line)
                line_set.add(line)
        return '\n'.join(new_lines)
    
    def __len__(self):
        return len(self.tokens)
    
        
class LyricsDataset(Dataset):
    '''torch dataset'''
    def __init__(self,songlist,is_train:bool):
        '''
        param:
        ---
        songlist: list of BertSong
        is_train: bool specify whether train or test
        '''
        self.songlist = songlist
        self.is_train = is_train
    
    def __getitem__(self,index):
        item = self.songlist[index]
        return torch.tensor(item.tokens),torch.tensor(item.segments),torch.tensor(item.label)
        
    def __len__(self):
        return len(self.songlist)


def my_collate(batch):
    'collate function for DataLoader'
    tokens_tensors = [i[0] for i in batch]
    segments_tensors = [i[1] for i in batch]
    label_tensors = torch.stack([i[2] for i in batch])
    # zero padding
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)    
    return tokens_tensors, segments_tensors, masks_tensors, label_tensors

