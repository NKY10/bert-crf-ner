from torch.utils.data import Dataset, DataLoader
import torch
import json
import os
# 加载 BERT

class NERDataset(Dataset):
    def __init__(self, file_path, max_len=512, tokenizer=None ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load label map if it exists
        # if label_map_path is not None and os.path.exists(label_map_path):
        #     with open(label_map_path, 'r') as f:
        #         self.label_map = json.load(f)
        # else:
        self.label_map = {'O': 0}  # Assume 'O' is always present
        self.sentences = []
        self.labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence = []
            label = []
            for line in f:
                if line == "\n":
                    if sentence:
                        self.sentences.append(sentence)
                        self.labels.append(label)
                        sentence = []
                        label = []
                else:

                    #print(line.strip().split())
                    #print(line,line=="",line==" ",line=="\n")
                    token, tag = line.strip().split()
                    sentence.append(token)
                    label.append(tag)
                    if tag not in self.label_map:
                        self.label_map[tag] = len(self.label_map)
                        
            # Save the label map
            # if label_map_path is not None:
            #     with open(label_map_path, 'w') as f:
            #         json.dump(self.label_map, f)
                    
            # Add the last sentence and labels if file doesn't end with newline
            if sentence:
                self.sentences.append(sentence)
                self.labels.append(label)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        inputs = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,    # 是否添加cls和sep
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()   
        label_ids = [self.label_map[l] for l in label]
        label_ids = [self.label_map[l] for l in label]
        if len(label_ids) > self.max_len - 2:
            label_ids = label_ids[:self.max_len-2]  # 截断至最大长度，不含 [CLS] 和 [SEP]
        label_ids = [0] + label_ids
        label_ids = label_ids + ( [0] * (self.max_len - len(label_ids)))  # 填充
        label_ids = torch.tensor(label_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids,
            #'sentence':sentence,
        }
