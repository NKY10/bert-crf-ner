from transformers import BertModel
from torchcrf import CRF
import torch.nn as nn
import torch

class BERTCRF(nn.Module):
    def __init__(self, num_labels, bert_path = 'bert-base-chinese' ,drop_out_rate=0.1):
        super(BERTCRF, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(drop_out_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.bool())
            return prediction
        
    


