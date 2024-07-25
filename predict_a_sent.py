import torch
from transformers import BertTokenizer
from modules.BERTCRF import BERTCRF
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = BERTCRF(17).to(device)
net.eval()
# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# Load trained model parameters
net.load_state_dict(torch.load("./model/model.parameters"))
label_map = {'O': 0, 'B-PER.NOM': 1, 'I-PER.NOM': 2, 'B-LOC.NAM': 3, 'I-LOC.NAM': 4, 'B-PER.NAM': 5, 'I-PER.NAM': 6, 'B-GPE.NAM': 7, 'I-GPE.NAM': 8, 'B-ORG.NAM': 9, 'I-ORG.NAM': 10, 'B-ORG.NOM': 11, 'I-ORG.NOM': 12, 'B-LOC.NOM': 13, 'I-LOC.NOM': 14, 'B-GPE.NOM': 15, 'I-GPE.NOM': 16}
label_map = {value: key for key, value in label_map.items()}
sent= "李云龙去打太原了"
inputs = tokenizer.encode_plus(
    sent,
    add_special_tokens=True,    # 是否添加cls和sep
    max_length=150,
    padding='max_length',
    truncation=True,
    return_token_type_ids=False,
    return_attention_mask=True,
    return_tensors='pt'
)
res = net(inputs["input_ids"].to(device),inputs["attention_mask"].to(device))[0][1:-1]
i,flag = 0,0
while i < len(sent):
    out = {"label":"","span":""}
    while res[i]!=0:
        flag = 1
        out["span"]+=sent[i]
        out["label"] = label_map[res[i]].replace("B-","").replace("I-","")
        i += 1
    if flag == 1:
        print(out)
        flag = 0
    i += 1