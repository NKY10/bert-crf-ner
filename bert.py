
# %%
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from modules.NERDataset import NERDataset
from modules.BERTCRF import BERTCRF
from torch.utils.tensorboard import SummaryWriter
from tools.evaluate import evaluate
import torch

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 打印出所选设备的信息
print(f"Using device: {device}")

# %%
# 加载训练、验证和测试数据
max_len = 150
batch_size = 16
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

trainset = NERDataset('dataset/weibo/train.txt', max_len=max_len, tokenizer=tokenizer)
train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# Check if label_map was created and saved
print("Label Map:", trainset.label_map)
testset = NERDataset('dataset/weibo/test.txt', max_len=max_len, tokenizer=tokenizer)
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
# valset = NERDataset('val.txt', label_map_path="dataset/val_labels.json", max_len=max_len)
# val_dataloader = DataLoader(valset, batch_size=16, shuffle=True)

# %%
# 加载模型
label_map = trainset.label_map
model = BERTCRF(num_labels=len(label_map)).to(device)

# %%
# 训练模型
from torch import optim
optimizer_grouped_parameters = [
    {'params': model.crf.parameters(), 'lr': 3e-2},
    {'params': model.bert.parameters(), 'lr': 5e-5}
]


optimizer = optim.AdamW(optimizer_grouped_parameters)

# 初始化 TensorBoard
writer = SummaryWriter()

# %%
from tqdm.auto import tqdm

# ... 其他导入和模型定义 ...

# 创建一个tqdm进度条，用于显示训练进度


for epoch in tqdm(range(40),desc="epochs"):  
    model.train()
    total_loss = 0
    maxf1 = 0.5
    train_dataloader = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        # 将数据移动到GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        loss = model(input_ids, attention_mask, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        # 更新进度条的后缀，显示平均损失
        train_dataloader.set_postfix({'loss': '{:.6f}'.format(loss.item())})

    avg_train_loss = total_loss / len(train_dataloader)

    precision, recall, f1 = evaluate(model, test_dataloader,device=device)

    print(f'Epoch {epoch+1}, Loss: {avg_train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    
    # 记录到 TensorBoard
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Precision/val', precision, epoch)
    writer.add_scalar('Recall/val', recall, epoch)
    writer.add_scalar('F1/val', f1, epoch)
    #writer.add_scalar('Accuracy/val', accuracy, epoch)

    if f1 > maxf1:
        maxf1 = f1
        torch.save(model.state_dict(), './model/model.parameters')

# 关闭 TensorBoard
writer.close()
