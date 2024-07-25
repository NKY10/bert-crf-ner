from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import torch


def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            batch_predictions = model(input_ids, attention_mask)
            
            # 将预测结果和真实标签转换为list
            batch_predictions = [p for p in batch_predictions]  # 转换为list，便于后续处理
            labels = labels.tolist()  # 将tensor转换为list

            # 扩展预测结果和真实标签
            predictions.extend(batch_predictions)
            true_labels.extend(labels)

    # 注意：CRF的decode输出是经过mask的，即它只包含非padding部分的预测
    # 因此，我们需要从真实标签中去除padding部分
    pred_flat,labels_flat= [],[]
    for i in range(len(predictions)):
        no_padding_length = len(predictions[i])
        pred_flat += predictions[i]
        labels_flat += true_labels[i][0:no_padding_length]    
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%",sum(pred_flat))
    # Calculate metrics
    precision = precision_score(labels_flat, pred_flat, average='weighted', zero_division=0)
    recall = recall_score(labels_flat, pred_flat, average='weighted', zero_division=0)
    f1 = f1_score(labels_flat, pred_flat, average='weighted', zero_division=0)
    return precision, recall, f1
