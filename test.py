import os
#from datasets import load_dataset
import numpy as np
import evaluate
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import torch
from sklearn.metrics import confusion_matrix, classification_report

# 設置檢查點路徑
output_dir='./tests'         # 輸出目錄

"""# 資料集預處理

"""

# 加載數據集
data = pd.read_csv("kaggle_fake_train.csv")
total_rows_before = len(data)

data = data.dropna(subset=["title"])  # 去除文本为空的行
dropped_title = total_rows_before - len(data)

data = data.dropna(subset=["text"])  # 去除文本为空的行
dropped_content = total_rows_before - dropped_title - len(data)

data = data.dropna(subset=["label"])  # 去除 reliability 列为空的行
dropped_label = total_rows_before - dropped_title -dropped_content- len(data)

data["text"] = data["title"] + " " + data["text"]

data = data[["text", "label"]]  # 保留 text跟label


print(f"Rows dropped due to empty title: {dropped_title}")
print(f"Rows dropped due to empty content: {dropped_content}")
print(f"Rows dropped due to empty label: {dropped_label}")

#data = data.rename(columns={"reliability": "labels"})
dataset = Dataset.from_pandas(data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
test_dataset = dataset['test']

"""# 處理數據集"""

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    # Tokenize the text
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    # Check if the length exceeds max_length
    tokenized["valid"] = [len(tokenizer.encode(text, truncation=True, max_length=512)) <= 512 for text in examples["text"]]
    return tokenized

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)


# Filter out samples that exceed max_length
train_dataset = train_dataset.filter(lambda example: example["valid"])
test_dataset = test_dataset.filter(lambda example: example["valid"])

print(f"length of train_set",len(train_dataset))
print(f"length of tset_set",len(test_dataset))


# Remove the 'valid' column as it's no longer needed
train_dataset = train_dataset.remove_columns(["valid"])
test_dataset = test_dataset.remove_columns(["valid"])

"""#  設置評估指標"""

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"]}

# 訓練參數設置
training_args = TrainingArguments(
    output_dir=output_dir,          # 輸出目錄
    num_train_epochs=5,              # 訓練輪數
    per_device_train_batch_size=16,   # 每個設備的訓練批量大小
    per_device_eval_batch_size=16,    # 每個設備的評估批量大小
    warmup_steps=1000,                # 熱身步數
    weight_decay=0.01,               # 權重衰減
    logging_dir='./logs',            # 日誌目錄
    logging_steps=10,
    save_steps=10,                  # 儲存檢查點的步數間隔
    save_total_limit=2               # 只保留最新的兩個檢查點
)

"""# 訓練模型"""

# 檢查是否有現存的檢查點

last_checkpoint = None

if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
    print("start from checkpoint")
# 获取最近保存的检查点路径
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getctime)  # 获取最新的检查点路径
        print(f"chepoints!!!!!!!!!!!!!",last_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(last_checkpoint)

    # 從檢查點加載模型
else:
    print("start from beginning")
    # 如果沒有檢查點，從頭開始訓練
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)


# 訓練模型
if os.path.exists(output_dir):
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

# 儲存模型和狀態
trainer.save_model(output_dir)
trainer.save_state()



def collate_fn(batch):
    # 将 batch 中的数据转换为张量
    return {
        'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
        'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
        'token_type_ids': torch.stack([torch.tensor(item['token_type_ids']) for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    }
# 構建測試數據加載器
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

"""# 評估模型"""

def evaluate_model(model, test_dataloader, device):
    ground_truth = []
    prediction = []
    model.eval()  # 設置為評估模式
    with torch.no_grad():
        for batch in test_dataloader:
            if isinstance(batch, dict):
                batch = {k: torch.tensor(v).to(device) if isinstance(v, list) else v.to(device) for k, v in batch.items()}
            else:
                batch = {k: torch.tensor(v).to(device) if isinstance(v, list) else v.to(device) for k, v in dict(zip(test_dataset.column_names, batch)).items()}

            outputs = model(**batch)
            y_pred = outputs.logits.argmax(dim=1)
            ground_truth.extend(batch['labels'].cpu().numpy())
            prediction.extend(y_pred.cpu().numpy())

    print(confusion_matrix(ground_truth, prediction))
    print(classification_report(ground_truth, prediction))

# 訓練完成後進行評估
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
evaluate_model(model, test_dataloader, device)