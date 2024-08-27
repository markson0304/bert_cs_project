import os
from datasets import load_dataset
import numpy as np
import evaluate
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

# 設置檢查點路徑
output_dir='./results'         # 輸出目錄
checkpoint_dir = os.path.join(output_dir, 'checkpoint-375')

# 加載數據集
dataset = load_dataset("yelp_review_full")

# 處理數據集
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 選取較小的訓練和評估集進行快速測試
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# 設置評估指標
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 訓練參數設置
training_args = TrainingArguments(
    output_dir=output_dir,          # 輸出目錄
    num_train_epochs=3,              # 訓練輪數
    per_device_train_batch_size=8,   # 每個設備的訓練批量大小
    per_device_eval_batch_size=8,    # 每個設備的評估批量大小
    warmup_steps=500,                # 熱身步數
    weight_decay=0.01,               # 權重衰減
    logging_dir='./logs',            # 日誌目錄
    logging_steps=10,
    save_steps=100,                  # 儲存檢查點的步數間隔
    save_total_limit=2               # 只保留最新的兩個檢查點
)



# 檢查是否有現存的檢查點
if os.path.exists(checkpoint_dir):
    print("start from checkpoint")
    # 從檢查點加載模型
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

    # 繼續訓練
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint='./results/checkpoint-375')
else:
    print("start from beggining")
    # 如果沒有檢查點，從頭開始訓練
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()


# 儲存模型和狀態
trainer.save_model(output_dir)
trainer.save_state()
