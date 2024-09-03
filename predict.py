from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 加載微調過的模型和標記器
model_path = "results/checkpoint-1015"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# 要進行預測的輸入文本
titles = ["Nearly half of Twitter accounts discussing coronavirus are likely bots, researchers say","Mapping the worldwide spread of the coronavirus"]
contents = ["Nearly half of the Twitter accounts sharing information about the novel coronavirus are likely bots, according to researchers at Carnegie Mellon University.","World Mapping the worldwide spread of the coronavirus Warning: This graphic requires JavaScript. Please enable JavaScript for the best experience."]
texts = [f"{title} {content}" for title, content in zip(titles, contents)]

# 使用標記器將文本轉換為模型輸入格式
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 將模型設置為評估模式
model.eval()

# 禁用梯度計算
with torch.no_grad():
    outputs = model(**inputs)

# 獲取預測結果
logits = outputs.logits

# 將logits轉換為概率分佈
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# 獲取每個文本的預測標籤
predictions = torch.argmax(probabilities, dim=-1)

# 打印預測結果
for text, pred in zip(texts, predictions):
    print(f"Title: {titles}")
    #print(f"Text: {contents}")
    print(f"Predicted class: {pred.item()}")
