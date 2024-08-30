from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 加載微調過的模型和標記器
model_path = "results/checkpoint-375"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# 要進行預測的輸入文本
titles = ["Everything you need to know about the coronavirus","Novel Coronavirus Cases Confirmed To Be Spreading"]
contents = ["Public health experts around the globe are scrambling to understand, track, and contain a new virus that appeared in Wuhan, China, at the beginning of December 2019. The World Health Organization (WHO) named the disease caused by the virus COVID-19.","The first two coronavirus cases in Europe have been detected in France, and a second case has been confirmed in America as China expands its efforts to control its outbreak."]
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
