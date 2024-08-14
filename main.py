from transformers import AutoTokenizer, BertModel
from transformers import AutoImageProcessor, ResNetForImageClassification
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset # consider local dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import evaluate
from typing import List, Dict
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image

def SSA(T: List[Dict], x, k: int): # Sample Search Approach for contrastive learning
    # args:
    # T: training set (assume list of dict [{'label': , 'feature': }, ...])
    # x: input news
    # k: sample size, the number of top samples to select
    # return:
    # S_p: Positive Sample Set
    # S_n: Negative Sample Set
    S_p = []
    S_n = []

    # Cosine similarity function
    cosine_similarity = nn.CosineSimilarity(dim=0)

    for sample in T:
        # if i.label == x.label:
        if sample['label'] == x['label']:
            S_p.append(sample)
        else:
            S_n.append(sample)



    x_feature_tensor = torch.tensor(x['feature'], dtype=torch.float32)


    # SSA這邊應該是拿multi heads之後的feature做比對的? 文字沒辦法cosine similarity吧? 畢竟要向量內積
    # 不確定 feature， 應該改成 text? 在看一下
    # Calculate and store cosine similarity for positive samples

    # 論文內沒有分 文字 圖片 轉成vector  所以才會說 一個vector能表示新聞內文 那 (應該會是從文字提取feature出來?)
    # 既然 提取feature 不是訓練中的話 那麼應該要把 feature提取 從這個function提出去
    # 不然 每次計算 loss 都call SSA. 盡量把所有計算都往外提 除非不得已
    #  x = {label: , feature} (多一個變數 存 資料全field-除label的field+新增feature field(在外部計算好))
    # List to store cosine similarities for S_p
    C = []
    for sample in S_p:
        if 'feature' in sample:
            sample_feature_tensor = torch.tensor(sample['feature'], dtype=torch.float32)
            sim = F.cosine_similarity(sample_feature_tensor.unsqueeze(0), x_feature_tensor.unsqueeze(0), dim=1)
            # Convert sim to scalar value by taking its mean or other aggregation
            C.append((sample, sim.mean().item()))  # .mean().item() converts tensor to scalar
        else:
            raise KeyError("The key 'feature' is missing in the sample")

    # Sort positive samples by cosine similarity in descending order
    #for i in range(len(C) - 1):
    #    for j in range(len(C) - i - 1):
    #        if C[j][1] < C[j + 1][1]:
    #            C[j], C[j + 1] = C[j + 1], C[j]

    # Select top-k positive samples
    #S_p_selected = []
    #for i in range(min(k, len(C))):
    #    S_p_selected.append(C[i][0])

    C.sort(key=lambda x: x[1], reverse=True)
    S_p_selected = [c[0] for c in C[:k]]
    
    # Calculate and store inverse cosine similarity for negative samples
    D = []
    for sample in S_n:
        if 'feature' in sample:
            sample_feature_tensor = torch.tensor(sample['feature'], dtype=torch.float32)
            sim = 1 / F.cosine_similarity(sample_feature_tensor.unsqueeze(0), x_feature_tensor.unsqueeze(0), dim=1)
            # Convert sim to scalar value by taking its mean or other aggregation
            D.append((sample, sim.mean().item()))  # .mean().item() converts tensor to scalar
        else:
            raise KeyError("The key 'feature' is missing in the sample")

        
    # Sort negative samples by inverse cosine similarity in descending order
    D.sort(key=lambda x: x[1], reverse=True)
    #for i in range(len(D) - 1):
    #    for j in range(len(D) - i - 1):
    #        if D[j][1] < D[j + 1][1]:
    #            D[j], D[j + 1] = D[j + 1], D[j]

    # Select top-k negative samples
    #S_n_selected = []
    #for i in range(min(k, len(D))):
    #    S_n_selected.append(D[i][0])
    #for j in range(len(D) - 1):
    #    if D[j][1] < D[j + 1][1]:
    #        D[j], D[j + 1] = D[j + 1], D[j]

    S_n_selected = [d[0] for d in D[:k]]

    return S_p_selected, S_n_selected    

class Custom_loss(nn.Module):
    def __init__(self):
        super(Custom_loss, self).__init__()
        self.loss1_fn = nn.CrossEntropyLoss()

        
    def forward(self, label, predict, input):
        ################# Cross-Entropy Loss ################
        cross_entropy_loss = self.loss1_fn(predict, label)

        ################# Contrastive Loss ###################
        S_p, S_n = SSA(train_data, input, sample_set_size) # type: ignore #(看要不要傳train_set)
        
        # Compute cosine similarities
        # cos_sim(x, sp)
        cos_sim_pos = torch.tensor([F.cosine_similarity(input['feature'], s_p['feature'], dim=1) for s_p in S_p])
        # cos_sim(x, sn)
        #cos_sim_neg = torch.tensor([F.cosine_similarity(input['feature'], s_n['feature'], dim=1) for s_n in S_n])
        cos_sim_neg = [F.cosine_similarity(torch.tensor(input['feature'], dtype=torch.float32),
                                   torch.tensor(s_n['feature'], dtype=torch.float32), dim=1)
               for s_n in S_n]

        # 如果你需要将列表转换为 Tensor，你可以在计算完相似度之后再执行这一步：
        cos_sim_neg = torch.stack(cos_sim_neg)

        # Compute the numerator and denominator of the loss
        # exp(cos(x, sp))
        numerator = torch.exp(cos_sim_pos)
        # Σ exp(cos(x, sn))
        denominator = torch.sum(torch.exp(cos_sim_neg))
    
        # Compute the log term
        # log( exp(cos(x, sp)) / Σ exp(cos(x, sn)))
        log_term = torch.log(numerator / denominator)
    
        # Compute the final loss
        contrasive_loss = -1 / (2 * sample_set_size) * torch.sum(log_term)
        
        ################ overall loss #####################
        # L_0 = (1-α)L_c + αL_s
        overall_loss = ( cross_entropy_loss * (1 - joint_learning_weight) + contrasive_loss * joint_learning_weight )
        
        return overall_loss


class Fake_news_detection(nn.Module):
  def __init__ (self):
    super(Fake_news_detection, self).__init__()
    self.bert = 'bert-base-uncased' #find huggingface pretrained bert
    self.resnet = 'microsoft/resnet-50' #find huggingface pretrained ResNet50
    
    # Bert extracts text feature
    self.tokenizer = AutoTokenizer.from_pretrained(self.bert)
    self.bert_model = BertModel.from_pretrained(self.bert)
    self.fc1 = nn.Linear(self.bert_model.config.hidden_size, 768) # check dim setting
    

    # self.transformer1 = nn.Transformer()
    self.transformer1 = nn.TransformerEncoder(
       nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, activation='relu'),
       num_layers=1
    )
    
    # fusion
    #self.multihead_attn = nn.MultiheadAttention(embed_dim=(768 * 2), num_heads=8)

    #joint learning:cross-entropy
    self.pool1 = nn.AdaptiveAvgPool1d(1)
    self.pool2 = nn.AdaptiveAvgPool1d(1)
    self.dropout = nn.Dropout(p=0.5)
    self.sigmoid = nn.Sigmoid()
    
    #joint learning:contrastive learning (no layer)
    
  def forward(self, x):
    # text feature extraction
    feature_text = self.tokenizer(x['text'], return_tensors='pt', padding=True, truncation=True, max_length=256)
    feature_text = self.bert_model(**feature_text).last_hidden_state #pooler_output 
    feature_text = F.relu(self.fc1(feature_text))


    # predict
    # y = self.pool1(feature_text.permute(1, 2, 0)).squeeze() # origin dim : (sequence_length, batch_size, feature_dim) -> (batch_size, feature_dim, sequence_length)
    # y = self.pool2(y.unsqueeze(-1)).squeeze() # undersampling
    # y = self.dropout(y)
    # y = self.sigmoid(y)
    ##
    y = self.pool1(feature_text.permute(0, 2, 1)).squeeze(-1)  # permute to [batch_size, feature_dim, seq_length]
    y = self.dropout(y) 
    y = self.sigmoid(y)


    print(f"Output shape after pooling: {y.shape}")
    
    return y

# parameter
joint_learning_weight = 0.2
sample_set_size = 5
batchsize = 16
learning_rate = 5e-6
num_epochs = 1


if __name__ == '__main__':

    #check dict

    os.makedirs("model", exist_ok=True)
    
    model = Fake_news_detection()
    if os.path.exists("model/checkpoint.pth"):
        checkpoint = torch.load("model/checkpoint.pth")
        model.load_state_dict(checkpoint)
        print("Loaded model from checkpoint.")
    else:
        print("No checkpoint found, starting with a new model.")
    # load dataset
    # code參考來源 確認執行正常後刪除
    # https://discuss.huggingface.co/t/how-to-load-a-huggingface-dataset-from-local-path/53076 
    # https://github.com/huggingface/datasets/issues/6691
    # =======================================================

    
    ## data_files = {'real':'BuzzFeed_fake_news_content.csv','fake':'BuzzFeed_fak.csv'}
    ## raw_datasets = load_dataset(name='csv', data_dir='./fakenewsnet', data_files=data_files, delimiter=',')
    
    #raw_datasets.remove_columns(["3_way_label", "6_way_label"])
    #raw_datasets.rename_column("2_way_label", "label")
    ##raw_datasets.set_format("torch")




    # read CSV file
    
    ## BuzzFeed
    df_BuzzFeed_real=pd.read_csv('fakenewsnet/BuzzFeed_fake_news_content.csv')
    df_BuzzFeed_fake=pd.read_csv('fakenewsnet/BuzzFeed_real_news_content.csv')
    ## politiFact
    df_politiFact_real=pd.read_csv('fakenewsnet/PolitiFact_fake_news_content.csv')
    df_politiFact_fake=pd.read_csv('fakenewsnet/PolitiFact_real_news_content.csv')

    ## combine these two dataframes into a single dataframe
    #df_BuzzFeed_real['image'] = df_BuzzFeed_real.apply(lambda x: Image.open(f"dataset_images/{x.name}.jpg") if os.path.exists(f"dataset_images/{x.name}.jpg") else None, axis=1)
    #df_BuzzFeed_real = df_BuzzFeed_real[df_BuzzFeed_real['image'].notnull()]
    #df_BuzzFeed_fake['image'] = df_BuzzFeed_fake.apply(lambda x: Image.open(f"dataset_images/{x.name}.jpg") if os.path.exists(f"dataset_images/{x.name}.jpg") else None, axis=1)
    #df_BuzzFeed_fake = df_BuzzFeed_fake[df_BuzzFeed_fake['image'].notnull()]
    #df_politiFact_real['image'] = df_politiFact_real.apply(lambda x: Image.open(f"dataset_images/{x.name}.jpg") if os.path.exists(f"dataset_images/{x.name}.jpg") else None, axis=1)
    #df_politiFact_real = df_politiFact_real[df_politiFact_real['image'].notnull()]
    #df_politiFact_fake['image'] = df_politiFact_fake.apply(lambda x: Image.open(f"dataset_images/{x.name}.jpg") if os.path.exists(f"dataset_images/{x.name}.jpg") else None, axis=1)
    #df_politiFact_fake = df_politiFact_fake[df_politiFact_fake['image'].notnull()]
###########################

    df_BuzzFeed=pd.concat([df_BuzzFeed_real,df_BuzzFeed_fake],axis=0)
    df_politiFact=pd.concat([df_politiFact_real,df_politiFact_fake],axis=0)
    df_text = pd.concat([df_BuzzFeed, df_politiFact])
    df_text = shuffle(df_text)
    
    ##新增一行news type
    ## Fake: id -> Fake_1-Webpage; Real: id -> Real_1-Webpage
    df_text['label']=df_text['id'].apply(lambda x: x.split('_')[0])

    content_text=df_text['text']
    label_text=df_text['label']

    
    ##分成測試集和訓練集 , 取30%資料為測試集
    text_train, text_test, label_train, label_test = train_test_split(content_text, label_text, test_size=0.3, random_state=42)
    
    train_data = []
    for text, label in zip(text_train, label_train):
        # 使用 BERT 计算特征向量
        inputs = model.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        outputs = model.bert_model(**inputs)
        feature = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        
        # 将特征添加到每个样本中
        train_data.append({'text': text, 'label': label, 'feature': feature})

    test_data = [{'text': text, 'label': label} for text, label in zip(text_test, label_test)]
    #print(label_train,text_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batchsize, shuffle=False)
    ##============================================================

    # preprocess train_set: data augmentation(back translation), feature extracted for compute cos_sim
        # not yet determine translation method
        # not yet determine feature extract method
        # small dataset for code debug
    #train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(100))
    #test_dataset = raw_datasets["test"].shuffle(seed=42).select(range(100))

    # construct dataloader
    #train_dataloader = Dataloader(dataset=train_dataset, batch_size=batchsize)
    #test_dataloader = Dataloader(dataset=test_dataset, batch_size=batchsize)
    
    
    # set loss function
    custom_criterion = Custom_loss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_dataloader))
    
    # attach model to gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # checkpoint frequency
    save_frequency = 5


    # training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader: # huggingface turtorial need to check how to access batch
            #batch = {k: v.to(device) for k, v in batch.items()}

            # ========================================            
            text_inputs = batch['text']
            #labels = batch['label'].to(device)
            # 定义标签映射
            label_to_index = {"real": 0, "fake": 1}
            # 将 labels 转换为数值，并创建 PyTorch 张量
            labels = torch.tensor([label_to_index[label.lower()] for label in batch['label']]).to(device)

            #text_inputs = [text.to(device) for text in text_inputs]  # 需要根據實際情況調整
            # 1. 先使用tokenizer将文本转换为模型的输入格式
            tokenized_inputs = model.tokenizer(text_inputs, return_tensors='pt', padding=True, truncation=True, max_length=256)

            # 2. 然后将这些张量转移到设备上
            input_ids = tokenized_inputs['input_ids'].to(device)
            attention_mask = tokenized_inputs['attention_mask'].to(device)


            # Forward pass
            outputs = model({'text': text_inputs})
            # ========================================
            # outputs = model(**batch)
            
            # Compute Loss
            #print(labels.shape, outputs.shape) 
            loss = custom_criterion(labels, outputs, batch) #wait loss function completed
            # loss = custom_criterion(labels, outputs, batch) #wait loss function completed
            loss.backward()

            # Backward pass
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad() #chatGPT: reset gradient → assume gradient init 0
            # save model at each epoch

            if(epoch + 1) % save_frequency == 0:
                checkpoint_path = f"model/checkpoint_epoch_{epoch + 1}.pth"
                torch.save(model.state_dict(), checkpoint_path)
    # save model
    torch.save(model.state_dict(), "model/final_model.pth")
    print("Final model saved.")
    
    # save model
    # torch.save(model.state_dict(), "model/")
    #orch.save() #設checkpoint

    # test loop
    ### copy from huggingface "fine-tune a pretrained model"
    metric = evaluate.load("accuracy")
    model.eval() # setting, eval mode
    tokenizer = BertModel.from_pretrained('bert-base-uncased')

    for batch in test_dataloader:
        # 將文本轉換為 tokens
        encoding = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')

        # 將 encoding 移動到指定設備上
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        labels = torch.tensor(batch['labels']).to(device)

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()
    ###