from transformers import AutoTokenizer, BertModel
from transformers import AutoImageProcessor, ResNetForImageClassification
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset # consider local dataset
from torch.utils.data import Dataloader
from torch.optim import AdamW
from transformers import get_scheduler
import evaluate
from typing import List, Dict

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
            S_p.apeend(sample)
        else:
            S_n.append(sample)

    # SSA這邊應該是拿multi heads之後的feature做比對的? 文字沒辦法cosine similarity吧? 畢竟要向量內積
    # 不確定 feature， 應該改成 text? 在看一下
    # Calculate and store cosine similarity for positive samples
    C = []  # List to store cosine similarities for S_p

    # 論文內沒有分 文字 圖片 轉成vector  所以才會說 一個vector能表示新聞內文 那 (應該會是從文字提取feature出來?)
    # 既然 提取feature 不是訓練中的話 那麼應該要把 feature提取 從這個function提出去
    # 不然 每次計算 loss 都call SSA. 盡量把所有計算都往外提 除非不得已
    #  x = {label: , feature} (多一個變數 存 資料全field-除label的field+新增feature field(在外部計算好))
    for sample in S_p:
        sim = cosine_similarity(sample['feature'], x['feature']) # 目前認為sample , x 是 資料集的 一筆資料, 所以 一筆資料有多少field 取決於 別人現成資料集 怎麼建的
        C.append((sample, sim)) # 0: text, 1: cos_similarity

    # Sort positive samples by cosine similarity in descending order
    for i in range(len(C) - 1):
        for j in range(len(C) - i - 1):
            if C[j][1] < C[j + 1][1]:
                C[j], C[j + 1] = C[j + 1], C[j]

    # Select top-k positive samples
    S_p_selected = []
    for i in range(min(k, len(C))):
        S_p_selected.append(C[i][0])

    # Calculate and store inverse cosine similarity for negative samples
    D = []  # List to store inverse cosine similarities for S_n
    for sample in S_n:
        sim = 1 / cosine_similarity(sample['feature'], x['feature'])
        D.append((sample, sim))

    # Sort negative samples by inverse cosine similarity in descending order
    for i in range(len(D) - 1):
        for j in range(len(D) - i - 1):
            if D[j][1] < D[j + 1][1]:
                D[j], D[j + 1] = D[j + 1], D[j]

    # Select top-k negative samples
    S_n_selected = []
    for i in range(min(k, len(D))):
        S_n_selected.append(D[i][0])

    return S_p_selected, S_n_selected    

class Custom_loss(nn.Module):
    def __init__(self):
        super(Custom_loss, self).__init__()
        self.loss1_fn = nn.CrossEntropyLoss()

        
    def forward(self, label, predict, input):
        ################# Cross-Entropy Loss ################
        cross_entropy_loss = self.loss1_fn(predict, label)

        ################# Contrastive Loss ###################
        S_p, S_n = SSA(train_set, input, sample_set_size) #(看要不要傳train_set)
        
        # Compute cosine similarities
        # cos_sim(x, sp)
        cos_sim_pos = torch.tensor([F.cosine_similarity(input['feature'], s_p['feature'], dim=1) for s_p in S_p])
        # cos_sim(x, sn)
        cos_sim_neg = torch.tensor([F.cosine_similarity(input['feature'], s_n['feature'], dim=1) for s_n in S_n])
        
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
    
    #ResNet50 extracts image feature: image  → feature e^i → feature e^i'
    self.processor = AutoImageProcessor.from_pretrained(self.resnet)
    self.resnet_model = ResNetForImageClassification.from_pretrained(self.resnet)
    self.fc2 = nn.Linear(self.resnet_model.config.hidden_size, 512) #check dim setting
    self.fc3 = nn.Linear(512, 768) #check dim setting

    # self.transformer1 = nn.Transformer()
    self.transformer1 = nn.TransformerEncoder(
       nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048, activation='relu'),
       num_layers=1
    )
    
    # fusion
    self.multihead_attn = nn.MultiheadAttention(embed_dim=(768 * 2), num_heads=8)

    #joint learning:cross-entropy
    self.pool1 = nn.AdaptiveAvgPool1d(1)
    self.pool2 = nn.AdaptiveAvgPool1d(1)
    self.dropout = nn.Dropout(p=0.5)
    self.sigmoid = nn.Sigmoid()
    
    #joint learning:contrastive learning (no layer)
    
  def forward(self, x):
    # text feature extraction
    feature_text = self.tokenizer(x.text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    feature_text = self.bert_model(**feature_text).last_hidden_state
    feature_text = F.relu(self.fc1(feature_text))

    # image feature extraction
    image_inputs = self.processor(x.image, return_tensors='pt')
    image_output = self.resnet_model(**image_inputs).pooler_output
    feature_image = F.relu(self.fc2(image_output))
    feature_image = F.relu(self.fc3(feature_image))

    # image Transformer encoding
    feature_image = feature_image.unsqueeze(0) # Add a dimension
    feature_image = self.transformer1(feature_image).squeeze(0)


    # fusion
    feature_fusion = torch.cat((feature_text, feature_image), dim=1).unsqueeze(0)
    feature_fusion, _ = self.multihead_attn(feature_fusion, feature_fusion, feature_fusion) # 2 output, attn_output & attn_output_weights
    
    # predict
    y = self.pool1(feature_fusion.permute(1, 2, 0)).squeeze() # origin dim : (sequence_length, batch_size, feature_dim) -> (batch_size, feature_dim, sequence_length)
    y = self.pool2(y.unsqueeze(-1)).squeeze() # undersampling
    y = self.dropout(y)
    y = self.sigmoid(y)

    return y

# parameter
joint_learning_weight = 0.2
sample_set_size = 5
batchsize = 16
learning_rate = 5e-6
num_epochs = 100


if __name__ == '__main__':
    
    # load model if exists, o.w. new 
    model = Fake_news_detection()

    # load dataset
    train_set =
    test_set = 
    
    # preprocess train_set: data augmentation(back translation), feature extracted for compute cos_sim
        # not yet determine tranlation method
        # not yet determine feature extract method
    # construct dataloader
    train_dataloader = Dataloader(train_set, batch_size=batchsize)
    test_dataloader = Dataloader(test_set, batch_size=batchsize)
    
    
    # set loss function
    custom_criterion = Custom_loss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_dataloader))
    
    # attach model to gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    
    # training loop
    for epoch in range(num_epochs):
        for batch in train_dataloader: # huggingface turtorial need to check how to access batch
            batch = {k: v.to(device) for k, v in batch.items()} #key, value?
            outputs = model(**batch)
            
            # Compute Loss 
            loss = custom_criterion(k, outputs, ) #wait loss function completed
            loss.backward()

            # Backward pass
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad() #chatGPT: reset gradient → assume gradient init 0
        

    # test loop
    ### copy from huggingface "fine-tune a pretrained model"
    metric = evaluate.load("accuracy")
    model.eval() # setting, eval mode
    for batch in test_dataloader: # huggingface turtorial need to check how to access batch
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()
    ###
