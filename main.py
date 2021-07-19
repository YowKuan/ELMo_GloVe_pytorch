# -*- coding: utf-8 -*-
# +
import os

#change result directory here
result_directory = '/home/yoyo507017/BERT-training-pytorch/NLP-Death-PeiFu/eval_results/0719/'
if not os.path.exists(result_directory):
    
    os.makedirs(result_directory)
    print("directory created")
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# -


from data_pro import load_data_and_labels, Data
from model import Model
from config import opt


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


# +
device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")

opt.device = device
#Set your method here. emb_method(glove, bert, elmo), enc_method(cnn, rnn, transformer), model_label(name as you want, it )
opt.parse({'emb_method': 'glove', 'enc_method': 'rnn', 'model_label': 'glovernn'})

class_weights = [1, 5]
weights= torch.tensor(class_weights,dtype=torch.float)
weights = weights.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

model = Model(opt)
print(f"{now()} {opt.emb_method} init model finished")

if opt.use_gpu:
    model.to(device)
    
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
lr_sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)


# -

def train():
#     print("kwargs:", kwargs)
#     print("opt:", opt)


    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)
        
    #training data
    df_train = pd.read_csv('../NLP-Death-PeiFu/NLP_death30_20210625_processed_train.csv')
    x_train, x_test, y_train, y_test = train_test_split(df_train['術前診斷']+df_train['預計手術名稱'], df_train['Label'], 
                                                                random_state=2020, 
                                                                test_size=0.2, 
                                                                )

    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"{now()} train data: {len(train_data)}, test data: {len(test_data)}")
    
    best_acc = -0.1
    best_loss = float('inf')
    best_epoch = -1
    start_time = time.time()
    for epoch in range(1, opt.epochs):
        total_loss = 0.0
        model.train()
        for step, batch_data in enumerate(train_loader):
            if step%50 == 0:
                print(step)
            x, labels = batch_data
            labels = torch.LongTensor(labels)
            if opt.use_gpu:
                labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        
        #validation
        total_loss = 0
  
        # empty list to save the model predictions
        total_preds = []
        predict_all = []
        correct = 0
        num = 0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                x, labels = data
                num += len(labels)
                #output = model(x)
                preds = model(x)
                labels = torch.LongTensor(labels)
                if opt.use_gpu:
                    labels = labels.to(device)
                
                # compute the validation loss between actual and predicted values
                eval_loss = criterion(preds,labels)
                total_loss += eval_loss.item()
                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)



                #correct += (predict == labels).sum().item()
        total_preds  = np.concatenate(total_preds, axis=0)
        eval_loss = total_loss / len(test_loader) 
        model.train()
        #acc, total_preds, eval_loss = test(model, test_loader, criterion, device)
        #print(total_preds)
        #print(acc)
        if eval_loss < best_loss:
            print("model update")
            best_loss = eval_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'saved_weights_{}.pt'.format(opt.model_label))
        print(f"{now()} Epoch{epoch}: avg_valid_loss: {eval_loss}") # test_acc: {acc}")
        lr_sheduler.step()

    end_time = time.time()
    print("*"*20)
    print(f"{now()} finished; epoch {best_epoch} best loss: {best_loss}, time/epoch: {(end_time-start_time)/opt.epochs}")

train()

# +
#put test data here
df_test = pd.read_csv('../NLP-Death-PeiFu/NLP_death30_20210713_processed_test.csv')

x_test, y_test = df_test['術前診斷']+df_test['預計手術名稱'], df_test['Label']


test_data = Data(x_test, y_test)
test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

total_loss = 0
  
# empty list to save the model predictions
total_preds = []
total_probs=[]
num = 0
model.eval()
with torch.no_grad():
    for data in test_loader:
        x, labels = data
        num += len(labels)
        preds = model(x)
        #print(preds)
        labels = torch.LongTensor(labels)
        preds = preds.detach().cpu().numpy()
        probs = np.exp(preds[:,1])
        preds = np.argmax(preds, axis = 1)
        total_preds.extend(preds)
        total_probs.extend(probs)
# -

with open("../NLP-Death-PeiFu/eval_results/0719/glovernn.csv") as f:
    df = pd.read_csv(f)


#change probability to threshold
threshold_change = []
#for i in range(len(total_probs)):
for i in range(len(df['predicted'])):
    if df['predicted'][i] > 0.75:
    #if total_probs[i] > 0.82 :
        threshold_change.append(1)
    else:
        threshold_change.append(0)
print(len(x_test), len(y_test), len(total_probs))

from sklearn.metrics import classification_report
print(classification_report(y_test, threshold_change, target_names=['0', '1']))

df_test = pd.DataFrame({'text':x_test, 'labels':y_test, 'predicted':total_probs})
#pd.set_option('display.max_rows', df_train.shape[0]+1)
df_test['Deviation'] = abs(df_test['labels']-df_test['predicted'])
df_test['result'] = df_test['Deviation'] < 0.5
df_test.to_csv('{}{}.csv'.format(result_directory, opt.model_label), index=False)

# +
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('Groud Truth')
    plt.xlabel('Predicted mortality')
    plt.savefig('{}{}_ConfusionMatrix.png'.format(result_directory, opt.model_label), bbox_inches='tight')

cm = confusion_matrix(y_test, threshold_change)
df_cm = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
show_confusion_matrix(df_cm)

# +
from sklearn.metrics import accuracy_score, roc_curve, auc

def evaluate_roc(probs, test_y):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs
    fpr, tpr, threshold = roc_curve(test_y, preds)
    #print(threshold)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    print(optimal_idx)
    optimal_threshold = threshold[optimal_idx]
    print("Threshold value is:", optimal_threshold)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
#     y_pred = np.where(preds >= 0.50, 1, 0)
#     accuracy = accuracy_score(test_y, y_pred)
#     print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Plot ROC AUC
    plt.title('CAD AUROC Performance')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    plt.savefig('{}{}_AUROC.png'.format(result_directory, opt.model_label))
    return roc_auc


# -

evaluate_roc(total_probs, y_test)


