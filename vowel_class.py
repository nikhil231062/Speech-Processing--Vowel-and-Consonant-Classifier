import torch
import torch.nn as nn
import librosa
import numpy as np
import os

class lstm_model(nn.Module):
    def __init__(self,input,hidden,out):
        super(lstm_model,self).__init__()
        self.l1=nn.LSTM(input,hidden,dtype=torch.double)
        self.l2=nn.LSTM(hidden,hidden,dtype=torch.double)
        self.l3=nn.Linear(hidden,out,dtype=torch.double)
        self.l4=nn.Softmax(dim=2)
    def forward(self,data):
        # print(data)
        h0,_=self.l1(data)
        h0,_=self.l2(h0)
        e=self.l3(h0)
        return self.l4(e)
path='D:/codes/M.Tech_proj/SPP_assignment_data/SPP_assignment_data/noisy_speech/noisy_speech/train'
a_noisy=[]
i_noisy=[]
u_noisy=[]
for each in os.listdir(path):
    wav,sr=librosa.load(path+'/'+each)
    if each.split('_')[1] in ['II']:
        f0,_,_=librosa.pyin(wav,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'),sr=sr)
        f0[np.isnan(f0)]=0
        lpc=librosa.lpc(wav,order=16)
        # mfcc=librosa.feature.mfcc(wav)
        # spect=librosa.stft(wav)
        # spect=librosa.amplitude_to_db(np.abs(spect),ref=np.max)
        if each.split('_')[1]=='AA':
            a_noisy.append([f0,0])
        if each.split('_')[1]=='II':
            i_noisy.append([f0,1])
        if each.split('_')[1]=='UU':
            u_noisy.append([f0,2])
dataset=a_noisy+i_noisy+u_noisy
np.random.shuffle(dataset)

model_noisy=lstm_model(1,6,3)
lf=nn.CrossEntropyLoss()
optim=torch.optim.Adam(model_noisy.parameters(),lr=0.01)
model_noisy.to(device='cuda:0')
lf.to(device='cuda:0')
for epoch in range(100):
    l=0
    for each in dataset:
        model_noisy.zero_grad()
        pred=model_noisy(torch.tensor(each[0].reshape(each[0].shape[0],1,1)).to(torch.double).to(device='cuda:0'))
        pred.squeeze(dim=1)
        label=torch.tensor([0,0,0])
        label[each[1]]=1
        label=label.repeat(pred.shape[0],1)
        label=label.to(device='cuda:0')
        loss=lf(pred,label)
        loss.backward()
        with torch.no_grad():
            l+=loss
    optim.step()
    print("epoch {} loss {}".format(epoch,l))
torch.save(model_noisy,'D:/codes/M.Tech_proj/noisy_model')