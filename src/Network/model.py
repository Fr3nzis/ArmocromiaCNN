import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Network.blocks import ConvBlock


class CNNMultiTask(nn.Module):

    def __init__(self, num_stagioni: int = 4, num_sottotipi: int = 6):
        super().__init__()

        self.num_stagioni = num_stagioni
        self.num_sottotipi = num_sottotipi

        self.features = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )


        self.gap = nn.AdaptiveAvgPool2d((1, 1))


        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            out = self.features(dummy)
            out = self.gap(out)               
            flatten_dim = out.shape[1]       
        self.flatten_dim = flatten_dim       

        self.fc_stagione = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_stagioni)
        )

        self.fc_sottotipo = nn.Sequential(
            nn.Linear(self.flatten_dim + num_stagioni, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_sottotipi)
        )
        
    def forward(self, x):

        f = self.features(x)         
        f = self.gap(f)             
        f = torch.flatten(f, 1)       


        logits_stagione = self.fc_stagione(f)          
        prob_stagione = F.softmax(logits_stagione, 1)  


        cond_input = torch.cat([f, prob_stagione], dim=1)   
        logits_sottotipo = self.fc_sottotipo(cond_input)    

        return logits_stagione, logits_sottotipo
