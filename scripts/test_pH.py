import os
import pandas as pd
from tqdm import tqdm
import math
import json 
import pickle
import numpy as np 
from itertools import combinations
import base64
import hashlib
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import itertools
import random
import argparse


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)

class AttentionPoolingModel(nn.Module):
    def __init__(self, dim=512, use_attention=True, dropout=0.2):
        super().__init__()
        self.use_attention = use_attention
        self.dropout = nn.Dropout(dropout)

        if self.use_attention:
            self.attn = nn.Sequential(
                nn.Linear(dim, 64),
                nn.Tanh(),
                nn.LayerNorm(64),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask):
        B, L, D = x.shape 

        if self.use_attention:
            attn_scores = self.attn(x).squeeze(-1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        else:
            mask_expanded = mask.unsqueeze(-1).float()
            x_masked = x * mask_expanded
            lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            pooled = x_masked.sum(dim=1) / lengths

        pooled = self.norm(pooled)
        out = self.fc(pooled).squeeze(-1)
        return out

class pHDataset(Dataset):
    def __init__(self, data, max_length=8000):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emb_path = row['emb_path']
        feats, mask = self.load_embedding(emb_path)
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
        
    def load_embedding(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            feats = np.array(pickle.load(f)['mean_feats'].float().cpu())
        L = feats.shape[0]
        if L >= self.max_length:
            feats = feats[:self.max_length]
            mask = np.ones(self.max_length)
        else:
            feats = np.pad(feats, ((0, self.max_length-L), (0, 0)))
            mask = np.concatenate([np.ones(L), np.zeros(self.max_length-L)])
        return feats, mask

def evaluate(model, dataloader, device, scaler):
    model.eval()
    all_preds = []
    all_genome_ids = []
    with torch.no_grad():
        for i, (x, mask) in enumerate(dataloader):
            x, mask = x.to(device), mask.to(device)
            batch_size = len(x)
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(dataloader.dataset.data))
            batch_genome_ids = dataloader.dataset.data.iloc[start_idx:end_idx]['genome_id'].tolist()
            all_genome_ids.extend(batch_genome_ids)
            out = model(x, mask)
            all_preds.append(out.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    return all_genome_ids, y_pred_denorm
def main(args):
    emb_path = args.output_dir
    max_length = 8000
    USE_ATTENTION = True
    epoch = 10
    pt_path = f'./weights/downstream_tasks/train_attn_pH/len8000_attentionTrue/10_gene_attn_optim_model.pt'
    txt_path = './scripts/data/genome_unique_ids.csv'
    task = 'pH' 
    output_folder = args.output_dir + f'/test_attn_{task}'
    os.makedirs(output_folder, exist_ok=True)

    train_path = './scripts/data/split_meta_ph_phylo.csv'  
    train_data = pd.read_csv(train_path)
    scaler = MinMaxScaler()
    train_labels = train_data['label'].values.reshape(-1, 1) 
    scaler.fit(train_labels)

    sequences = pd.read_csv(txt_path)
    data_set = []
    for _, seq_info in tqdm(sequences.iterrows(), total=len(sequences)):
        genome_id = seq_info['genome_id']
        gene_emb_path = os.path.join(emb_path, genome_id,f'{genome_id}.pkl')
        if os.path.exists(gene_emb_path):
            data_set.append({'genome_id': genome_id, 'emb_path': gene_emb_path})
    data_df = pd.DataFrame(data_set)

    test_dataset = pHDataset(data_df, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionPoolingModel(dim=512, use_attention=USE_ATTENTION).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))

    genome_ids, y_pred_denorm = evaluate(model, test_loader, device, scaler)

    result_df = pd.DataFrame({
        'genome_id': genome_ids,
        f'predicted_{task}_value': y_pred_denorm
    })
    base_name = os.path.splitext(os.path.basename(pt_path))[0]
    output_path = os.path.join(output_folder, f'predicted_{task}_{base_name}_epoch{epoch}_results.csv')
    result_df.to_csv(output_path, index=False)

    print(f"Prediction results saved to: {output_path}")
    print(f"Predicted pH range: {np.min(y_pred_denorm):.2f} ~ {np.max(y_pred_denorm):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()
    main(args)