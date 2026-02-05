import os
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import argparse


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)


class AttentionPoolingModel(nn.Module):
    def __init__(self, dim=512, use_attention=False, dropout=0.2):
        super().__init__()
        self.use_attention = use_attention

        if self.use_attention:
            self.attn = nn.Sequential(
                nn.Linear(dim, 64),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )

        self.fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x, mask):
        if self.use_attention:
            attn_scores = self.attn(x).squeeze(-1)   # (B, L)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        else:
            mask = mask.unsqueeze(-1)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        out = self.fc(pooled)
        return out


class ProbioticInferenceDataset(Dataset):
    def __init__(self, data, max_length=8000):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        feats, mask = self.load_embedding(row['emb_path'])
        return (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )

    def load_embedding(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            feats = np.array(pickle.load(f)['mean_feats'].float().cpu())

        L = feats.shape[0]
        if L >= self.max_length:
            feats = feats[:self.max_length]
            mask = np.ones(self.max_length)
        else:
            pad = self.max_length - L
            feats = np.pad(feats, ((0, pad), (0, 0)))
            mask = np.concatenate([np.ones(L), np.zeros(pad)])

        return feats, mask


def inference(model, dataloader, device):
    model.eval()
    all_genome_ids = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i, (x, mask) in enumerate(tqdm(dataloader, desc="Infer")):
            x, mask = x.to(device), mask.to(device)

            start = i * dataloader.batch_size
            end = start + len(x)
            genome_ids = dataloader.dataset.data.iloc[start:end]['genome_id'].tolist()
            all_genome_ids.extend(genome_ids)

            out = model(x, mask)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(out, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)

    return all_genome_ids, all_preds, all_probs

def main(args):

    task = 'probiotic'
    emb_dir = args.output_dir
    txt_path = './scripts/data/genome_unique_ids.csv'

    epoch = 38
    level = args.level
    max_length = 8000
    USE_ATTENTION = False
    BATCH_SIZE = 1

    if level=='family':
        pt_path = f'./weights/downstream_tasks/train_attn_probiotic/len8000_attentionFalse_family/epoch38_gene_attn_optim_model.pt'
    elif level=='genus':
        pt_path = f'./weights/downstream_tasks/train_attn_probiotic/len8000_attentionFalse_genus/epoch28_gene_attn_optim_model.pt'
    else:
        raise ValueError(f'Invalid level: {level}')
    output_dir = args.output_dir + f'/test_attn_{task}_{level}'
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(txt_path)

    data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        genome_id = row['genome_id']
        emb_path = os.path.join(emb_dir,genome_id, f'{genome_id}.pkl')
        if os.path.exists(emb_path):
            data.append({
                'genome_id': genome_id,
                'emb_path': emb_path
            })

    data_df = pd.DataFrame(data)
    print(f"Total: {len(df)}, With embedding: {len(data_df)}")


    dataset = ProbioticInferenceDataset(data_df, max_length=max_length)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionPoolingModel(dim=512, use_attention=USE_ATTENTION).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))


    genome_ids, preds, probs = inference(model, loader, device)


    result_df = pd.DataFrame({
        'genome_id': genome_ids,
        f'predicted_{task}_label': preds,
        f'predicted_{task}_prob': probs
    })

    base = os.path.splitext(os.path.basename(pt_path))[0]
    out_path = os.path.join(output_dir, f'predicted_{task}_{base}.csv')
    result_df.to_csv(out_path, index=False)

    print(f"Saved to: {out_path}")
    print("Prediction distribution:")
    print(pd.Series(preds).value_counts())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--level', type=str, default='family')
    args = parser.parse_args()
    main(args)