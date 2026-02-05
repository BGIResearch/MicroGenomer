import os
import re  
import sys
import time
import json
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel

import warnings
warnings.filterwarnings("ignore")
project_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_path)


class DNAConverter:  
    def __init__(self):  
        self._DNA2ID_dict = {  
            "A": '1',  
            "T": '2',   
            "U": '2',   
            "C": '3',  
            "G": '4',   
            "N": '5',  
        }  

    def convert(self, dna_string):   
        return ''.join(self._DNA2ID_dict.get(base, '5') for base in dna_string)  



class DNAInference:
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.model_path = args.model_path
        self.max_seq_len = args.max_seq_len
        self.padding_length = args.max_seq_len
        self.save_dir = args.save_dir
        self.output_attentions = args.output_attentions
        self.output_hidden_states = args.output_hidden_states
        self.ckpt_path = args.ckpt_path
        self.args = args

        self.initialize_model(args)
        self.converter = DNAConverter()

    def initialize_model(self, args):
        os.makedirs(args.save_dir, exist_ok=True)

        # Model configuration based on model type
        NHIDDEN, FFN_HIDDEN, NLAYERS, NHEADS = self.get_model_parameters()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            ffn_hidden_size=FFN_HIDDEN,
            hidden_size=NHIDDEN,
            kv_channels=int(NHIDDEN / NHEADS),
            num_attention_heads=NHEADS,
            num_layers=NLAYERS,
        )

        if self.output_attentions:
            self.config.use_pytorch_sdpa = False

        # print(self.config)

        self.model = AutoModelForCausalLM.from_config(
            trust_remote_code=True, 
            config=self.config,
            ).eval().cuda().bfloat16()
        
        self.load_model(self.args.ckpt_path)

    def get_model_parameters(self):
        model_params = { # NHIDDEN, FFN_HIDDEN, NLAYERS, NHEADS
            '170M': (768, -1, 24, 16),
            '470M': (1280, -1, 24, 20),
        }
        for key in model_params:
            if key in self.ckpt_path:
                res = model_params[key]
                if res[1] == -1:
                    res = (res[0], int(res[0] * 8 / 3), res[2], res[3])
                return res
        raise ValueError(f"Unknown model type: {self.ckpt_path}")

    def load_sequences(self, csv_path):
        sequences = []
        data_df = pd.read_csv(csv_path, sep=',')
        if 'unique_id' not in data_df.columns:
            raise Exception('Must give out unique id!')
        if 'nt_seq' in data_df.columns:
            data_df['raw_seq'] = data_df.nt_seq
            data_df['nt_seq'] = data_df.nt_seq.apply(lambda x: self.converter.convert(x))
        else:
            raise Exception('Must give out nt_seq!')

        max_len = data_df['raw_seq'].apply(len).max()
        # print('max_len', max_len)
        # padding length
        self.padding_length = min(self.padding_length+1, max_len)
        return data_df

    def load_model(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        self.model.transformer.load_state_dict(state_dict)
        return self.model

    def infer(self, total_df, rng_seed=0):
        genome_id_list = total_df['genome_id'].unique()
        gid_uid2path = {}
        for genome_id in genome_id_list:
            res_dir = os.path.join(self.save_dir, genome_id)
            os.makedirs(res_dir, exist_ok=True)

            data_df = total_df[total_df['genome_id']==genome_id]
            sequences = [{'nt_seq': s.replace('|', '').upper()[:self.max_seq_len], 'raw_seq': r,
                            'unique_id': uid, 'genome_id': gid} 
                            for r, s, uid, gid in zip(data_df['raw_seq'].to_list(),
                                            data_df['nt_seq'].to_list(),
                                            data_df['unique_id'].to_list(),
                                            data_df['genome_id'].to_list(),
                                            )]   # cut by max_seq_len

            cut_points = [0]
            seq_lens = []
            input_ids, input_ids_mask, position_ids, attention_masks, sequence_masks, seq_ids = np.empty((0, self.padding_length)), np.empty((0, self.padding_length)), np.empty((0, self.padding_length)),\
                                    np.empty((0, self.padding_length)), np.empty((0, self.padding_length)), []

            gene_emb_list = []
            # get masked input
            for index, seq_info in tqdm(enumerate(sequences)):
                seq_len = len(seq_info['nt_seq'])
                seq_lens.append(seq_len+1)

                input_id_raw, input_id_mask, position_id, attention_mask, sequence_mask = self.add_mask(seq_info['nt_seq'])
                cut_points.append(cut_points[-1] + input_id_raw.shape[0])
                input_ids = np.concatenate((input_ids, input_id_raw))
                input_ids_mask = np.concatenate((input_ids_mask, input_id_mask))
                position_ids = np.concatenate((position_ids, position_id))
                attention_masks = np.concatenate((attention_masks, attention_mask))
                sequence_masks = np.concatenate((sequence_masks, sequence_mask))
                seq_ids += [seq_info['unique_id']] * input_id_raw.shape[0]
                if input_ids.shape[0] >= self.batch_size:
                    while input_ids.shape[0] >= self.batch_size:
                        max_seq_len = max(seq_lens)
                        attention_masks[:, 1] = max_seq_len - attention_masks[:, 0]
                        # inference
                        gene_emb = self.cal_gene_embed(input_ids[:self.batch_size, :max_seq_len], input_ids_mask[:self.batch_size, :max_seq_len], position_ids[:self.batch_size, :max_seq_len], 
                                            attention_masks[:self.batch_size, :max_seq_len], sequence_masks[:self.batch_size, :max_seq_len],
                                            seq_ids[:self.batch_size], res_dir, seq_lens
                                            )
                        gene_emb_list.append(gene_emb)
                        # init
                        input_ids, input_ids_mask, position_ids, attention_masks, sequence_masks, seq_ids = input_ids[self.batch_size:, :], input_ids_mask[self.batch_size:, :], \
                                            position_ids[self.batch_size:, :], attention_masks[self.batch_size:, :], sequence_masks[self.batch_size:, :], seq_ids[self.batch_size:]
                        seq_lens = []

            if input_ids.shape[0] > 0:
                gene_emb = self.cal_gene_embed(input_ids, input_ids_mask, position_ids, attention_masks, sequence_masks, seq_ids, res_dir, seq_lens)
                gene_emb_list.append(gene_emb)

            gene_emb_total = torch.cat(gene_emb_list, dim=0)  
            species_emb, species_attn_score = self.model.transformer.species_pool(gene_emb_total) # [batch_size, channels]

            species_emb_npy = species_emb.float().detach().cpu().numpy() # batch_size * hidden_size
            save_path = os.path.join(res_dir, f'{genome_id}.npy')
            np.save(save_path, species_emb_npy)

            save_path = os.path.join(res_dir, f'{genome_id}.pkl')
            with open(save_path, 'wb') as file:
                pickle.dump({"paths":data_df['unique_id'].to_list(), "mean_feats": gene_emb_total}, file)

        total_df['seq_embed_path'] = total_df.apply(lambda x: os.path.join(self.save_dir, x['genome_id'], f'{x["unique_id"]}.npy'), axis=1)
        total_df['genome_embed_path'] = total_df.apply(lambda x: os.path.join(self.save_dir, x['genome_id'], f'{x["genome_id"]}.npy'), axis=1)
        del total_df['nt_seq']
        total_df = total_df.rename(columns={'raw_seq': 'nt_seq'})

        return total_df
    
    def add_mask(self, seq):
        # input_to_id = np.array([self.tokenizer.TokenToId(aa) for aa in seq], dtype=np.int64) # no end token here
        input_to_id = np.array([self.tokenizer.TokenToId(aa) for aa in seq] + [self.tokenizer.get_command("eos")], dtype=np.int64)

        sequence_mask = np.array([np.zeros(len(input_to_id), dtype=np.bool_)])

        position_id_raw = np.array(range(input_to_id.shape[0]), dtype=np.int64)
        attention_mask_raw = np.array([input_to_id.shape[0]], dtype=np.int64)
        # padding
        input_to_id_padded = np.pad(input_to_id, (0, self.padding_length-len(input_to_id)), 'constant', constant_values=self.tokenizer.get_vocab()['[pad]'])
        position_id_padded = np.pad(position_id_raw, (0, self.padding_length-len(input_to_id)), 'constant', constant_values=0)
        attention_mask_padded = np.array([input_to_id.shape[0], self.padding_length-len(input_to_id)], dtype=np.int64)
        sequence_mask_padded = np.pad(sequence_mask, ((0,0), (0, self.padding_length-len(input_to_id))), 'constant', constant_values=False)
        
        # broadcast to mask length 
        current_input_ids = np.tile(input_to_id_padded, (sequence_mask_padded.shape[0], 1))
        current_position_ids = np.tile(position_id_padded, (sequence_mask_padded.shape[0], 1))
        full_attention_mask = np.tile(np.concatenate((attention_mask_padded, np.zeros(current_input_ids.shape[1] - len(attention_mask_padded), dtype=int)))
                                      , (sequence_mask_padded.shape[0], 1))
        
        # mask
        masked_input_ids = current_input_ids.copy()
        for i, mask in enumerate(sequence_mask_padded):
            masked_input_ids[i, mask] = self.tokenizer.TokenToId('tMASK')
        
        return current_input_ids, masked_input_ids, current_position_ids, full_attention_mask, sequence_mask_padded


    def cal_gene_embed(self, current_input_ids, masked_input_ids, current_position_ids, full_attention_mask, sequence_mask, seq_ids, res_dir, seq_lens):
        input_ids = torch.LongTensor(masked_input_ids).cuda()
        position_ids = torch.LongTensor(current_position_ids).cuda()
        full_attention_mask = torch.LongTensor(full_attention_mask).cuda()

        with torch.no_grad():
            lm_output = self.model.transformer(input_ids=input_ids, position_ids=position_ids,full_attention_mask=full_attention_mask)
            last_hidden_state = lm_output['last_hidden_state'].permute(1, 0, 2)

            bs = last_hidden_state.shape[0]
            if bs != 1:
                seq_max_length = max(seq_lens)
                mask = torch.stack(
                    [
                        F.pad(
                            torch.zeros(leng-1, last_hidden_state.shape[-1], device=last_hidden_state.device), 
                            (0, 0, 0, seq_max_length-leng+1), 
                            "constant", 
                            1.0
                        ) for leng in seq_lens
                    ]
                ) == 1.0
                
                last_hidden_state = last_hidden_state.masked_fill(mask, 0.0)
                glm_emb = self.model.transformer.glm_transform(last_hidden_state) # [seq_len, batch_size, channels]
                gene_emb, gene_attn_score = self.model.transformer.pool(glm_emb, ~mask[:,:,0].unsqueeze(-1)) # [seq_len, batch_size, channels]
            else:
                glm_emb = self.model.transformer.glm_transform(last_hidden_state[:,:-1,:]) # [seq_len, batch_size, channels]
                gene_emb, gene_attn_score = self.model.transformer.pool(glm_emb) # [seq_len, batch_size, channels]

            # species_emb, species_attn_score = self.model.transformer.species_pool(gene_emb) # [batch_size, channels]

        # Get last hidden state mean
        if self.output_hidden_states:
            gene_emb_npy = gene_emb.float().detach().cpu().numpy() # batch_size * hidden_size
            for i in range(gene_emb_npy.shape[0]):
                save_path = os.path.join(res_dir, f'{seq_ids[i]}.npy')
                np.save(save_path, gene_emb_npy[i, :(seq_lens[i]-1)])
        return gene_emb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./res/demo.csv', help='sequences')
    parser.add_argument('--task_name', type=str, default='demo_470m', help='task name')
    parser.add_argument('--max_seq_len', type=int, default=2000, help='max sequence len')
    parser.add_argument('--model_path', type=str, default='./models/HuggingFace', help='model path')
    parser.add_argument('--ckpt_path', type=str, default='./weights/MicroGenomer_470M/model_states.pt', help='ckpt_path')
    parser.add_argument('--save_dir', type=str, default='./results', help='save dir')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--output_attentions', type=bool, default=False, help='output_attentions')
    parser.add_argument('--output_hidden_states', type=bool, default=True, help='output_hidden_states')
    args = parser.parse_args()
    
    return args 

# Example usage
if __name__ == "__main__":
    args = get_args()

    dna_inference = DNAInference(args)
    data_df = dna_inference.load_sequences(args.csv_path)
    results_df = dna_inference.infer(data_df)
    # print(results_d
    results_df.to_csv(os.path.join(args.save_dir, results_df['genome_id'][0], results_df['genome_id'][0] + f'.csv'))
