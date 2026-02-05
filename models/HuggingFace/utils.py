import os
import math
import json  
import base64
import hashlib
import torch
from torch import nn
import torch.nn.functional as F


class Pooling(nn.Module):
    def __init__(self, pool_method='mean', hiddn_size=128):
        super().__init__()
        self.pool_method = pool_method
        self.hiddn_size = hiddn_size
        if 'attention' in pool_method:
            self.weight = nn.Parameter(torch.zeros((hiddn_size, 1)))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, mask = None):
        if mask is not None:
            x = x*mask

        attention_scores = torch.tensor(0.0,device=x.device,dtype=x.dtype)

        if self.pool_method == 'max':
            y = x.max(dim=1)[0]
        elif self.pool_method == 'sum':
            y = x.sum(dim=1)
        elif self.pool_method == 'mean':
            if mask is not None:
                y = x.sum(dim=1) / mask.squeeze().sum(-1).unsqueeze(-1)
            else:
                y = torch.mean(x, dim=1)
        elif self.pool_method == 'attention':
            attention_scores = torch.matmul(x, self.weight)
            if mask is not None:
                attention_scores = attention_scores.masked_fill(~mask, float('-inf'))  
            attention_weights = F.softmax(attention_scores, dim=1)
            y = torch.sum(attention_weights.repeat(1,1,self.hiddn_size) * x, dim=1)
        else:
            raise NotImplementedError(f"Unsupport pooling method: {self.pool_method}")

        return y, attention_scores


class ConvLayer(nn.Module):
      def __init__(self, hidden_size, out_dim):
            super().__init__()
            # padding=(kernel_size-1)/2
            self.conv3 = nn.Conv1d(hidden_size, 256, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv1d(hidden_size, 128, kernel_size=5, stride=1, padding=2)
            self.conv7 = nn.Conv1d(hidden_size, 128, kernel_size=7, stride=1, padding=3)
            self.activation = nn.GELU()
            self.glm_transform = nn.Linear(hidden_size, out_dim)
            self.conv_transform = nn.Linear(512, out_dim)

      def forward(self, x):
            """Conv forward pass.
            Args:
            x: Input tensor of shape (batch_size, seq_len, channels)
            Returns:
            Output tensor of shape (batch_size, seq_len, channels)
            """

            x = x.transpose(1, 2) # [batch_size, channels, seq_len]

            out3 = self.activation(self.conv3(x))
            out5 = self.activation(self.conv5(x))  
            out7 = self.activation(self.conv7(x)) 
            out = torch.cat([out3, out5, out7], dim=1)

            x = x.transpose(1, 2)
            x = self.glm_transform(x)

            out = out.transpose(1, 2)
            out = self.conv_transform(out)

            return x + out