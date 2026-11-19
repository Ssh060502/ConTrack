"""History representation pretraining, independent of the control adapter."""
import torch
from torch import nn


class HistoryEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, history=32, width=96, layers=2, heads=4, latent_dim=64):
        super().__init__()
        self.input = nn.Linear(2*state_dim+action_dim,width)
        self.position = nn.Parameter(torch.zeros(1,history,width))
        nn.init.normal_(self.position,std=.02)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=2*width, dropout=.1,
                                            activation='gelu',batch_first=True,norm_first=True)
        self.transformer = nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.latent = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,latent_dim),nn.Tanh())
        self.parameters_head = nn.Linear(latent_dim,2)
        self.dynamics_head = nn.Sequential(nn.Linear(latent_dim+state_dim+action_dim,128),nn.ELU(),nn.Linear(128,state_dim))

    def encode(self, history, lengths):
        size = history.shape[1]
        padding = torch.arange(size,device=history.device)[None,:] >= lengths[:,None]
        causal = torch.ones(size,size,device=history.device,dtype=torch.bool).triu(1)
        encoded = self.transformer(self.input(history)+self.position[:,:size],mask=causal,src_key_padding_mask=padding)
        return self.latent(encoded[torch.arange(len(lengths),device=history.device),lengths-1])

    def forward(self, history, lengths, current_state, current_action):
        z = self.encode(history,lengths)
        return self.parameters_head(z), self.dynamics_head(torch.cat([z,current_state,current_action],dim=-1)), z
