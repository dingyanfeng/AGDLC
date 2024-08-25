import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig

class SingleLSTMModel(nn.Module):
    def __init__(self, vocab_size, seq_length, emb_size, hidden_size, module_type):
        super(SingleLSTMModel, self).__init__()
        self.module_type = module_type
        if module_type == "LSTM":
            self.embedding = nn.Embedding(vocab_size, emb_size)
            self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
            self.lin = nn.Linear(seq_length * hidden_size, 4)
        elif module_type == "biLSTM":
            self.embedding = nn.Embedding(vocab_size, emb_size)
            self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
            self.lin = nn.Linear(2 * seq_length * hidden_size, 4)
        elif module_type == "xLSTM":
            xlstm_cfg = f""" 
            vocab_size: {vocab_size}
            mlstm_block:
              mlstm:
                conv1d_kernel_size: 4
                qkv_proj_blocksize: 4
                num_heads: 4
            slstm_block:
              slstm:
                backend: 'vanilla'
                num_heads: 4
                conv1d_kernel_size: 4
                bias_init: powerlaw_blockdependent
              feedforward:
                proj_factor: 1.3
                act_fn: gelu
            context_length: {seq_length}
            num_blocks: 2
            embedding_dim: {emb_size}
            slstm_at: [] #[1]
            """
            cfg = OmegaConf.create(xlstm_cfg)
            cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
            self.xlstm_stack = xLSTMLMModel(cfg)
            self.lin = nn.Linear(seq_length*vocab_size, 4)


    def forward(self, inp):
        if self.module_type == "xLSTM":
            lstm_out = self.xlstm_stack(inp)
        elif self.module_type == "LSTM" or self.module_type == "biLSTM":
            emb = self.embedding(inp)               # [batch_size, seq_length] => [batch_size, seq_length, emb_size]
            lstm_out, (hn, cn) = self.lstm(emb)     # [batch_size, seq_length, hidden_size]
        batch_size = lstm_out.size()[0]
        flat = lstm_out.reshape(batch_size, -1)    # [batch_size, seq_length*hidden_size]
        out = self.lin(flat)                    # [batch_size, vocab_size]
        logits = F.log_softmax(out, dim=1)
        return logits                     


class MultiLSTMModel(nn.Module):
    def __init__(self, skmer_list, vocab_sizes, emb_sizes, hidden_sizes, seq_length, module_type):
        super(MultiLSTMModel, self).__init__()
        self.skmer_list = skmer_list
        max_vocab = max(vocab_sizes.values())
        self.lstm_models = SingleLSTMModel(vocab_size=max_vocab, seq_length=len(skmer_list)*seq_length, emb_size=emb_sizes, hidden_size=hidden_sizes, module_type=module_type)

    def forward(self, inputs):
        value_list = [value for value in inputs.values()]
        concatenated_data = torch.cat(value_list, dim=1)
        logits = self.lstm_models(concatenated_data)
        return logits
