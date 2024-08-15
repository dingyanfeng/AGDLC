import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleLSTMModel(nn.Module):
    def __init__(self, vocab_size, seq_length, emb_size, hidden_size, module_type):
        super(SingleLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        if module_type == "LSTM":
            self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, batch_first=True)
            self.lin = nn.Linear(seq_length * hidden_size, 4)
        elif module_type == "biLSTM":
            self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
            self.lin = nn.Linear(2 * seq_length * hidden_size, 4)

    def forward(self, inp):
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
        self.total_vocab = sum(vocab_sizes.values())
        self.lstm_models = nn.ModuleList([SingleLSTMModel(vocab_size=vocab_sizes[i], seq_length=seq_length, emb_size=emb_sizes[i], hidden_size=hidden_sizes[i], module_type=module_type) for i in skmer_list])
        self.weights = nn.Parameter(torch.ones(len(skmer_list)) / torch.tensor(skmer_list, dtype=torch.float32), requires_grad=True)

    def forward(self, inputs):
        lstm_outputs = {}
        for index, i in enumerate(self.skmer_list):
            lstm_out = self.lstm_models[index](inputs[i])
            lstm_outputs[i]=lstm_out
        # logits_mix = sum(lstm_outputs.values()) / len(lstm_outputs)
        normalized_weights = nn.functional.softmax(self.weights, dim=0)
        logits_mix = sum(normalized_weights[i] * value for i, value in enumerate(lstm_outputs.values()))
        return logits_mix
