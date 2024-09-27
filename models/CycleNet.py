import torch
import torch.nn as nn

class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        # roll the queue tensor to the appropriate index
        rolled_data = torch.stack([torch.roll(self.data, shifts=(-i.item(), 0), dims=(0, 1)) for i in index])

        outputs = []
        for i in range(rolled_data.size(0)):
            if length <= self.cycle_len:
                output = rolled_data[i, :length]
            else:
                # repeat the queue data to match the required length
                num_repeats = length // self.cycle_len
                remainder = length % self.cycle_len
                output = torch.cat([rolled_data[i]] * num_repeats + [rolled_data[i, :remainder]])
            outputs.append(output)

        return torch.stack(outputs)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, x, cycle_index):
        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # remove the cycle of the input data
        x = x - self.cycleQueue(cycle_index, self.seq_len)

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y