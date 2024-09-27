import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.use_revin = configs.use_revin

        self.period_len = configs.period_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * self.period_len // 2,
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)


    def forward(self, x):
        batch_size = x.shape[0]

        # normalization and permute     b,s,c -> b,c,s
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1).unsqueeze(1)
            x = (x - seq_mean).permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)

        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        y = self.linear(x)  # bc,w,m

        # bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        if self.use_revin:
            y = y.permute(0, 2, 1) + seq_mean
        else:
            y = y.permute(0, 2, 1)

        return y