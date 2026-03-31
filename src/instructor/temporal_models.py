import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class SingleStageTCNModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 input_dim,
                 causal_conv=True):
        super(SingleStageTCNModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**layer_idx,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv,
                                     mstcn_dropout=0.5))
            for layer_idx in range(num_layers)
        ])

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        # Only return the last output
        output_last_ts = out[:, :, -1] 
        return output_last_ts


class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=3,
                 mstcn_dropout=0.5):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(mstcn_dropout)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


class DilatedSmoothLayer(nn.Module):
    def __init__(self, causal_conv=True):
        super(DilatedSmoothLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation1 = 1
        self.dilation2 = 5
        self.kernel_size = 5
        if self.causal_conv:
            self.conv_dilated1 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation1 * 2 * 2,
                                           dilation=self.dilation1)
            self.conv_dilated2 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation2 * 2 * 2,
                                           dilation=self.dilation2)

        else:
            self.conv_dilated1 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation1 * 2,
                                           dilation=self.dilation1)
            self.conv_dilated2 = nn.Conv1d(7,
                                           7,
                                           self.kernel_size,
                                           padding=self.dilation2 * 2,
                                           dilation=self.dilation2)
        self.conv_1x1 = nn.Conv1d(7, 7, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x1 = self.conv_dilated1(x)
        x1 = self.conv_dilated2(x1[:, :, :-4])
        out = F.relu(x1)
        if self.causal_conv:
            out = out[:, :, :-((self.dilation2 * 2) * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)