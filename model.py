import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_size=96, hidden_size=64, num_layers=1, output_size=32):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x)
        return self.out(r_out)