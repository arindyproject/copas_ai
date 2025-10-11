class LSTM_MultiLabel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # ambil hidden terakhir
        x = self.dropout(h_last)
        x = torch.sigmoid(self.fc(x))  # multi-label
        return x
