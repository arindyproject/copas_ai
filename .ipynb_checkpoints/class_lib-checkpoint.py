import torch
import torch.nn as nn

#========================================================================================================================
#                                     Model MULTI LABEL
#========================================================================================================================

#Model LSTM untuk multi-label classification.
#========================================================================================================================
class LSTM_MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2, bidirectional=False):
        """
        Model LSTM untuk multi-label classification.

        Args:
            input_size (int): Jumlah fitur input pada setiap timestep.
            hidden_size (int): Ukuran hidden state LSTM.
            num_layers (int): Jumlah lapisan LSTM.
            num_classes (int): Jumlah label output (multi-label).
            dropout (float): Dropout antar layer LSTM.
            bidirectional (bool): Jika True, gunakan bidirectional LSTM.
        """
        super(LSTM_MultiLabelClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # Faktor pengali jika bidirectional
        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        # Fully connected untuk klasifikasi multi-label
        self.fc = nn.Linear(lstm_output_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: tensor dengan shape (batch_size, seq_length, input_size)
        """
        # LSTM menghasilkan output dan (hidden, cell)
        out, _ = self.lstm(x)

        # Ambil output dari timestep terakhir
        out = out[:, -1, :]

        # Fully connected layer
        out = self.fc(out)

        # Aktivasi sigmoid untuk multi-label (tiap label independen)
        out = self.sigmoid(out)
        return out
#========================================================================================================================

#Model GRU untuk multi-label classification.
#========================================================================================================================
class GRU_MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2, bidirectional=False):
        """
        Model GRU untuk multi-label classification.

        Args:
            input_size (int): Jumlah fitur input pada setiap timestep.
            hidden_size (int): Ukuran hidden state GRU.
            num_layers (int): Jumlah lapisan GRU.
            num_classes (int): Jumlah label output (multi-label).
            dropout (float): Dropout antar layer GRU.
            bidirectional (bool): Jika True, gunakan bidirectional GRU.
        """
        super(GRU_MultiLabelClassifier, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        gru_output_size = hidden_size * (2 if bidirectional else 1)

        self.fc = nn.Linear(gru_output_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: tensor dengan shape (batch_size, seq_length, input_size)
        """
        out, _ = self.gru(x)

        # Ambil hidden state dari timestep terakhir
        out = out[:, -1, :]

        # Fully connected + sigmoid (multi-label)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
#========================================================================================================================

#Model RNN sederhana untuk multi-label classification.
#========================================================================================================================
class RNN_MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2, bidirectional=False):
        """
        Model RNN sederhana untuk multi-label classification.

        Args:
            input_size (int): Jumlah fitur input pada setiap timestep.
            hidden_size (int): Ukuran hidden state RNN.
            num_layers (int): Jumlah lapisan RNN.
            num_classes (int): Jumlah label output (multi-label).
            dropout (float): Dropout antar layer RNN.
            bidirectional (bool): Jika True, gunakan bidirectional RNN.
        """
        super(RNN_MultiLabelClassifier, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
            nonlinearity='tanh'  # bisa diganti 'relu' juga
        )

        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        self.fc = nn.Linear(rnn_output_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: tensor dengan shape (batch_size, seq_length, input_size)
        """
        # out: (batch_size, seq_length, hidden_size)
        out, _ = self.rnn(x)

        # Ambil output timestep terakhir
        out = out[:, -1, :]

        # Fully connected + sigmoid
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
#========================================================================================================================

#Model Multi-Label Classification dengan input 1D (bukan time series).
#========================================================================================================================
class MultiLabel1DClassifier(nn.Module):
    """
    Model Multi-Label Classification dengan input 1D (bukan time series).

    Args:
        input_dim (int): Jumlah fitur input.
        hidden_dims (list[int]): Ukuran layer tersembunyi. Contoh: [128, 64].
        output_dim (int): Jumlah label (kelas keluaran multi-label).
        dropout (float): Rasio dropout untuk regularisasi (default=0.2).
        activation (str): Jenis aktivasi ('relu', 'tanh', 'leakyrelu', default='relu').

    Input shape:
        (batch_size, input_dim)

    Output shape:
        (batch_size, output_dim)
        Nilai output berupa probabilitas antar 0 dan 1 (karena menggunakan sigmoid).
    """

    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=5, dropout=0.2, activation='relu'):
        super(MultiLabel1DClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # Pilih fungsi aktivasi
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation. Pilih: 'relu', 'tanh', atau 'leakyrelu'")

        # Hidden layers
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # karena multi-label classification

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass dari model.

        Args:
            x (torch.Tensor): Input tensor dengan bentuk (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor dengan bentuk (batch_size, output_dim)
        """
        return self.model(x)
#========================================================================================================================

#Model Multi-Label Classification dengan dua input (fitur ganda 2D, bukan time series).
#========================================================================================================================
class MultiLabel2DClassifier(nn.Module):
    """
    Model Multi-Label Classification dengan dua input (fitur ganda 2D, bukan time series).

    Args:
        input_dim_1 (int): Jumlah fitur untuk input pertama.
        input_dim_2 (int): Jumlah fitur untuk input kedua.
        hidden_dims_1 (list[int]): Ukuran hidden layer untuk input pertama. Contoh: [128, 64].
        hidden_dims_2 (list[int]): Ukuran hidden layer untuk input kedua. Contoh: [64, 32].
        combined_dims (list[int]): Ukuran hidden layer setelah dua input digabung.
        output_dim (int): Jumlah label output (multi-label).
        dropout (float): Rasio dropout untuk regularisasi (default=0.2).
        activation (str): Jenis aktivasi ('relu', 'tanh', 'leakyrelu', default='relu').

    Input shape:
        x1: (batch_size, input_dim_1)
        x2: (batch_size, input_dim_2)

    Output shape:
        (batch_size, output_dim)
        Nilai output berupa probabilitas antar 0 dan 1 (karena menggunakan sigmoid).
    """

    def __init__(self, 
                 input_dim_1, input_dim_2,
                 hidden_dims_1=[128, 64],
                 hidden_dims_2=[128, 64],
                 combined_dims=[64],
                 output_dim=5,
                 dropout=0.2,
                 activation='relu'):
        super(MultiLabel2DClassifier, self).__init__()

        # Pilih fungsi aktivasi
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation. Pilih: 'relu', 'tanh', atau 'leakyrelu'")

        # --- Sub-network untuk input 1 ---
        layers_1 = []
        prev_dim = input_dim_1
        for hdim in hidden_dims_1:
            layers_1.append(nn.Linear(prev_dim, hdim))
            layers_1.append(act_fn)
            layers_1.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net1 = nn.Sequential(*layers_1)

        # --- Sub-network untuk input 2 ---
        layers_2 = []
        prev_dim = input_dim_2
        for hdim in hidden_dims_2:
            layers_2.append(nn.Linear(prev_dim, hdim))
            layers_2.append(act_fn)
            layers_2.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net2 = nn.Sequential(*layers_2)

        # --- Network gabungan ---
        combined_input_dim = hidden_dims_1[-1] + hidden_dims_2[-1]
        layers_combined = []
        prev_dim = combined_input_dim
        for hdim in combined_dims:
            layers_combined.append(nn.Linear(prev_dim, hdim))
            layers_combined.append(act_fn)
            layers_combined.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers_combined.append(nn.Linear(prev_dim, output_dim))
        layers_combined.append(nn.Sigmoid())  # multi-label
        self.net_combined = nn.Sequential(*layers_combined)

    def forward(self, x1, x2):
        """
        Forward pass dari model.

        Args:
            x1 (torch.Tensor): Input pertama (batch_size, input_dim_1)
            x2 (torch.Tensor): Input kedua (batch_size, input_dim_2)
        
        Returns:
            torch.Tensor: Output tensor dengan bentuk (batch_size, output_dim)
        """
        out1 = self.net1(x1)
        out2 = self.net2(x2)
        combined = torch.cat((out1, out2), dim=1)
        return self.net_combined(combined)
#========================================================================================================================

#Model Multi-Label Classification dengan tiga input (fitur 1D, bukan time series).
#========================================================================================================================
class MultiLabel3DClassifier(nn.Module):
    """
    Model Multi-Label Classification dengan tiga input (fitur 1D, bukan time series).

    Args:
        input_dim_1 (int): Jumlah fitur untuk input pertama.
        input_dim_2 (int): Jumlah fitur untuk input kedua.
        input_dim_3 (int): Jumlah fitur untuk input ketiga.
        hidden_dims_1 (list[int]): Ukuran hidden layer untuk input pertama.
        hidden_dims_2 (list[int]): Ukuran hidden layer untuk input kedua.
        hidden_dims_3 (list[int]): Ukuran hidden layer untuk input ketiga.
        combined_dims (list[int]): Ukuran hidden layer setelah tiga input digabung.
        output_dim (int): Jumlah label output (multi-label).
        dropout (float): Rasio dropout untuk regularisasi (default=0.2).
        activation (str): Jenis aktivasi ('relu', 'tanh', 'leakyrelu', default='relu').

    Input shape:
        x1: (batch_size, input_dim_1)
        x2: (batch_size, input_dim_2)
        x3: (batch_size, input_dim_3)

    Output shape:
        (batch_size, output_dim)
        Nilai output berupa probabilitas antar 0 dan 1 (karena menggunakan sigmoid).
    """

    def __init__(self, 
                 input_dim_1, input_dim_2, input_dim_3,
                 hidden_dims_1=[128, 64],
                 hidden_dims_2=[128, 64],
                 hidden_dims_3=[128, 64],
                 combined_dims=[64],
                 output_dim=5,
                 dropout=0.2,
                 activation='relu'):
        super(MultiLabel3DClassifier, self).__init__()

        # Pilih fungsi aktivasi
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation. Pilih: 'relu', 'tanh', atau 'leakyrelu'")

        # --- Sub-network untuk input 1 ---
        layers_1 = []
        prev_dim = input_dim_1
        for hdim in hidden_dims_1:
            layers_1.append(nn.Linear(prev_dim, hdim))
            layers_1.append(act_fn)
            layers_1.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net1 = nn.Sequential(*layers_1)

        # --- Sub-network untuk input 2 ---
        layers_2 = []
        prev_dim = input_dim_2
        for hdim in hidden_dims_2:
            layers_2.append(nn.Linear(prev_dim, hdim))
            layers_2.append(act_fn)
            layers_2.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net2 = nn.Sequential(*layers_2)

        # --- Sub-network untuk input 3 ---
        layers_3 = []
        prev_dim = input_dim_3
        for hdim in hidden_dims_3:
            layers_3.append(nn.Linear(prev_dim, hdim))
            layers_3.append(act_fn)
            layers_3.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net3 = nn.Sequential(*layers_3)

        # --- Network gabungan ---
        combined_input_dim = hidden_dims_1[-1] + hidden_dims_2[-1] + hidden_dims_3[-1]
        layers_combined = []
        prev_dim = combined_input_dim
        for hdim in combined_dims:
            layers_combined.append(nn.Linear(prev_dim, hdim))
            layers_combined.append(act_fn)
            layers_combined.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers_combined.append(nn.Linear(prev_dim, output_dim))
        layers_combined.append(nn.Sigmoid())  # multi-label
        self.net_combined = nn.Sequential(*layers_combined)

    def forward(self, x1, x2, x3):
        """
        Forward pass dari model.

        Args:
            x1 (torch.Tensor): Input pertama (batch_size, input_dim_1)
            x2 (torch.Tensor): Input kedua (batch_size, input_dim_2)
            x3 (torch.Tensor): Input ketiga (batch_size, input_dim_3)
        
        Returns:
            torch.Tensor: Output tensor dengan bentuk (batch_size, output_dim)
        """
        out1 = self.net1(x1)
        out2 = self.net2(x2)
        out3 = self.net3(x3)

        combined = torch.cat((out1, out2, out3), dim=1)
        return self.net_combined(combined)
#========================================================================================================================


#========================================================================================================================
#                                     Model MULTI CLASS
#========================================================================================================================

# Model LSTM untuk multi-class classification.
#========================================================================================================================
class LSTM_MultiClassClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2, bidirectional=False):
        """
        Model LSTM untuk multi-class classification.

        Args:
            input_size (int): Jumlah fitur pada setiap timestep (misalnya panjang vektor fitur per waktu).
            hidden_size (int): Ukuran (dimensi) dari hidden state di dalam LSTM.
            num_layers (int): Jumlah layer LSTM yang ditumpuk (stacked).
            num_classes (int): Jumlah kelas output (setiap input hanya satu kelas benar).
            dropout (float, optional): Nilai dropout antar layer LSTM untuk mencegah overfitting. Default: 0.2.
            bidirectional (bool, optional): Jika True, gunakan LSTM dua arah (forward & backward). Default: False.
        """
        super(LSTM_MultiClassClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        self.fc = nn.Linear(lstm_output_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass dari model.

        Args:
            x (Tensor): Tensor input dengan shape (batch_size, seq_length, input_size)

        Returns:
            Tensor: Probabilitas prediksi untuk setiap kelas dengan shape (batch_size, num_classes)
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # Ambil output dari timestep terakhir
        out = self.fc(out)
        out = self.softmax(out)
        return out
#========================================================================================================================

#Model GRU untuk multi-class classification.
#========================================================================================================================
class GRU_MultiClassClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2, bidirectional=False):
        """
        Model GRU untuk multi-class classification.

        Args:
            input_size (int): Jumlah fitur pada setiap timestep.
            hidden_size (int): Ukuran hidden state dalam GRU.
            num_layers (int): Jumlah layer GRU yang ditumpuk.
            num_classes (int): Jumlah kelas output.
            dropout (float, optional): Dropout antar layer GRU. Default: 0.2.
            bidirectional (bool, optional): Jika True, gunakan GRU dua arah. Default: False.
        """
        super(GRU_MultiClassClassifier, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        gru_output_size = hidden_size * (2 if bidirectional else 1)

        self.fc = nn.Linear(gru_output_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass dari model GRU.

        Args:
            x (Tensor): Tensor input dengan shape (batch_size, seq_length, input_size)

        Returns:
            Tensor: Probabilitas prediksi kelas dengan shape (batch_size, num_classes)
        """
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Ambil timestep terakhir
        out = self.fc(out)
        out = self.softmax(out)
        return out
#========================================================================================================================

#Model RNN sederhana untuk multi-class classification.
#========================================================================================================================
class RNN_MultiClassClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2, bidirectional=False):
        """
        Model RNN sederhana untuk multi-class classification.

        Args:
            input_size (int): Jumlah fitur per timestep.
            hidden_size (int): Ukuran hidden state pada RNN.
            num_layers (int): Jumlah layer RNN.
            num_classes (int): Jumlah kelas output.
            dropout (float, optional): Dropout antar layer RNN. Default: 0.2.
            bidirectional (bool, optional): Jika True, gunakan RNN dua arah. Default: False.
        """
        super(RNN_MultiClassClassifier, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
            nonlinearity='tanh'  # bisa juga 'relu'
        )

        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        self.fc = nn.Linear(rnn_output_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass dari model RNN.

        Args:
            x (Tensor): Tensor input dengan shape (batch_size, seq_length, input_size)

        Returns:
            Tensor: Probabilitas prediksi kelas dengan shape (batch_size, num_classes)
        """
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.softmax(out)
        return out
#========================================================================================================================

#Model Multi-Class Classification dengan input 1D (fitur tunggal).
#========================================================================================================================
class MultiClass1DClassifier(nn.Module):
    """
    Model Multi-Class Classification dengan input 1D (fitur tunggal).

    Args:
        input_dim (int): Jumlah fitur input.
        hidden_dims (list[int]): Ukuran layer tersembunyi. Contoh: [128, 64].
        output_dim (int): Jumlah kelas (kategori output).
        dropout (float): Rasio dropout untuk regularisasi.
        activation (str): Jenis aktivasi ('relu', 'tanh', 'leakyrelu').

    Input shape:
        (batch_size, input_dim)

    Output shape:
        (batch_size, output_dim) — Probabilitas antar kelas (softmax).
    """

    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=5, dropout=0.2, activation='relu'):
        super(MultiClass1DClassifier, self).__init__()

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation: gunakan 'relu', 'tanh', atau 'leakyrelu'")

        layers = []
        prev_dim = input_dim

        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=1))  # multi-class output

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
#========================================================================================================================

#Model Multi-Class Classification dengan dua input 2D.
#========================================================================================================================
class MultiClass2DClassifier(nn.Module):
    """
    Model Multi-Class Classification dengan dua input 2D.

    Args:
        input_dim_1 (int): Jumlah fitur untuk input pertama.
        input_dim_2 (int): Jumlah fitur untuk input kedua.
        hidden_dims_1 (list[int]): Ukuran hidden layer untuk input pertama.
        hidden_dims_2 (list[int]): Ukuran hidden layer untuk input kedua.
        combined_dims (list[int]): Ukuran hidden layer setelah dua input digabung.
        output_dim (int): Jumlah kelas.
        dropout (float): Rasio dropout.
        activation (str): Jenis aktivasi ('relu', 'tanh', 'leakyrelu').

    Input:
        x1: (batch_size, input_dim_1)
        x2: (batch_size, input_dim_2)

    Output:
        (batch_size, output_dim)
    """

    def __init__(self,
                 input_dim_1, input_dim_2,
                 hidden_dims_1=[128, 64],
                 hidden_dims_2=[128, 64],
                 combined_dims=[64],
                 output_dim=5,
                 dropout=0.2,
                 activation='relu'):
        super(MultiClass2DClassifier, self).__init__()

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation")

        # Network untuk input 1
        layers_1 = []
        prev_dim = input_dim_1
        for hdim in hidden_dims_1:
            layers_1.append(nn.Linear(prev_dim, hdim))
            layers_1.append(act_fn)
            layers_1.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net1 = nn.Sequential(*layers_1)

        # Network untuk input 2
        layers_2 = []
        prev_dim = input_dim_2
        for hdim in hidden_dims_2:
            layers_2.append(nn.Linear(prev_dim, hdim))
            layers_2.append(act_fn)
            layers_2.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net2 = nn.Sequential(*layers_2)

        # Gabungan
        combined_input_dim = hidden_dims_1[-1] + hidden_dims_2[-1]
        layers_combined = []
        prev_dim = combined_input_dim
        for hdim in combined_dims:
            layers_combined.append(nn.Linear(prev_dim, hdim))
            layers_combined.append(act_fn)
            layers_combined.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers_combined.append(nn.Linear(prev_dim, output_dim))
        layers_combined.append(nn.Softmax(dim=1))  # multi-class

        self.net_combined = nn.Sequential(*layers_combined)

    def forward(self, x1, x2):
        out1 = self.net1(x1)
        out2 = self.net2(x2)
        combined = torch.cat((out1, out2), dim=1)
        return self.net_combined(combined)
#========================================================================================================================

#Model Multi-Class Classification dengan tiga input berbeda (fitur ganda 3D).
#========================================================================================================================
class MultiClass3DClassifier(nn.Module):
    """
    Model Multi-Class Classification dengan tiga input berbeda (fitur ganda 3D).

    Args:
        input_dim_1 (int): Jumlah fitur untuk input pertama.
        input_dim_2 (int): Jumlah fitur untuk input kedua.
        input_dim_3 (int): Jumlah fitur untuk input ketiga.
        hidden_dims_1 (list[int]): Ukuran hidden layer untuk input pertama.
        hidden_dims_2 (list[int]): Ukuran hidden layer untuk input kedua.
        hidden_dims_3 (list[int]): Ukuran hidden layer untuk input ketiga.
        combined_dims (list[int]): Ukuran hidden layer setelah tiga input digabung.
        output_dim (int): Jumlah kelas.
        dropout (float): Rasio dropout.
        activation (str): Jenis aktivasi ('relu', 'tanh', 'leakyrelu').

    Input:
        x1: (batch_size, input_dim_1)
        x2: (batch_size, input_dim_2)
        x3: (batch_size, input_dim_3)

    Output:
        (batch_size, output_dim)
    """

    def __init__(self, 
                 input_dim_1, input_dim_2, input_dim_3,
                 hidden_dims_1=[128, 64],
                 hidden_dims_2=[128, 64],
                 hidden_dims_3=[128, 64],
                 combined_dims=[64],
                 output_dim=5,
                 dropout=0.2,
                 activation='relu'):
        super(MultiClass3DClassifier, self).__init__()

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation")

        # Sub-network 1
        layers_1 = []
        prev_dim = input_dim_1
        for hdim in hidden_dims_1:
            layers_1.append(nn.Linear(prev_dim, hdim))
            layers_1.append(act_fn)
            layers_1.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net1 = nn.Sequential(*layers_1)

        # Sub-network 2
        layers_2 = []
        prev_dim = input_dim_2
        for hdim in hidden_dims_2:
            layers_2.append(nn.Linear(prev_dim, hdim))
            layers_2.append(act_fn)
            layers_2.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net2 = nn.Sequential(*layers_2)

        # Sub-network 3
        layers_3 = []
        prev_dim = input_dim_3
        for hdim in hidden_dims_3:
            layers_3.append(nn.Linear(prev_dim, hdim))
            layers_3.append(act_fn)
            layers_3.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net3 = nn.Sequential(*layers_3)

        # Gabungan
        combined_input_dim = hidden_dims_1[-1] + hidden_dims_2[-1] + hidden_dims_3[-1]
        layers_combined = []
        prev_dim = combined_input_dim
        for hdim in combined_dims:
            layers_combined.append(nn.Linear(prev_dim, hdim))
            layers_combined.append(act_fn)
            layers_combined.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers_combined.append(nn.Linear(prev_dim, output_dim))
        layers_combined.append(nn.Softmax(dim=1))  # multi-class

        self.net_combined = nn.Sequential(*layers_combined)

    def forward(self, x1, x2, x3):
        out1 = self.net1(x1)
        out2 = self.net2(x2)
        out3 = self.net3(x3)
        combined = torch.cat((out1, out2, out3), dim=1)
        return self.net_combined(combined)
#========================================================================================================================



#========================================================================================================================
#                                                Regression
#========================================================================================================================

#Model LSTM untuk regresi (memprediksi nilai kontinu).
#========================================================================================================================
class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2, bidirectional=False):
        """
        Model LSTM untuk regresi (memprediksi nilai kontinu).

        Args:
            input_size (int): Jumlah fitur pada setiap timestep.
            hidden_size (int): Ukuran hidden state dalam LSTM.
            num_layers (int): Jumlah layer LSTM.
            output_size (int, optional): Jumlah output regresi. Default: 1.
            dropout (float, optional): Dropout antar layer LSTM. Default: 0.2.
            bidirectional (bool, optional): Jika True, gunakan LSTM dua arah. Default: False.
        """
        super(LSTM_Regression, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, output_size)

    def forward(self, x):
        """
        Forward pass model LSTM regresi.

        Args:
            x (Tensor): Tensor input dengan shape (batch_size, seq_length, input_size)

        Returns:
            Tensor: Prediksi kontinu dengan shape (batch_size, output_size)
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # Ambil output timestep terakhir
        out = self.fc(out)
        return out
#========================================================================================================================

#Model GRU untuk regresi (prediksi nilai kontinu).
#========================================================================================================================
class GRU_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2, bidirectional=False):
        """
        Model GRU untuk regresi (prediksi nilai kontinu).

        Args:
            input_size (int): Jumlah fitur input per timestep.
            hidden_size (int): Ukuran hidden state GRU.
            num_layers (int): Jumlah layer GRU yang ditumpuk.
            output_size (int, optional): Jumlah nilai yang diprediksi. Default: 1.
            dropout (float, optional): Dropout antar layer GRU. Default: 0.2.
            bidirectional (bool, optional): Jika True, gunakan GRU dua arah. Default: False.
        """
        super(GRU_Regression, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_output_size, output_size)

    def forward(self, x):
        """
        Forward pass model GRU regresi.

        Args:
            x (Tensor): Tensor input (batch_size, seq_length, input_size)

        Returns:
            Tensor: Nilai prediksi (batch_size, output_size)
        """
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
#========================================================================================================================

#Model RNN sederhana untuk regresi.
#========================================================================================================================
class RNN_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2, bidirectional=False):
        """
        Model RNN sederhana untuk regresi.

        Args:
            input_size (int): Jumlah fitur per timestep.
            hidden_size (int): Ukuran hidden state RNN.
            num_layers (int): Jumlah layer RNN.
            output_size (int, optional): Jumlah output regresi. Default: 1.
            dropout (float, optional): Dropout antar layer RNN. Default: 0.2.
            bidirectional (bool, optional): Jika True, gunakan RNN dua arah. Default: False.
        """
        super(RNN_Regression, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
            nonlinearity='tanh'  # bisa diganti 'relu'
        )

        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_output_size, output_size)

    def forward(self, x):
        """
        Forward pass model RNN regresi.

        Args:
            x (Tensor): Tensor input (batch_size, seq_length, input_size)

        Returns:
            Tensor: Prediksi kontinu (batch_size, output_size)
        """
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
#========================================================================================================================

#Model Regression sederhana dengan satu input (fitur 1D).
#========================================================================================================================
class Model1DRegression(nn.Module):
    """
    Model Regression sederhana dengan satu input (fitur 1D).

    Args:
        input_dim (int): Jumlah fitur input.
        hidden_dims (list[int]): Ukuran layer tersembunyi. Contoh: [128, 64].
        output_dim (int): Jumlah nilai output (default=1).
        dropout (float): Rasio dropout.
        activation (str): Jenis aktivasi ('relu', 'tanh', 'leakyrelu').

    Input shape:
        (batch_size, input_dim)

    Output shape:
        (batch_size, output_dim)
    """

    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=1, dropout=0.2, activation='relu'):
        super(Model1DRegression, self).__init__()

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation")

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, output_dim))  # Linear output (tanpa aktivasi)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
#========================================================================================================================

#Model Regression dengan dua input fitur (fitur ganda 2D).
#========================================================================================================================
class Model2DRegression(nn.Module):
    """
    Model Regression dengan dua input fitur (fitur ganda 2D).

    Args:
        input_dim_1 (int): Jumlah fitur untuk input pertama.
        input_dim_2 (int): Jumlah fitur untuk input kedua.
        hidden_dims_1 (list[int]): Ukuran hidden layer untuk input pertama.
        hidden_dims_2 (list[int]): Ukuran hidden layer untuk input kedua.
        combined_dims (list[int]): Ukuran hidden layer setelah dua input digabung.
        output_dim (int): Jumlah nilai output (default=1).
        dropout (float): Rasio dropout.
        activation (str): Jenis aktivasi ('relu', 'tanh', 'leakyrelu').

    Input:
        x1: (batch_size, input_dim_1)
        x2: (batch_size, input_dim_2)

    Output:
        (batch_size, output_dim)
    """

    def __init__(self,
                 input_dim_1, input_dim_2,
                 hidden_dims_1=[128, 64],
                 hidden_dims_2=[128, 64],
                 combined_dims=[64],
                 output_dim=1,
                 dropout=0.2,
                 activation='relu'):
        super(Model2DRegression, self).__init__()

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation")

        # Sub-network 1
        layers_1 = []
        prev_dim = input_dim_1
        for hdim in hidden_dims_1:
            layers_1.append(nn.Linear(prev_dim, hdim))
            layers_1.append(act_fn)
            layers_1.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net1 = nn.Sequential(*layers_1)

        # Sub-network 2
        layers_2 = []
        prev_dim = input_dim_2
        for hdim in hidden_dims_2:
            layers_2.append(nn.Linear(prev_dim, hdim))
            layers_2.append(act_fn)
            layers_2.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net2 = nn.Sequential(*layers_2)

        # Gabungan
        combined_input_dim = hidden_dims_1[-1] + hidden_dims_2[-1]
        layers_combined = []
        prev_dim = combined_input_dim
        for hdim in combined_dims:
            layers_combined.append(nn.Linear(prev_dim, hdim))
            layers_combined.append(act_fn)
            layers_combined.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers_combined.append(nn.Linear(prev_dim, output_dim))
        self.net_combined = nn.Sequential(*layers_combined)

    def forward(self, x1, x2):
        out1 = self.net1(x1)
        out2 = self.net2(x2)
        combined = torch.cat((out1, out2), dim=1)
        return self.net_combined(combined)
#========================================================================================================================

#Model Regression dengan tiga input (fitur ganda 3D).
#========================================================================================================================
class Model3DRegression(nn.Module):
    """
    Model Regression dengan tiga input (fitur ganda 3D).

    Args:
        input_dim_1 (int): Jumlah fitur untuk input pertama.
        input_dim_2 (int): Jumlah fitur untuk input kedua.
        input_dim_3 (int): Jumlah fitur untuk input ketiga.
        hidden_dims_1 (list[int]): Ukuran hidden layer untuk input pertama.
        hidden_dims_2 (list[int]): Ukuran hidden layer untuk input kedua.
        hidden_dims_3 (list[int]): Ukuran hidden layer untuk input ketiga.
        combined_dims (list[int]): Ukuran hidden layer setelah tiga input digabung.
        output_dim (int): Jumlah nilai output (default=1).
        dropout (float): Rasio dropout.
        activation (str): Jenis aktivasi ('relu', 'tanh', 'leakyrelu').

    Input:
        x1: (batch_size, input_dim_1)
        x2: (batch_size, input_dim_2)
        x3: (batch_size, input_dim_3)

    Output:
        (batch_size, output_dim)
    """

    def __init__(self, 
                 input_dim_1, input_dim_2, input_dim_3,
                 hidden_dims_1=[128, 64],
                 hidden_dims_2=[128, 64],
                 hidden_dims_3=[128, 64],
                 combined_dims=[64],
                 output_dim=1,
                 dropout=0.2,
                 activation='relu'):
        super(Model3DRegression, self).__init__()

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation")

        # Sub-network 1
        layers_1 = []
        prev_dim = input_dim_1
        for hdim in hidden_dims_1:
            layers_1.append(nn.Linear(prev_dim, hdim))
            layers_1.append(act_fn)
            layers_1.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net1 = nn.Sequential(*layers_1)

        # Sub-network 2
        layers_2 = []
        prev_dim = input_dim_2
        for hdim in hidden_dims_2:
            layers_2.append(nn.Linear(prev_dim, hdim))
            layers_2.append(act_fn)
            layers_2.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net2 = nn.Sequential(*layers_2)

        # Sub-network 3
        layers_3 = []
        prev_dim = input_dim_3
        for hdim in hidden_dims_3:
            layers_3.append(nn.Linear(prev_dim, hdim))
            layers_3.append(act_fn)
            layers_3.append(nn.Dropout(dropout))
            prev_dim = hdim
        self.net3 = nn.Sequential(*layers_3)

        # Gabungan
        combined_input_dim = hidden_dims_1[-1] + hidden_dims_2[-1] + hidden_dims_3[-1]
        layers_combined = []
        prev_dim = combined_input_dim
        for hdim in combined_dims:
            layers_combined.append(nn.Linear(prev_dim, hdim))
            layers_combined.append(act_fn)
            layers_combined.append(nn.Dropout(dropout))
            prev_dim = hdim

        layers_combined.append(nn.Linear(prev_dim, output_dim))  # Linear output
        self.net_combined = nn.Sequential(*layers_combined)

    def forward(self, x1, x2, x3):
        out1 = self.net1(x1)
        out2 = self.net2(x2)
        out3 = self.net3(x3)
        combined = torch.cat((out1, out2, out3), dim=1)
        return self.net_combined(combined)
#========================================================================================================================
