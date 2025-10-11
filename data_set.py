import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

#========================================================================================================================
#                                     Dataset MULTI LABEL
#========================================================================================================================

#Dataset PyTorch untuk multi-label classification berbasis LSTM/time series.
#========================================================================================================================
class MultiLabelTimeSeriesDataset(Dataset):
    """
        Dataset PyTorch untuk multi-label classification berbasis LSTM/RNN/GRU timeseries.

        Args:
            df (pd.DataFrame): Data dalam bentuk DataFrame.
            seq_length (int): Panjang sequence input.
            slide (int): Langkah geser antar sequence.
            feature_cols (list[str]): Nama kolom yang digunakan sebagai fitur.
            target_cols (list[str]): Nama kolom yang digunakan sebagai target.
            device (torch.device | str | None): Device untuk tensor ('cuda' atau 'cpu').
        
        Contoh data:
            data = {
                "temp": [30, 32, 33, 35, 36, 34, 33, 31, 30, 29, 28],
                "humidity": [60, 61, 63, 65, 67, 68, 66, 64, 62, 61, 60],
                "label1": [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                "label2": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            }
            df = pd.DataFrame(data)

            # Tentukan fitur dan target secara eksplisit
            feature_cols = ["temp", "humidity"]
            target_cols = ["label1", "label2"]

            # Buat dataset
            dataset = MultiLabelTimeSeriesDataset(df, seq_length=3, slide=1,
                                                feature_cols=feature_cols,
                                                target_cols=target_cols)

            # Cek hasil
            loader = DataLoader(dataset, batch_size=2)
            for X, y in loader:
                print("X:", X.shape)  # (batch, seq_len, num_features)
                print("y:", y.shape)  # (batch, num_labels)
                break
        """
    def __init__(self, df, seq_length=10, slide=1, feature_cols=None, target_cols=None):
        if feature_cols is None or target_cols is None:
            raise ValueError("feature_cols dan target_cols harus ditentukan sebagai list nama kolom.")

        self.seq_length = seq_length
        self.slide = slide
        self.device = torch.device(device if device else "cpu")

        # Ambil data fitur dan target
        X = df[feature_cols].values.astype(float)
        y = df[target_cols].values.astype(float)

        # Normalisasi fitur ke [0, 1] jika diminta
        if normalize:
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
            X = (X - X_min) / (X_max - X_min + 1e-8)

        # Simpan sequence dan target
        sequences = []
        targets = []

        for i in range(0, len(df) - seq_length, slide):
            X_seq = X[i:i + seq_length]
            y_next = y[i + seq_length - 1]  # target dari akhir sequence
            sequences.append(X_seq)
            targets.append(y_next)

        # Konversi ke tensor dan langsung kirim ke device
        self.sequences = torch.tensor(sequences, dtype=torch.float32, device=self.device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
#========================================================================================================================

# Dataset untuk multi-label classification 1D.
#========================================================================================================================
class MultiLabel1DDataset(Dataset):
    """
        Dataset untuk multi-label classification 1D.

        Args:
            df (pd.DataFrame): Data input.
            feature_cols (list[str]): Nama kolom fitur.
            target_cols (list[str]): Nama kolom target (multi-label).
            normalize (bool): Jika True, fitur akan dinormalisasi ke [0, 1].
            device (torch.device | str | None): Device untuk tensor ('cuda' atau 'cpu').

        Contoh data:
            data = {
                "umur": [20, 25, 30, 35, 40, 45],
                "berat": [60, 70, 80, 85, 90, 95],
                "tinggi": [160, 165, 170, 175, 180, 185],
                "label_gigi": [1, 0, 0, 0, 1, 0],
                "label_jantung": [0, 1, 1, 0, 1, 1],
                "label_mata": [0, 0, 0, 1, 0, 0]
            }
            df = pd.DataFrame(data)

            # Tentukan kolom fitur dan target
            feature_cols = ["umur", "berat", "tinggi"]
            target_cols = ["label_gigi", "label_jantung", "label_mata"]

            # Buat dataset
            dataset = MultiLabel1D(df, feature_cols, target_cols, normalize=True)

            # Buat DataLoader
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

            # Cek contoh batch
            for X, y in dataloader:
                print("X shape:", X.shape)  # (batch, num_features)
                print("y shape:", y.shape)  # (batch, num_labels)
                print(X)
                print(y)
                break
        """
    def __init__(self, df, feature_cols, target_cols, normalize=True):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.device = torch.device(device if device else "cpu")

        X = df[feature_cols].values.astype(float)
        y = df[target_cols].values.astype(float)

        if normalize:
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
            X = (X - X_min) / (X_max - X_min + 1e-8)

        # Simpan sebagai tensor dan langsung kirim ke device
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
#========================================================================================================================

#Dataset untuk multi-label classification dengan 2 input (fitur ganda).
#========================================================================================================================
class MultiLabel2DDataset(Dataset):
    """
        Dataset untuk multi-label classification dengan 2 input (fitur ganda).

        Args:
            df (pd.DataFrame): Data input.
            feature1_cols (list[str]): Nama kolom untuk input pertama.
            feature2_cols (list[str]): Nama kolom untuk input kedua.
            target_cols (list[str]): Nama kolom target (multi-label).
            normalize (bool): Jika True, fitur akan dinormalisasi ke [0, 1].
            device (torch.device | str | None): Device untuk tensor ('cuda' atau 'cpu').

        Contoh data:
            data = {
                "umur": [20, 25, 30, 35, 40, 45],
                "berat": [60, 70, 80, 85, 90, 95],
                "tinggi": [160, 165, 170, 175, 180, 185],
                "imt": [23.4, 25.7, 27.6, 28.0, 29.1, 30.5],
                "label_gigi": [1, 0, 0, 1, 0, 0],
                "label_jantung": [0, 1, 1, 0, 1, 1],
                "label_mata": [0, 0, 1, 0, 0, 1]
            }

            df = pd.DataFrame(data)

            # Input 1: fitur biologis dasar
            feature1_cols = ["umur", "berat", "tinggi"]

            # Input 2: fitur turunan (misalnya IMT)
            feature2_cols = ["imt"]

            # Target multi-label
            target_cols = ["label_gigi", "label_jantung", "label_mata"]

            # Buat dataset
            dataset = MultiLabel2D(df, feature1_cols, feature2_cols, target_cols)

            # Buat DataLoader
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

            # Cek batch
            for X1, X2, y in dataloader:
                print("X1 shape:", X1.shape)  # (batch, num_features_1)
                print("X2 shape:", X2.shape)  # (batch, num_features_2)
                print("y shape:", y.shape)    # (batch, num_labels)
                print()
                break
        """    
    def __init__(self, df, feature1_cols, feature2_cols, target_cols, normalize=True):
        self.feature1_cols = feature1_cols
        self.feature2_cols = feature2_cols
        self.target_cols = target_cols
        self.device = torch.device(device if device else "cpu")

        X1 = df[feature1_cols].values.astype(float)
        X2 = df[feature2_cols].values.astype(float)
        y = df[target_cols].values.astype(float)

        if normalize:
            X1 = self._normalize(X1)
            X2 = self._normalize(X2)

        # Simpan sebagai tensor dan langsung kirim ke device
        self.X1 = torch.tensor(X1, dtype=torch.float32, device=self.device)
        self.X2 = torch.tensor(X2, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)

    def _normalize(self, X):
        """Normalisasi data ke rentang [0, 1]."""
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        return (X - X_min) / (X_max - X_min + 1e-8)

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]
#========================================================================================================================

#Dataset untuk multi-label classification dengan 3 input (fitur ganda 3 sumber).
#========================================================================================================================
class MultiLabel3DDataset(Dataset):
    """
        Dataset untuk multi-label classification dengan 3 input (fitur ganda 3 sumber).

        Args:
            df (pd.DataFrame): Data input.
            feature1_cols (list[str]): Nama kolom untuk input pertama.
            feature2_cols (list[str]): Nama kolom untuk input kedua.
            feature3_cols (list[str]): Nama kolom untuk input ketiga.
            target_cols (list[str]): Nama kolom target (multi-label).
            normalize (bool): Jika True, fitur akan dinormalisasi ke [0, 1].
            device (torch.device | str | None): Device untuk tensor ('cuda' atau 'cpu').

        Contoh data:
            data = {
                # Fitur kelompok 1 (fisik)
                "umur": [20, 25, 30, 35, 40, 45],
                "berat": [60, 70, 80, 85, 90, 95],
                "tinggi": [160, 165, 170, 175, 180, 185],

                # Fitur kelompok 2 (medis)
                "imt": [23.4, 25.7, 27.6, 28.0, 29.1, 30.5],
                "tekanan_darah": [120, 125, 130, 140, 145, 150],

                # Fitur kelompok 3 (teks / embedding / skor)
                "skor_keluhan": [0.3, 0.6, 0.9, 0.4, 0.7, 0.2],
                "skor_kondisi": [0.1, 0.8, 0.5, 0.6, 0.9, 0.3],

                # Target multi-label
                "label_gigi": [1, 0, 0, 1, 0, 0],
                "label_jantung": [0, 1, 1, 0, 1, 1],
                "label_mata": [0, 0, 1, 0, 0, 1]
            }

            df = pd.DataFrame(data)

            # Tentukan kolom fitur dan target
            feature1_cols = ["umur", "berat", "tinggi"]
            feature2_cols = ["imt", "tekanan_darah"]
            feature3_cols = ["skor_keluhan", "skor_kondisi"]
            target_cols = ["label_gigi", "label_jantung", "label_mata"]

            # Buat dataset
            dataset = MultiLabel3D(df, feature1_cols, feature2_cols, feature3_cols, target_cols)

            # Buat DataLoader
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

            # Cek batch
            for X1, X2, X3, y in dataloader:
                print("X1 shape:", X1.shape)  # (batch, fitur1)
                print("X2 shape:", X2.shape)  # (batch, fitur2)
                print("X3 shape:", X3.shape)  # (batch, fitur3)
                print("y shape:", y.shape)    # (batch, num_labels)
                print()
                break
    """
    def __init__(self, df, feature1_cols, feature2_cols, feature3_cols, target_cols, normalize=True, device=None):
        self.feature1_cols = feature1_cols
        self.feature2_cols = feature2_cols
        self.feature3_cols = feature3_cols
        self.target_cols = target_cols

        self.device = torch.device(device if device else "cpu")

        X1 = df[feature1_cols].values.astype(float)
        X2 = df[feature2_cols].values.astype(float)
        X3 = df[feature3_cols].values.astype(float)
        y = df[target_cols].values.astype(float)

        if normalize:
            X1 = self._normalize(X1)
            X2 = self._normalize(X2)
            X3 = self._normalize(X3)

        # Simpan sebagai tensor dan kirim ke device
        self.X1 = torch.tensor(X1, dtype=torch.float32, device=self.device)
        self.X2 = torch.tensor(X2, dtype=torch.float32, device=self.device)
        self.X3 = torch.tensor(X3, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)

    def _normalize(self, X):
        """Normalisasi data ke rentang [0, 1]."""
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        return (X - X_min) / (X_max - X_min + 1e-8)

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.X3[idx], self.y[idx]
#========================================================================================================================


#========================================================================================================================
#                                     Dataset MULTI CLASS
#========================================================================================================================

#Dataset PyTorch untuk multi-class classification berbasis time series / LSTM.
#========================================================================================================================
class MultiClassTimeSeriesDataset(Dataset):
    """
    Dataset PyTorch untuk multi-class classification berbasis time series / LSTM.
    
    Args:
        df (pd.DataFrame): Data sumber.
        feature_stems (list[str]): List nama dasar kolom fitur (misal ['open', 'high', 'low', 'close']).
        target_stem (str): Nama dasar kolom target (misal 'label').
        seq_length (int): Panjang sequence (misal 10).
        slide (int): Pergeseran antar sequence (misal 1).
        device (torch.device, optional): Tempat menyimpan tensor (default: CPU).
    
    Contoh:
        data = {
            "open": np.random.rand(100),
            "high": np.random.rand(100),
            "low": np.random.rand(100),
            "close": np.random.rand(100),
            "label": np.random.randint(0, 3, size=100)  # 3 kelas
        }
        df = pd.DataFrame(data)

        dataset = MultiClassTimeSeriesDataset(
            df,
            feature_stems=["open", "high", "low", "close"],
            target_stem="label",
            seq_length=10,
            slide=1
        )

        x, y = dataset[0]
        print("X shape:", x.shape)  # (seq_length, num_features)
        print("Y:", y)
    """
    def __init__(self, df, feature_stems, target_stem, seq_length=10, slide=1, device=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_stems = feature_stems
        self.target_stem = target_stem
        self.seq_length = seq_length
        self.slide = slide
        self.device = device or torch.device("cpu")

        # Pastikan semua kolom fitur ada
        for stem in self.feature_stems:
            if stem not in df.columns:
                raise ValueError(f"Kolom fitur '{stem}' tidak ditemukan di DataFrame.")

        if target_stem not in df.columns:
            raise ValueError(f"Kolom target '{target_stem}' tidak ditemukan di DataFrame.")

        self.features = df[self.feature_stems].values
        self.targets = df[self.target_stem].values

        # Total banyak sequence
        self.indices = [
            i for i in range(0, len(self.features) - seq_length, slide)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.features[i:i + self.seq_length]
        y = self.targets[i + self.seq_length]  # Target setelah sequence
        
        # Tensor
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)  # class index

        return x_tensor, y_tensor
#========================================================================================================================

#Dataset PyTorch untuk multi-class classification berbasis data 1D (bukan time series).
#========================================================================================================================
class MultiClass1DDataset(Dataset):
    """
    Dataset untuk multi-class classification 1D.

    Args:
        df (pd.DataFrame): Data input.
        feature_cols (list[str]): Nama kolom fitur.
        target_col (str): Nama kolom target (kelas tunggal, bukan multi-label).
        normalize (bool): Jika True, fitur akan dinormalisasi ke [0, 1].
        device (torch.device | str | None): Device untuk tensor ('cuda' atau 'cpu').

    Contoh data:
        data = {
            "umur": [20, 25, 30, 35, 40, 45],
            "berat": [60, 70, 80, 85, 90, 95],
            "tinggi": [160, 165, 170, 175, 180, 185],
            "kelas": [0, 1, 2, 1, 0, 2]
        }
        df = pd.DataFrame(data)

        # Tentukan kolom fitur dan target
        feature_cols = ["umur", "berat", "tinggi"]
        target_col = "kelas"

        # Buat dataset
        dataset = MultiClass1D(df, feature_cols, target_col, normalize=True)

        # Buat DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Cek contoh batch
        for X, y in dataloader:
            print("X shape:", X.shape)  # (batch, num_features)
            print("y shape:", y.shape)  # (batch,)
            print(X)
            print(y)
            break
    """
    def __init__(self, df, feature_cols, target_col, normalize=True, device=None):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.device = torch.device(device if device else "cpu")

        X = df[feature_cols].values.astype(float)
        y = df[target_col].values.astype(int)  # target berupa integer (kelas)

        if normalize:
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
            X = (X - X_min) / (X_max - X_min + 1e-8)

        # Simpan tensor ke device
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.long, device=self.device)  # long untuk CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
#========================================================================================================================

#Dataset PyTorch untuk multi-class classification berbasis 2D (time series).
#========================================================================================================================
class MultiClass2DDataset(Dataset):
    """
    Dataset untuk multi-class classification dengan 2 input (fitur ganda).

    Args:
        df (pd.DataFrame): Data input.
        feature1_cols (list[str]): Nama kolom untuk input pertama.
        feature2_cols (list[str]): Nama kolom untuk input kedua.
        target_col (str): Nama kolom target (kelas tunggal).
        normalize (bool): Jika True, fitur akan dinormalisasi ke [0, 1].
        device (torch.device | str | None): Device untuk tensor ('cuda' atau 'cpu').

    Contoh data:
        data = {
            "umur": [20, 25, 30, 35, 40, 45],
            "berat": [60, 70, 80, 85, 90, 95],
            "tinggi": [160, 165, 170, 175, 180, 185],
            "imt": [23.4, 25.7, 27.6, 28.0, 29.1, 30.5],
            "kelas": [0, 1, 2, 1, 0, 2]
        }

        df = pd.DataFrame(data)

        feature1_cols = ["umur", "berat", "tinggi"]
        feature2_cols = ["imt"]
        target_col = "kelas"

        dataset = MultiClass2DDataset(df, feature1_cols, feature2_cols, target_col)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        for X1, X2, y in dataloader:
            print("X1 shape:", X1.shape)  # (batch, num_features_1)
            print("X2 shape:", X2.shape)  # (batch, num_features_2)
            print("y shape:", y.shape)    # (batch,)
            print()
            break
    """
    def __init__(self, df, feature1_cols, feature2_cols, target_col, normalize=True, device=None):
        self.feature1_cols = feature1_cols
        self.feature2_cols = feature2_cols
        self.target_col = target_col
        self.device = torch.device(device if device else "cpu")

        X1 = df[feature1_cols].values.astype(float)
        X2 = df[feature2_cols].values.astype(float)
        y = df[target_col].values.astype(int)  # target berupa kelas integer

        if normalize:
            X1 = self._normalize(X1)
            X2 = self._normalize(X2)

        # Simpan tensor ke device
        self.X1 = torch.tensor(X1, dtype=torch.float32, device=self.device)
        self.X2 = torch.tensor(X2, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.long, device=self.device)  # long untuk CrossEntropyLoss

    def _normalize(self, X):
        """Normalisasi data ke rentang [0, 1]."""
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        return (X - X_min) / (X_max - X_min + 1e-8)

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]
#========================================================================================================================

#Dataset untuk multi-class classification dengan 3 input (fitur dari 3 sumber).
#========================================================================================================================
class MultiClass3DDataset(Dataset):
    """
    Dataset untuk multi-class classification dengan 3 input (fitur dari 3 sumber).

    Args:
        df (pd.DataFrame): Data input.
        feature1_cols (list[str]): Nama kolom untuk input pertama.
        feature2_cols (list[str]): Nama kolom untuk input kedua.
        feature3_cols (list[str]): Nama kolom untuk input ketiga.
        target_col (str): Nama kolom target (kelas tunggal).
        normalize (bool): Jika True, fitur akan dinormalisasi ke [0, 1].
        device (torch.device | str | None): Device untuk tensor ('cuda' atau 'cpu').

    Contoh penggunaan:
        data = {
            # Fitur kelompok 1 (fisik)
            "umur": [20, 25, 30, 35, 40, 45],
            "berat": [60, 70, 80, 85, 90, 95],
            "tinggi": [160, 165, 170, 175, 180, 185],

            # Fitur kelompok 2 (medis)
            "imt": [23.4, 25.7, 27.6, 28.0, 29.1, 30.5],
            "tekanan_darah": [120, 125, 130, 140, 145, 150],

            # Fitur kelompok 3 (teks / embedding / skor)
            "skor_keluhan": [0.3, 0.6, 0.9, 0.4, 0.7, 0.2],
            "skor_kondisi": [0.1, 0.8, 0.5, 0.6, 0.9, 0.3],

            # Target multi-class
            "kelas": [0, 1, 2, 1, 0, 2]
        }

        df = pd.DataFrame(data)

        feature1_cols = ["umur", "berat", "tinggi"]
        feature2_cols = ["imt", "tekanan_darah"]
        feature3_cols = ["skor_keluhan", "skor_kondisi"]
        target_col = "kelas"

        dataset = MultiClass3DDataset(df, feature1_cols, feature2_cols, feature3_cols, target_col)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        for X1, X2, X3, y in dataloader:
            print("X1 shape:", X1.shape)
            print("X2 shape:", X2.shape)
            print("X3 shape:", X3.shape)
            print("y shape:", y.shape)
            print()
            break
    """
    def __init__(self, df, feature1_cols, feature2_cols, feature3_cols, target_col, normalize=True, device=None):
        self.feature1_cols = feature1_cols
        self.feature2_cols = feature2_cols
        self.feature3_cols = feature3_cols
        self.target_col = target_col

        self.device = torch.device(device if device else "cpu")

        # Ambil data dari DataFrame
        X1 = df[feature1_cols].values.astype(float)
        X2 = df[feature2_cols].values.astype(float)
        X3 = df[feature3_cols].values.astype(float)
        y = df[target_col].values.astype(int)  # kelas tunggal (integer)

        # Normalisasi ke rentang [0, 1] jika diaktifkan
        if normalize:
            X1 = self._normalize(X1)
            X2 = self._normalize(X2)
            X3 = self._normalize(X3)

        # Simpan sebagai tensor ke device
        self.X1 = torch.tensor(X1, dtype=torch.float32, device=self.device)
        self.X2 = torch.tensor(X2, dtype=torch.float32, device=self.device)
        self.X3 = torch.tensor(X3, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.long, device=self.device)  # long untuk CrossEntropyLoss

    def _normalize(self, X):
        """Normalisasi data ke rentang [0, 1]."""
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        return (X - X_min) / (X_max - X_min + 1e-8)

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.X3[idx], self.y[idx]
#========================================================================================================================


#========================================================================================================================
#                                                Regression
#========================================================================================================================

# Dataset PyTorch untuk regresi berbasis time series (LSTM/GRU/Transformer).
#========================================================================================================================
class RegressionDatasetTimeSeries(Dataset):
    def __init__(self, df, seq_length=10, slide=1, feature_cols=None, target_col=None):
        """
        Dataset PyTorch untuk regresi berbasis time series (LSTM/GRU/Transformer).
        
        Args:
            df (pd.DataFrame): Data dalam bentuk DataFrame.
            seq_length (int): Panjang sequence input (misal 10 berarti 10 langkah ke belakang).
            slide (int): Langkah geser antar sequence (default 1).
            feature_cols (list): Kolom yang digunakan sebagai fitur input.
            target_col (str): Kolom target yang akan diprediksi.
        
        Contoh:
            data = {
                "Open": [10, 11, 12, 13, 14, 15, 16],
                "High": [11, 12, 13, 14, 15, 16, 17],
                "Low": [9, 10, 11, 12, 13, 14, 15],
                "Close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5]
            }
            df = pd.DataFrame(data)

            # Membuat dataset
            dataset = RegressionDatasetTimeSeries(df, seq_length=3, feature_cols=["Open", "High", "Low"], target_col="Close")

            # Contoh output
            print("Total sample:", len(dataset))
            X, y = dataset[0]
            print("X shape:", X.shape)   # (3, 3) -> 3 langkah x 3 fitur
            print("y:", y)        
        """
        self.seq_length = seq_length
        self.slide = slide
        
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Konversi ke numpy untuk efisiensi
        data = df[feature_cols].values
        targets = df[target_col].values
        
        self.X, self.y = [], []
        
        for i in range(0, len(df) - seq_length, slide):
            seq_x = data[i:i+seq_length]
            seq_y = targets[i+seq_length]  # target langkah berikutnya
            self.X.append(seq_x)
            self.y.append(seq_y)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
#========================================================================================================================

# Dataset untuk regresi 1D (fitur tabular).
#========================================================================================================================
class Regression1DDataset(Dataset):
    """
    Dataset untuk regresi 1D (fitur tabular).

    Args:
        df (pd.DataFrame): Data input.
        feature_cols (list[str]): Nama kolom fitur.
        target_col (str): Nama kolom target (nilai kontinu).
        normalize (bool): Jika True, fitur akan dinormalisasi ke [0, 1].
        device (torch.device | str | None): Device ('cuda' atau 'cpu').

    Contoh data:
        data = {
            "umur": [20, 25, 30, 35, 40, 45],
            "berat": [60, 70, 80, 85, 90, 95],
            "tinggi": [160, 165, 170, 175, 180, 185],
            "tekanan_darah": [110, 115, 120, 130, 140, 150]
        }
        df = pd.DataFrame(data)

        feature_cols = ["umur", "berat", "tinggi"]
        target_col = "tekanan_darah"

        dataset = Regression1DDataset(df, feature_cols, target_col, normalize=True)
    """
    def __init__(self, df, feature_cols, target_col, normalize=True, device=None):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.device = torch.device(device if device else "cpu")

        X = df[feature_cols].values.astype(float)
        y = df[target_col].values.astype(float).reshape(-1, 1)  # target float (bisa multi-output)

        if normalize:
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
            X = (X - X_min) / (X_max - X_min + 1e-8)

        # Simpan ke tensor
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
#========================================================================================================================

#Dataset untuk regresi dengan 2 input (fitur ganda).
#========================================================================================================================
class Regression2DDataset(Dataset):
    """
    Dataset untuk regresi dengan 2 input (fitur ganda).

    Args:
        df (pd.DataFrame): Data input.
        feature1_cols (list[str]): Nama kolom untuk input pertama.
        feature2_cols (list[str]): Nama kolom untuk input kedua.
        target_col (str): Nama kolom target (nilai kontinu).
        normalize (bool): Jika True, fitur akan dinormalisasi ke [0, 1].
        device (torch.device | str | None): Device ('cuda' atau 'cpu').

    Contoh data:
        data = {
            "umur": [20, 25, 30, 35, 40, 45],
            "berat": [60, 70, 80, 85, 90, 95],
            "tinggi": [160, 165, 170, 175, 180, 185],
            "imt": [23.4, 25.7, 27.6, 28.0, 29.1, 30.5],
            "tekanan_darah": [120.5, 122.1, 125.3, 128.9, 132.2, 135.8]  # nilai kontinu
        }

        df = pd.DataFrame(data)

        feature1_cols = ["umur", "berat", "tinggi"]
        feature2_cols = ["imt"]
        target_col = "tekanan_darah"

        dataset = Regression2DDataset(df, feature1_cols, feature2_cols, target_col)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        for X1, X2, y in dataloader:
            print("X1 shape:", X1.shape)  # (batch, num_features_1)
            print("X2 shape:", X2.shape)  # (batch, num_features_2)
            print("y shape:", y.shape)    # (batch, 1)
            print()
            break
    """
    def __init__(self, df, feature1_cols, feature2_cols, target_col, normalize=True, device=None):
        self.feature1_cols = feature1_cols
        self.feature2_cols = feature2_cols
        self.target_col = target_col
        self.device = torch.device(device if device else "cpu")

        # Ambil fitur dan target
        X1 = df[feature1_cols].values.astype(float)
        X2 = df[feature2_cols].values.astype(float)
        y = df[target_col].values.astype(float).reshape(-1, 1)  # target berupa float (kontinu)

        # Normalisasi fitur ke [0, 1]
        if normalize:
            X1 = self._normalize(X1)
            X2 = self._normalize(X2)

        # Simpan tensor ke device
        self.X1 = torch.tensor(X1, dtype=torch.float32, device=self.device)
        self.X2 = torch.tensor(X2, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)

    def _normalize(self, X):
        """Normalisasi data ke rentang [0, 1]."""
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        return (X - X_min) / (X_max - X_min + 1e-8)

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]
#========================================================================================================================

#Dataset untuk regresi dengan 3 input (fitur dari 3 sumber/kelompok).
#========================================================================================================================
class Regression3DDataset(Dataset):
    """
    Dataset untuk regresi dengan 3 input (fitur dari 3 sumber/kelompok).

    Args:
        df (pd.DataFrame): Data input.
        feature1_cols (list[str]): Kolom fitur pertama (misalnya: data fisik).
        feature2_cols (list[str]): Kolom fitur kedua (misalnya: data medis).
        feature3_cols (list[str]): Kolom fitur ketiga (misalnya: embedding atau skor).
        target_col (str): Kolom target (nilai kontinu).
        normalize (bool): Jika True, semua fitur dinormalisasi ke [0, 1].
        device (torch.device | str | None): Device ('cuda' atau 'cpu').

    Contoh data:
        data = {
            # Fitur kelompok 1 (fisik)
            "umur": [20, 25, 30, 35, 40, 45],
            "berat": [60, 70, 80, 85, 90, 95],
            "tinggi": [160, 165, 170, 175, 180, 185],

            # Fitur kelompok 2 (medis)
            "imt": [23.4, 25.7, 27.6, 28.0, 29.1, 30.5],
            "tekanan_darah": [120, 125, 130, 140, 145, 150],

            # Fitur kelompok 3 (embedding / skor / representasi teks)
            "skor_keluhan": [0.3, 0.6, 0.9, 0.4, 0.7, 0.2],
            "skor_kondisi": [0.1, 0.8, 0.5, 0.6, 0.9, 0.3],

            # Target regresi
            "hasil_prediksi": [5.2, 6.1, 7.3, 6.8, 7.9, 8.4]
        }

        df = pd.DataFrame(data)

        # Tentukan kolom
        feature1_cols = ["umur", "berat", "tinggi"]
        feature2_cols = ["imt", "tekanan_darah"]
        feature3_cols = ["skor_keluhan", "skor_kondisi"]
        target_col = "hasil_prediksi"

        # Buat dataset
        dataset = Regression3DDataset(df, feature1_cols, feature2_cols, feature3_cols, target_col)

        # Buat DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        for X1, X2, X3, y in dataloader:
            print("X1:", X1.shape)
            print("X2:", X2.shape)
            print("X3:", X3.shape)
            print("y:", y.shape)
            break
    """
    def __init__(self, df, feature1_cols, feature2_cols, feature3_cols, target_col, normalize=True, device=None):
        self.feature1_cols = feature1_cols
        self.feature2_cols = feature2_cols
        self.feature3_cols = feature3_cols
        self.target_col = target_col
        self.device = torch.device(device if device else "cpu")

        # Ambil data fitur dan target
        X1 = df[feature1_cols].values.astype(float)
        X2 = df[feature2_cols].values.astype(float)
        X3 = df[feature3_cols].values.astype(float)
        y = df[target_col].values.astype(float).reshape(-1, 1)  # target kontinu

        # Normalisasi fitur
        if normalize:
            X1 = self._normalize(X1)
            X2 = self._normalize(X2)
            X3 = self._normalize(X3)

        # Konversi ke tensor dan kirim ke device
        self.X1 = torch.tensor(X1, dtype=torch.float32, device=self.device)
        self.X2 = torch.tensor(X2, dtype=torch.float32, device=self.device)
        self.X3 = torch.tensor(X3, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)

    def _normalize(self, X):
        """Normalisasi ke [0, 1]."""
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        return (X - X_min) / (X_max - X_min + 1e-8)

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.X3[idx], self.y[idx]
#========================================================================================================================