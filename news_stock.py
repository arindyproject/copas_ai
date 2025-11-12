import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
tqdm.pandas()  
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import torch
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import os
import ast
import re

class NewsStock:
    def __init__(
        self,
        file_news=None,
        file_stock=None,
        start_date=None,
        end_date=None,
        cutoff_time=None,
        norm_vector=False,
        bert=False,
        pretrained_model="indobenchmark/indobert-lite-base-p2",
        mode_news="csv",  # ✅ mode: 'csv' atau 'mysql'
        mode_stock="csv",  # ✅ mode: 'csv' atau 'mysql'
        table_news="news_data",  # nama tabel berita di MySQL
        table_stock="stock_data",  # nama tabel saham di MySQL
        config_path="config.txt"
    ):
        self.mode_news = mode_news
        self.mode_stock = mode_stock
        self.cutoff_time = cutoff_time
        self.norm_vector = norm_vector

        # ----------------------------------------------------------
        # 🔧 Load config.txt (untuk mode MySQL)
        # ----------------------------------------------------------
        config = {}
        with open(config_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    config[key.strip()] = value.strip().strip('"')

        db_user = config.get("db_user", "root")
        db_pass = config.get("db_pass", "")
        db_host = config.get("db_host", "localhost")
        db_name = config.get("db_name", "news_db")
        port = 3306                # opsional jika bukan default

        encoded_pass = quote_plus(db_pass)
        self.engine = create_engine(f"mysql+pymysql://{db_user}:{encoded_pass}@{db_host}:{port}/{db_name}")

        # ----------------------------------------------------------
        # 📰 NEWS DATA
        # ----------------------------------------------------------
        if mode_news == "csv":
            self.file_news = file_news
            self.news_data = pd.read_csv(self.file_news)
        elif mode_news == "mysql":
            print(f"🔗 Memuat data berita dari MySQL tabel '{table_news}'...")
            self.news_data = pd.read_sql(f"SELECT * FROM {table_news}", self.engine)
        else:
            raise ValueError("mode_news harus 'csv' atau 'mysql'")

        self.news_data['tgl'] = pd.to_datetime(self.news_data['tgl'], errors='coerce')
        self.news_data = self.news_data.sort_values(by='tgl', ascending=True)

        # filter berdasarkan tanggal
        if start_date:
            self.news_data = self.news_data[self.news_data['tgl'] >= pd.to_datetime(start_date)]
        if end_date:
            self.news_data = self.news_data[self.news_data['tgl'] <= pd.to_datetime(end_date)]

        # tanggal dan waktu terpisah
        self.news_data['date_only'] = self.news_data['tgl'].dt.date
        self.news_data['time_only'] = self.news_data['tgl'].dt.time

        # hapus duplikat
        self.news_data = self.news_data.drop_duplicates(subset=['judul', 'nama_sumber', 'date_only'], keep='first')

        # ----------------------------------------------------------
        # 📈 STOCK DATA
        # ----------------------------------------------------------
        if mode_stock == "csv":
            self.file_stock = file_stock
            self.stock_data = pd.read_csv(self.file_stock)
        elif mode_stock == "mysql":
            print(f"🔗 Memuat data saham dari MySQL tabel '{table_stock}'...")
            self.stock_data = pd.read_sql(f"SELECT * FROM {table_stock}", self.engine)

        # pastikan format tanggal
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], format='mixed', dayfirst=True)
        self.stock_data = self.stock_data.sort_values(by='Date', ascending=True)

        # ubah kolom numerik
        self.stock_data["Change"] = (
            self.stock_data["Change"].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

        # parse volume
        self.stock_data['Volume'] = self.stock_data['Volume'].apply(self.parse_volume)

        # ubah kolom harga
        numeric_cols = ['Close', 'Open', 'High', 'Low']
        for col in numeric_cols:
            self.stock_data[col] = (
                self.stock_data[col].astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .astype(float)
            )

        # filter tanggal
        if start_date:
            self.stock_data = self.stock_data[self.stock_data['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            self.stock_data = self.stock_data[self.stock_data['Date'] <= pd.to_datetime(end_date)]

        # status naik/turun
        self.stock_data['Status'] = self.stock_data['Change'].apply(lambda x: 'GOOD' if x > 0 else 'BAD' )

        # hapus duplikat
        self.stock_data = self.stock_data.drop_duplicates(subset=['Date'], keep='first')

        # ----------------------------------------------------------
        # ⚙️ Device dan BERT
        # ----------------------------------------------------------
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.m_bert = None
        if bert:
            print("🔍 Memuat model BERT...")
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            self.m_bert = AutoModel.from_pretrained(pretrained_model)
            self.m_bert.to(self.device)
            self.m_bert.eval()

    def parse_volume(self,val):
        if pd.isna(val):
            return None
        val = val.replace('.', '').replace(',', '.').strip()  # ganti pemisah desimal
        if val.endswith('B'):
            return float(val[:-1]) * 1e9
        elif val.endswith('M'):
            return float(val[:-1]) * 1e6
        elif val.endswith('K'):
            return float(val[:-1]) * 1e3
        else:
            return float(val)
    
    def check_types(self):
        print("News Data Types:")
        print(self.news_data.dtypes)
        print(self.news_data.shape)

        print("\nStock Data Types:")
        print(self.stock_data.dtypes)
        print(self.stock_data.shape)

        print("\nMarge News and Stock:")
        merged_data = self.marge_news_stock()
        print(merged_data.dtypes)
        print(merged_data.shape)

    def encode_bertembedding(self, text):
        encoded_inputs  = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64
        ).to(self.device)
        # Dapatkan embedding dari model
        with torch.no_grad():
            outputs = self.m_bert(**encoded_inputs)
            vector = outputs.pooler_output  # [1, 768]
        vec = vector.cpu().numpy().flatten()
        # Normalisasi vektor jika diperlukan 0 -1
        if self.norm_vector:
            norm = np.linalg.norm(vec)
            if norm != 0:
                vec = vec / norm
        return vec
    
    def get_unique_news_date(self):
        return self.news_data['date_only'].unique()

    def get_unique_stock_date(self):
        return self.stock_data['Date'].unique()
    
    def show_news(self):
        return self.news_data
    
    def show_stock(self):
        return self.stock_data
    
    def show_stock_status_chart_dounut(self):
        status_counts = self.stock_data['Status'].value_counts()
        labels = status_counts.index
        sizes = status_counts.values
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Stock Status Distribution')
        plt.show()
    
    def show_chart_news_nama_sumber_chart_dounut(self):
        news_count = self.news_data['nama_sumber'].value_counts()
        labels = news_count.index
        sizes = news_count.values       
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('News Source Distribution')
        plt.show()

    def show_chart_news_nama_sumber_bar(self, top_n=10):

        # Hitung jumlah berita per sumber
        news_count = self.news_data['nama_sumber'].value_counts().head(top_n)
        labels = news_count.index
        sizes = news_count.values

        # Hitung persentase
        percentages = (sizes / sizes.sum()) * 100

        # Buat figure
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(labels, sizes, color='skyblue', edgecolor='black')

        # Tambahkan label jumlah & persentase di ujung batang
        for bar, count, pct in zip(bars, sizes, percentages):
            ax.text(
                bar.get_width() + max(sizes) * 0.01,  # sedikit ke kanan ujung batang
                bar.get_y() + bar.get_height() / 2,
                f'{count} ({pct:.1f}%)',
                va='center',
                fontsize=9
            )

        # Styling
        ax.set_xlabel('Jumlah Berita', fontsize=11)
        ax.set_ylabel('Sumber Berita', fontsize=11)
        ax.set_title(f'Top {top_n} Sumber Berita', fontsize=13, fontweight='bold')
        ax.invert_yaxis()  # agar yang terbesar di atas
        plt.tight_layout()
        plt.show()

    def marge_news_stock(self, mode="same_day"):
        """
        Menggabungkan data berita dan saham berdasarkan tanggal.
        
        Parameter:
        - mode: 
            "same_day"  → berita hari ini cocok dengan saham hari yang sama
            "next_day"  → berita hari ini cocok dengan saham hari berikutnya (+1 hari)
        """

        # Pastikan kolom tanggal dalam format datetime
        self.news_data['date_only'] = pd.to_datetime(self.news_data['date_only'], errors='coerce')
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], errors='coerce')

        # Buat kolom date_merge sesuai mode
        if mode == "next_day":
            self.news_data['date_merge'] = self.news_data['date_only'] + pd.Timedelta(days=1)
        else:
            self.news_data['date_merge'] = self.news_data['date_only']

        # Jika ada cutoff_time, geser berita setelah cutoff ke hari berikutnya
        if self.cutoff_time:
            cutoff_time_obj = pd.to_datetime(self.cutoff_time).time()
            after_cutoff = self.news_data['time_only'] > cutoff_time_obj
            self.news_data.loc[after_cutoff, 'date_merge'] = (
                self.news_data.loc[after_cutoff, 'date_merge'] + pd.Timedelta(days=1)
            )

        # Merge berita dan saham berdasarkan date_merge
        merged_data = pd.merge(
            self.news_data,
            self.stock_data,
            left_on='date_merge',
            right_on='Date',
            how='left'
        )

        # Hapus baris tanpa Status atau judul kosong
        merged_data = merged_data.dropna(subset=['Status', 'judul'])

        #print information
        print("\n\n=====================Merge News and Stock=====================")
        print("Total Stock data : ", self.stock_data.shape)
        print("Total News data  : ", self.news_data.shape)
        print("Total Merged data: ", merged_data.shape)
        print(f"✅ Selesai merge berita dan saham (mode: {mode}) dengan {merged_data.shape[0]} baris hasil.")
        return merged_data

    def marge_news_stock_add_vector(self, target_cols=['judul'], mode='same_day', save=False, path=""):
        """
        Menambahkan kolom embedding (vector) untuk kolom teks seperti 'judul' atau 'konten'
        setelah melakukan merge berita dan saham.

        Parameter:
        - target_cols : list kolom teks yang ingin diubah jadi embedding
        - mode        : 'same_day' untuk merge di tanggal sama
                        'next_day' untuk merge dengan saham hari berikutnya
        """

        # Ambil hasil merge sesuai mode
        merged_data = self.marge_news_stock(mode=mode)

        # Loop setiap kolom target dan buat embedding
        for col in target_cols:
            new_col = f"vec_{col}"  # nama kolom embedding baru
            print(f"🔄 Membuat embedding untuk kolom: {col} → {new_col}")
            merged_data[new_col] = merged_data[col].progress_apply(self.encode_bertembedding)

        #print information
        print("\n\n=====================Add Embedding Vector=====================")
        print("Target Columns  : ", target_cols)
        print(f"✅ Selesai membuat embedding untuk {len(target_cols)} kolom (mode: {mode})")

        # Simpan jika diminta
        if save:
            filename = f"{path}/merged_with_vector_{mode}.csv"

            # Buat salinan agar tidak ubah data asli
            df_to_save = merged_data.copy()

            # Temukan kolom yang dimulai dengan "vec_"
            vec_cols = [col for col in df_to_save.columns if col.startswith("vec_")]

            # Konversi setiap kolom vektor menjadi string "1 2 3 4"
            for col in vec_cols:
                def vec_to_str(v):
                    if isinstance(v, (list, np.ndarray)):
                        # ubah jadi string tanpa koma dan kurung
                        return " ".join(map(str, np.array(v).flatten().tolist()))
                    elif isinstance(v, str):
                        return v.strip("[]").replace(",", " ")
                    else:
                        return ""
                df_to_save[col] = df_to_save[col].apply(vec_to_str)

            # Simpan ke CSV
            df_to_save.to_csv(filename, index=False)
            print(f"💾 Data telah disimpan ke file: {filename}")
            print(f"📊 Kolom vektor yang dikonversi: {vec_cols}")

        return merged_data
    

 
    def load_merged_vector_data(self, mode='same_day', path='data/vector_cache'):
        """
        Memuat kembali data hasil 'marge_news_stock_add_vector' dari file CSV
        dan mengonversi kolom dengan prefix 'vec_' kembali menjadi np.array.
        Mendukung format seperti '1.1 4.2 -5.3' tanpa tanda koma.
        """

        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"merged_with_vector_{mode}.csv")

        if not os.path.exists(filename):
            print(f"⚠️ File '{filename}' tidak ditemukan.")
            return None

        print(f"📂 Memuat data dari: {filename}")
        df = pd.read_csv(filename)

        vec_cols = [col for col in df.columns if col.startswith("vec_")]

        def safe_parse_vec(x):
            if not isinstance(x, str):
                return np.array(x, dtype=np.float32)

            s = x.strip()
            # Format seperti "[1.2 3.4 -5.6]" atau "1.2 3.4 -5.6"
            s = s.strip("[]")  # hilangkan tanda kurung
            if not s:
                return np.array([], dtype=np.float32)
            try:
                # Pisahkan berdasarkan spasi
                parts = re.split(r"\s+", s)
                nums = [float(p) for p in parts if p.strip() != ""]
                return np.array(nums, dtype=np.float32)
            except Exception:
                # Jika gagal parsing, kembalikan None
                return np.array([], dtype=np.float32)

        for col in vec_cols:
            print(f"🔄 Konversi kolom {col} → np.array ...")
            df[col] = df[col].apply(safe_parse_vec)

        print(f"✅ Data berhasil dimuat ({len(df)} baris, {len(df.columns)} kolom)")
        print(f"📊 Kolom vector: {vec_cols}")
        return df


    def create_torch_dataset(
        self,
        data,
        feature_cols,
        target_col,
        task_type="multi_label",   # opsi: "regression", "multi_class", "multi_label"
        test_size=0.2,
        random_state=42,
        batch_size=32,
        shuffle=True
    ):
        """
        Membuat dataset dan DataLoader untuk PyTorch dari DataFrame.
        """
        print("\n===================== Membuat Dataset Torch =====================")
        print(f"🧩 Fitur       : {feature_cols}")
        print(f"🎯 Target      : {target_col}")
        print(f"📘 Task Type   : {task_type}")
        print(f"📊 Test Size   : {test_size}")
        print(f"⚙️ Batch Size  : {batch_size}")

        # === Siapkan X ===
        X_parts = []
        for col in feature_cols:
            if isinstance(data[col].iloc[0], np.ndarray):  # jika vektor
                X_parts.append(np.stack(data[col].values))
            else:
                X_parts.append(data[col].values.reshape(-1, 1))
        X = np.concatenate(X_parts, axis=1).astype(np.float32)

        labels = None

        # === Siapkan y ===
        if task_type == "multi_label":
            labels = sorted(data[target_col].dropna().unique())
            print(f"🧾 Label unik ({target_col}) → {labels}")

            for lbl in labels:
                new_col = f"y_{target_col}_{lbl.upper()}"
                data[new_col] = (data[target_col] == lbl).astype(int)

            y_cols = [f"y_{target_col}_{lbl.upper()}" for lbl in labels]
            y = data[y_cols].values.astype(np.float32)

        elif task_type == "multi_class":
            le = LabelEncoder()
            data[f"y_{target_col}"] = le.fit_transform(data[target_col].astype(str))
            labels = list(le.classes_)
            print(f"🧾 Label unik ({target_col}) → {labels}")
            y = data[f"y_{target_col}"].values.astype(np.int64).reshape(-1, 1)

        elif task_type == "regression":
            y = data[[target_col]].values.astype(np.float32)
        else:
            raise ValueError("task_type harus 'regression', 'multi_class', atau 'multi_label'")

        # === Split data ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )

        # === Convert ke tensor ===
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        if task_type == "multi_class":
            y_train_tensor = torch.tensor(y_train.squeeze(), dtype=torch.long)
            y_test_tensor = torch.tensor(y_test.squeeze(), dtype=torch.long)
        else:
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # === Dataset & DataLoader ===
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # === Print Info ===
        print(f"\n✅ Dataset berhasil dibuat:")
        print(f"   🔹 Train size : {len(train_dataset)} sampel")
        print(f"   🔹 Test size  : {len(test_dataset)} sampel")

        # === Print contoh sampel ===
        print("\n🧠 Contoh Sampel Train:")
        for i in range(min(2, len(train_dataset))):
            X_sample, y_sample = train_dataset[i]
            print(f"  ▶️ X[{i}].shape: {tuple(X_sample.shape)} | y[{i}]: {y_sample.tolist()}")

        print("\n🧪 Contoh Sampel Test:")
        for i in range(min(2, len(test_dataset))):
            X_sample, y_sample = test_dataset[i]
            print(f"  ▶️ X[{i}].shape: {tuple(X_sample.shape)} | y[{i}]: {y_sample.tolist()}")

        print(f"\n🎯 Labels: {labels}")

        return train_loader, test_loader, X_train, X_test, y_train, y_test, labels
    
   

file_news   = 'data_news/news.csv'
file_stock  = 'data_news/ihsg.csv'
start_date  = '2009-01-01'
cutoff_time = '12:00:00'
news_stock = NewsStock(
    file_news, 
    file_stock, 
    start_date=start_date, 
    cutoff_time=cutoff_time, 
    bert=True, 
    mode_news="mysql", 
    table_news='google_news'
)
#print(news_stock.check_types())
#print(news_stock.show_news())
#print(news_stock.show_stock())
#print(news_stock.marge_news_stock_add_vector())#[['Date', 'date_merge','date_only', 'embedding', 'judul', 'Status']])
#news_stock.show_stock_status_chart_dounut()
#news_stock.show_chart_news_nama_sumber_bar()
#print(news_stock.encode_bertembedding(["Pasar saham menguat hari ini","Ekonomi Indonesia tumbuh pesat"])[:10])

# Ambil tanggal unik dari berita dan saham
#news_dates = set(news_stock.get_unique_news_date())
#stock_dates = set(news_stock.get_unique_stock_date())

# Cari irisan (tanggal yang sama)
#same_dates = news_dates.intersection(stock_dates)

# Tampilkan hasil
#print("Jumlah tanggal unik di berita :", len(news_dates))
#print("Jumlah tanggal unik di saham   :", len(stock_dates))
#print("Jumlah tanggal yang sama       :", len(same_dates))
#print("\nTanggal yang sama:")
#print(sorted(same_dates))


merged_vec = news_stock.marge_news_stock_add_vector(target_cols=['judul'], save=True, path="data_news")
#merged_vec = news_stock.load_merged_vector_data(mode='same_day', path="data_news")
train_loader, test_loader, X_train, X_test, y_train, y_test, labels = news_stock.create_torch_dataset(
    data=merged_vec,
    feature_cols=['vec_judul'],
    target_col='Status',
    task_type='multi_label',
    test_size=0.2,
    batch_size=16
)

