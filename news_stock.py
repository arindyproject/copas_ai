import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, AutoModel
import torch
import numpy as np


class NewsStock:
    def __init__(self, file_news, file_stock, start_date=None, end_date=None, cutoff_time=None,norm_vector=False,bert=False, pretrained_model = "indobenchmark/indobert-lite-base-p2"):
        #news data
        #--------------------------------------------------------------------------
        self.file_news = file_news
        self.news_data = pd.read_csv(self.file_news)
        self.news_data['tgl'] = pd.to_datetime(self.news_data['tgl'])
        self.news_data = self.news_data.sort_values(by='tgl', ascending=True)
        #start and end date
        if start_date:
            self.news_data = self.news_data[self.news_data['tgl'] >= pd.to_datetime(start_date)]
        if end_date:
            self.news_data = self.news_data[self.news_data['tgl'] <= pd.to_datetime(end_date)]
        #date only
        self.news_data['date_only'] = self.news_data['tgl'].dt.date
        #time only
        self.news_data['time_only'] = self.news_data['tgl'].dt.time
        #buang duplicate
        self.news_data = self.news_data.drop_duplicates(subset=['judul', 'nama_sumber', 'date_only'], keep='first')
        #--------------------------------------------------------------------------

        #stock data
        #--------------------------------------------------------------------------
        self.file_stock = file_stock
        self.stock_data = pd.read_csv(self.file_stock)
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], format='mixed', dayfirst=True)
        self.stock_data = self.stock_data.sort_values(by='Date', ascending=True)    
        # Bersihkan dan ubah kolom Change
        self.stock_data["Change"] = (
            self.stock_data["Change"]
            .astype(str)                        # pastikan string
            .str.replace("%", "", regex=False)  # hilangkan tanda persen
            .str.replace(",", ".", regex=False) # ubah koma jadi titik
            .astype(float)                      # ubah ke float
        )
        # Ubah kolom Volume
        self.stock_data['Volume'] = self.stock_data['Volume'].apply(self.parse_volume)
        # Ubah kolom numerik lainnya
        numeric_cols = ['Close', 'Open', 'High', 'Low']
        for col in numeric_cols:
            self.stock_data[col] = (
                self.stock_data[col]
                .astype(str)
                .str.replace('.', '', regex=False)  # hapus pemisah ribuan
                .str.replace(',', '.', regex=False)  # ubah koma menjadi titik desimal
                .astype(float)
            )
        #start and end date
        if start_date:
            self.stock_data = self.stock_data[self.stock_data['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            self.stock_data = self.stock_data[self.stock_data['Date'] <= pd.to_datetime(end_date)]
        #add status column
        self.stock_data['Status'] = self.stock_data['Change'].apply(lambda x: 'GOOD' if x > 0 else ('BAD' if x < 0 else 'NEUTRAL'))
        # buang duplicate
        self.stock_data = self.stock_data.drop_duplicates(subset=['Date'], keep='first')
        #--------------------------------------------------------------------------

        #cutoff time
        self.cutoff_time = cutoff_time

        #device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #load bert model
        self.tokenizer = None
        self.m_bert    = None
        if bert:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            self.m_bert    = AutoModel.from_pretrained(pretrained_model)
            self.m_bert.to(self.device)
            self.m_bert.eval()

        #norm vector
        self.norm_vector = norm_vector

    
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




    def marge_news_stock(self):
        # Pastikan tipe datetime
        self.news_data['date_only'] = pd.to_datetime(self.news_data['date_only'], errors='coerce')
        self.stock_data['Date']     = pd.to_datetime(self.stock_data['Date'], errors='coerce')

        # Konversi ke string (format YYYY-MM-DD)
        #self.news_data['date_str'] = self.news_data['date_only'].dt.strftime('%Y-%m-%d')
        #self.stock_data['date_str'] = self.stock_data['Date'].dt.strftime('%Y-%m-%d')

        # Filter berdasarkan cutoff_time jika ada
        if self.cutoff_time:
            cutoff_time_obj = pd.to_datetime(self.cutoff_time).time()
            self.news_data = self.news_data[self.news_data['time_only'] <= cutoff_time_obj]

        # Lakukan merge antara berita dan data saham berdasarkan tanggal
        merged_data = pd.merge(
            self.news_data,
            self.stock_data,
            left_on='date_only',
            right_on='Date',
            how='left'
        )

        # Hapus baris yang tidak memiliki status saham atau judul kosong
        merged_data = merged_data.dropna(subset=['Status', 'judul'])

        return merged_data
    
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

file_news   = 'data_news/news.csv'
file_stock  = 'data_news/ihsg.csv'
start_date  = '2009-01-01'
cutoff_time = '15:00:00'
news_stock = NewsStock(file_news, file_stock, start_date=start_date, cutoff_time=None)
print(news_stock.check_types())
#print(news_stock.show_news())
#print(news_stock.show_stock())
print(news_stock.marge_news_stock()[['Date','date_only', 'judul', 'Status']])
#news_stock.show_stock_status_chart_dounut()
#news_stock.show_chart_news_nama_sumber_bar()
#print(news_stock.encode_bertembedding(["Pasar saham menguat hari ini","Ekonomi Indonesia tumbuh pesat"])[:10])

# Ambil tanggal unik dari berita dan saham
news_dates = set(news_stock.get_unique_news_date())
stock_dates = set(news_stock.get_unique_stock_date())

# Cari irisan (tanggal yang sama)
same_dates = news_dates.intersection(stock_dates)

# Tampilkan hasil
print("Jumlah tanggal unik di berita :", len(news_dates))
print("Jumlah tanggal unik di saham   :", len(stock_dates))
print("Jumlah tanggal yang sama       :", len(same_dates))
print("\nTanggal yang sama:")
#print(sorted(same_dates))