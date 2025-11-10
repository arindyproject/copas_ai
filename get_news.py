import feedparser
from datetime import datetime
from urllib.parse import quote_plus
import requests
import json
from zoneinfo import ZoneInfo
from googlenewsdecoder import gnewsdecoder
import pandas as pd
from sqlalchemy import create_engine, text, inspect

# --- Baca konfigurasi ---
config = {}
with open("config.txt", "r") as f:
    for line in f:
        if "=" in line:
            key, value = line.strip().split("=", 1)
            config[key.strip()] = value.strip().strip('"')

class News:
    def __init__(self, query="", google_script_url="", hl="id", gl='ID', ceid='ID:id', config={}):
        self.query = query
        self.hl = hl
        self.gl = gl
        self.ceid = ceid
        self.google_script_url = google_script_url
        self.timezone = ZoneInfo("Asia/Jakarta")

        # Koneksi ke MySQL
        db_user = config.get("db_user", "root")
        db_pass = config.get("db_pass", "")
        db_host = config.get("db_host", "localhost")
        db_name = config.get("db_name", "news_db")
        self.engine = create_engine(f"mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}")

    # ------------------------------------------------------------
    # 1️⃣ Ambil berita dari Google News RSS
    # ------------------------------------------------------------
    def get_news(self):
        url = f"https://news.google.com/rss/search?q={quote_plus(self.query)}&hl={self.hl}&gl={self.gl}&ceid={self.ceid}"
        feed = feedparser.parse(url)
        out = []

        for entry in feed.entries:
            if hasattr(entry, "published_parsed"):
                published_dt = datetime(*entry.published_parsed[:6]).astimezone(self.timezone)
                published_str = published_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                published_str = "Unknown"

            source_name = getattr(entry.source, "title", "Tidak diketahui") if hasattr(entry, "source") else "Tidak diketahui"
            source_link = getattr(entry.source, "href", "Tidak tersedia") if hasattr(entry, "source") else "Tidak tersedia"

            try:
                decoded_url = gnewsdecoder(entry.link, interval=1)
                link_asli = decoded_url["decoded_url"] if decoded_url.get("status") else entry.link
            except Exception:
                link_asli = entry.link

            title_clean = entry.title
            if f" - {source_name}" in title_clean:
                title_clean = title_clean.replace(f" - {source_name}", "").strip()

            out.append({
                'tgl': published_str,
                'judul': title_clean,
                'rss': 'Google News',
                'nama_sumber': source_name,
                'link_sumber': source_link,
                'query': self.query,
                'link_berita': link_asli,
                'konten': '',
                'ringkasan': ''
            })
        return out

    # ------------------------------------------------------------
    # 2️⃣ Kirim ke Google Sheet via Google Apps Script
    # ------------------------------------------------------------
    def send_to_google_sheet_batch(self, sheet_name="news"):
        news_list = self.get_news()
        data = {"sheet": sheet_name, "items": news_list}
        headers = {"Content-Type": "application/json"}
        try:
            res = requests.post(self.google_script_url, data=json.dumps(data), headers=headers)
            print(f"✅ {self.query} -> {res.status_code}")
        except Exception as e:
            print(f"❌ Gagal kirim batch {self.query}: {e}")

    # ------------------------------------------------------------
    # 3️⃣ Buat tabel jika belum ada
    # ------------------------------------------------------------
    def create_table_if_not_exists(self, table_name="news_data"):
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tgl DATETIME,
            judul TEXT,
            rss VARCHAR(50),
            nama_sumber VARCHAR(255),
            link_sumber TEXT,
            query VARCHAR(255),
            link_berita TEXT,
            konten TEXT,
            ringkasan TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_news (link_berita)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        with self.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.commit()
        print(f"🧱 Tabel '{table_name}' siap digunakan ✅")

    # ------------------------------------------------------------
    # 4️⃣ Simpan ke database MySQL (hindari duplikat)
    # ------------------------------------------------------------
    def save_to_mysql(self, table_name="news_data"):
        self.create_table_if_not_exists(table_name)
        news_list = self.get_news()
        if not news_list:
            print(f"⚠️ Tidak ada berita untuk query '{self.query}'")
            return

        df = pd.DataFrame(news_list)
        inspector = inspect(self.engine)

        # Hindari duplikat: ambil judul+tgl yang sudah ada
        if inspector.has_table(table_name):
            with self.engine.connect() as conn:
                existing = pd.read_sql(f"SELECT judul, tgl FROM {table_name}", conn)
            
            df["tgl"] = pd.to_datetime(df["tgl"], errors="coerce")
            existing["tgl"] = pd.to_datetime(existing["tgl"], errors="coerce")
            merged = pd.merge(df, existing, on=["judul", "tgl"], how="left", indicator=True)
            df_new = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        else:
            df_new = df

        if len(df_new) == 0:
            print(f"⚠️ Tidak ada berita baru untuk '{self.query}' (semua sudah ada).")
            return

        # Simpan berita baru
        try:
            df_new.to_sql(table_name, con=self.engine, if_exists="append", index=False)
            print(f"✅ {len(df_new)} berita baru '{self.query}' berhasil disimpan ke '{table_name}'")
        except Exception as e:
            print(f"❌ Gagal simpan ke MySQL untuk '{self.query}': {e}")
    
    def save_to_mysql_one_by_one(self, table_name="news_data"):
        # Pastikan tabel sudah ada
        self.create_table_if_not_exists(table_name)

        # Ambil berita dari sumber
        news_list = self.get_news()
        if not news_list:
            print(f"⚠️ Tidak ada berita untuk query '{self.query}'")
            return

        df = pd.DataFrame(news_list)
        df["tgl"] = pd.to_datetime(df["tgl"], errors="coerce")

        # Cek apakah tabel sudah ada
        inspector = inspect(self.engine)
        if not inspector.has_table(table_name):
            print(f"ℹ️ Tabel '{table_name}' belum ada, membuat baru...")
            self.create_table_if_not_exists(table_name)

        count_new = 0

        with self.engine.begin() as conn:
            for _, row in df.iterrows():
                link_berita = row.get("link_berita")
                if not link_berita:
                    continue  # skip jika tidak ada link

                # Cek apakah link sudah ada di database
                check_sql = text(f"SELECT COUNT(*) AS cnt FROM `{table_name}` WHERE link_berita = :link_berita")
                result = conn.execute(check_sql, {"link_berita": link_berita}).scalar()

                if result and result > 0:
                    continue  # sudah ada, skip

                # Siapkan data insert
                insert_sql = text(f"""
                    INSERT INTO `{table_name}` 
                    (tgl, judul, rss, nama_sumber, link_sumber, query, link_berita, konten, ringkasan, created_at)
                    VALUES (:tgl, :judul, :rss, :nama_sumber, :link_sumber, :query, :link_berita, :konten, :ringkasan, :created_at)
                """)

                params = {
                    "tgl": row.get("tgl"),
                    "judul": row.get("judul"),
                    "rss": row.get("rss"),
                    "nama_sumber": row.get("nama_sumber"),
                    "link_sumber": row.get("link_sumber"),
                    "query": row.get("query"),
                    "link_berita": link_berita,
                    "konten": row.get("konten"),
                    "ringkasan": row.get("ringkasan"),
                    "created_at": datetime.now()
                }

                try:
                    conn.execute(insert_sql, params)
                    count_new += 1
                except Exception as e:
                    print(f"⚠️ Gagal insert berita '{row.get('judul')[:50]}...': {e}")

        if count_new > 0:
            print(f"✅ {count_new} berita baru '{self.query}' berhasil disimpan ke '{table_name}'")
        else:
            print(f"⚠️ Tidak ada berita baru untuk '{self.query}' (semua link_berita sudah ada).")


emiten = [
    # Sektor Keuangan (Financials) - Maksimal
    "BBCA",  # Bank Central Asia Tbk.
    "BBRI",  # Bank Rakyat Indonesia (Persero) Tbk.
    "BMRI",  # Bank Mandiri (Persero) Tbk.
    "BBNI",  # Bank Negara Indonesia (Persero) Tbk.
    "BRIS",  # Bank Syariah Indonesia Tbk.
    "BDMN",  # Bank Danamon Indonesia Tbk.
    "PNBN",  # Bank Panin Tbk.
    "ARTO",  # Bank Jago Tbk. (Bank Digital)
    "MEGA",  # Bank Mega Tbk.
    "ADMF",  # Adira Dinamika Multi Finance Tbk.
    "BTPS",  # Bank BTPN Syariah Tbk.
    "BTPN",  # Bank BTPN Tbk.
    "NISP",  # Bank OCBC NISP Tbk.
    "CIMB",  # CIMB Niaga Tbk.
    "BFIT",  # BFI Finance Indonesia Tbk.
    "PNLF",  # Panin Financial Tbk. (Asuransi/Keuangan)
    "FREN",  # Smartfren Telecom Tbk. (meskipun lebih ke Infrastruktur, sering diperdagangkan)
    "MIDI",  # Midi Utama Indonesia Tbk. (Alfamidi)

    # Sektor Energi (Energy) - Maksimal
    "ADRO",  # Adaro Energy Indonesia Tbk.
    "ITMG",  # Indo Tambangraya Megah Tbk.
    "PGAS",  # Perusahaan Gas Negara Tbk.
    "MEDC",  # Medco Energi Internasional Tbk.
    "PTBA",  # Bukit Asam Tbk.
    "INDY",  # Indika Energy Tbk.
    "HRUM",  # Harum Energy Tbk. (Batu Bara & Nikel)
    "BYAN",  # Bayan Resources Tbk.
    "ELSA",  # Elnusa Tbk. (Jasa Minyak & Gas)
    "PETR",  # Petrosea Tbk. (Kontraktor tambang)
    "ENRG",  # Energi Mega Persada Tbk.

    # Sektor Barang Baku (Basic Materials) - Maksimal
    "ANTM",  # Aneka Tambang Tbk.
    "INCO",  # Vale Indonesia Tbk.
    "BRPT",  # Barito Pacific Tbk.
    "TPIA",  # Chandra Asri Pacific Tbk.
    "ESSA",  # Surya Esa Perkasa Tbk.
    "SMGR",  # Semen Indonesia (Persero) Tbk.
    "INTP",  # Indocement Tunggal Prakarsa Tbk.
    "MDKA",  # Merdeka Copper Gold Tbk.
    "AKRA",  # AKR Corporindo Tbk.
    "TKIM",  # Pabrik Kertas Tjiwi Kimia Tbk.
    "FASW",  # Fajar Surya Wisesa Tbk.
    "DSNG",  # Dharma Satya Nusantara Tbk. (Agrikultur)
    "CASA",  # Capital Financial Indonesia Tbk.
    "TOTO",  # Surya Toto Indonesia Tbk. (Sanitaryware)

    # Sektor Perindustrian (Industrials) - Maksimal
    "ASII",  # Astra International Tbk.
    "UNTR",  # United Tractors Tbk.
    "ADHI",  # Adhi Karya (Persero) Tbk.
    "WIKA",  # Wijaya Karya (Persero) Tbk.
    "PTPP",  # PP (Persero) Tbk.
    "JSMR",  # Jasa Marga (Persero) Tbk.
    "GIAA",  # Garuda Indonesia (Persero) Tbk.
    "ARNA",  # Arwana Citramulia Tbk. (Keramik)
    "IMAS",  # Indomobil Sukses Internasional Tbk.
    "BULL",  # Buana Lintas Lautan Tbk.
    "IKPT",  # Inti Karya Persada Teknik Tbk.
    "SRSN",  # Indo Acidatama Tbk.

    # Sektor Barang Konsumen Primer (Consumer Non-Cyclicals) - Maksimal
    "INDF",  # Indofood Sukses Makmur Tbk.
    "ICBP",  # Indofood CBP Sukses Makmur Tbk.
    "UNVR",  # Unilever Indonesia Tbk.
    "AMRT",  # Sumber Alfaria Trijaya Tbk. (Alfamart)
    "GGRM",  # Gudang Garam Tbk.
    "HMSP",  # HM Sampoerna Tbk.
    "ROTI",  # Nippon Indosari Corpindo Tbk.
    "AALI",  # Astra Agro Lestari Tbk. (CPO)
    "LSIP",  # London Sumatera Indonesia Tbk. (CPO)
    "MAIN",  # Malindo Feedmill Tbk.
    "CPIN",  # Charoen Pokphand Indonesia Tbk.
    "JPFA",  # Japfa Comfeed Indonesia Tbk.
    "SMCB",  # Solusi Bangun Indonesia Tbk. (Semen)

    # Sektor Barang Konsumen Non-Primer (Consumer Cyclicals) - Maksimal
    "ACES",  # Ace Hardware Indonesia Tbk.
    "MAPI",  # Mitra Adiperkasa Tbk.
    "ERAA",  # Erajaya Swasembada Tbk.
    "LPPF",  # Matahari Department Store Tbk.
    "RALS",  # Ramayana Lestari Sentosa Tbk.
    "SIDO",  # Industri Jamu dan Farmasi Sido Muncul Tbk.
    "ADMR",  # Adaro Minerals Indonesia Tbk. (Baru)
    "ULTJ",  # Ultrajaya Milk Industry Tbk.
    "SCMA",  # Surya Citra Media Tbk. (Media)
    "MNCN",  # Media Nusantara Citra Tbk. (Media)
    "VIVA",  # Visi Media Asia Tbk. (Media)
    "TAXI",  # Express Transindo Utama Tbk. (Transportasi)

    # Sektor Kesehatan (Health Care) - Maksimal
    "KLBF",  # Kalbe Farma Tbk.
    "MIKA",  # Mitra Keluarga Karyasehat Tbk.
    "HEAL",  # Medikaloka Hermina Tbk.
    "PRDA",  # Prodia Widyahusada Tbk.
    "INAF",  # Indofarma Tbk.
    "SAME",  # Sarana Meditama Metropolitan Tbk.
    "PPRO",  # PP Properti Tbk. (Juga di Properti)

    # Sektor Properti dan Real Estat (Properties & Real Estate) - Maksimal
    "BSDE",  # Bumi Serpong Damai Tbk.
    "PWON",  # Pakuwon Jati Tbk.
    "SMRA",  # Summarecon Agung Tbk.
    "CTRA",  # Ciputra Development Tbk.
    "APLN",  # Agung Podomoro Land Tbk.
    "LPKR",  # Lippo Karawaci Tbk.
    "DMAS",  # Puradelta Lestari Tbk.
    "KIJA",  # Kawasan Industri Jababeka Tbk.
    "MTLA",  # Metropolitan Land Tbk.
    "CTRS",  # Ciputra Residence Tbk.

    # Sektor Teknologi (Technology) - Maksimal
    "GOTO",  # GoTo Gojek Tokopedia Tbk.
    "BUKA",  # Bukalapak.com Tbk.
    "EMTK",  # Elang Mahkota Teknologi Tbk.
    "DCII",  # DCI Indonesia Tbk.
    "EDGE",  # Indointernet Tbk.
    "MLPT",  # Multipolar Technology Tbk.
    "DIGI",  # Digital Rantai Maya Tbk.
    "PMMP",  # Panca Mitra Multiperdana Tbk.

    # Sektor Infrastruktur (Infrastructure) - Maksimal
    "TLKM",  # Telkom Indonesia (Persero) Tbk.
    "EXCL",  # XL Axiata Tbk.
    "ISAT",  # Indosat Tbk.
    "TOWR",  # Sarana Menara Nusantara Tbk.
    "TBIG",  # Tower Bersama Infrastructure Tbk.
    "MTEL",  # Dayamitra Telekomunikasi Tbk.
    "PGN",    # Perusahaan Gas Negara Tbk.

    # Sektor Transportasi dan Logistik (Transportation & Logistic) - Maksimal
    "BIRD",  # Blue Bird Tbk.
    "TMAS",  # Temas Tbk.
    "SMDR",  # Samudera Indonesia Tbk.
    "ASSA",  # Adi Sarana Armada Tbk.
    "WINS",  # Wintermar Offshore Marine Tbk.
    "LINK",   # Link Net Tbk.
    "CMPP",   # Centrum Laba Sejahtera Tbk.
    "LRNA"    # Laguna Resources Tbk.
]
query_list =  ['ekonomi', 'politik', 'geo politik', 'emas', 'saham'] + emiten

# === CETAK waktu lokal Jakarta ===
now_jakarta = datetime.now(ZoneInfo("Asia/Jakarta"))
print("🕒 Waktu Jakarta:", now_jakarta.strftime("%Y-%m-%d %H:%M:%S"))
print("==============================================")

for i in query_list:
    print(f"🔎 Query: {i}")
    news = News(query=i, google_script_url="", config=config)
    news.save_to_mysql_one_by_one(table_name="google_news")
    print('')