from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from tqdm import tqdm
import json
class CandlePatternMatcher:
    '''
     format historical_data:
        Timestamp,Open,High,Low,Close,Volume
    '''
    def __init__(self, 
                    historical_data: pd.DataFrame,
                    start_time: str,
                    max_lookback: int = 0,
                    sample_data_test = 2,
                    window_size = 6,
                    threshold   = {
                        "persentase_identik" : 11,
                        "mse" : 0.018,
                        "rmse": 0.1,
                        "cosine_similarity" : 96,
                        "pearson_similarity": 90
                    },
                    numeric_cols=['Open', 'High', 'Low', 'Close']
                ):
        self.historical_data = historical_data
        self.start_time = pd.Timestamp(start_time)
        self.max_lookback = max_lookback #0 is to all data
        self.sample_data_test = sample_data_test
        self.window_size = window_size
        self.threshold = threshold
        self.numeric_cols = numeric_cols
    
    #Global Normalization
    #-------------------------------------------------------------------------
    def gobal_norm(self, df: pd.DataFrame):
        global_min = df[self.numeric_cols].min().min()
        global_max = df[self.numeric_cols].max().max()
        return round((df[self.numeric_cols] - global_min) / (global_max - global_min), 3)
    
    #Persentase Identik
    #-------------------------------------------------------------------------
    def persentase_identik(self, matrik_utama, matrik_pembanding):
        """
        Menghitung persentase kesamaan elemen (nilai identik) antara dua matriks.
        Args:
            matrik_utama (list[list] | np.ndarray): Matriks pertama (referensi)
            matrik_pembanding (list[list] | np.ndarray): Matriks kedua (dibandingkan)
        Returns:
            float: Persentase kesamaan dalam rentang 0–100
        """
        # Konversi ke numpy array
        a = np.array(matrik_utama)
        b = np.array(matrik_pembanding)

        # Pastikan ukuran sama
        if a.shape != b.shape:
            raise ValueError(f"Ukuran matriks berbeda: {a.shape} vs {b.shape}")

        # Hitung jumlah elemen identik
        sama = np.sum(a == b)
        total = a.size

        # Persentase identik
        persentase = (sama / total) * 100
        return round(persentase, 2)
    
    #Mean Squared Error (MSE) atau Root Mean Squared Error (RMSE)
    #-------------------------------------------------------------------------
    def mse_rmse(self, matrik_utama, matrik_pembanding):
        """
        Menghitung Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE)
        antara dua matriks dengan ukuran sama.
        Args:
            matrik_utama (list[list] | np.ndarray): Matriks referensi
            matrik_pembanding (list[list] | np.ndarray): Matriks pembanding
        Returns:
            dict: {'MSE': float, 'RMSE': float}
        """
        # Konversi ke numpy array
        a = np.array(matrik_utama, dtype=float)
        b = np.array(matrik_pembanding, dtype=float)

        # Pastikan ukuran sama
        if a.shape != b.shape:
            raise ValueError(f"Ukuran matriks berbeda: {a.shape} vs {b.shape}")

        # Hitung selisih
        diff = a - b

        # Mean Squared Error
        mse = np.mean(diff ** 2)

        # Root Mean Squared Error
        rmse = np.sqrt(mse)

        return {'MSE': round(mse, 4), 'RMSE': round(rmse, 4)}
    
    #Cosine Similarity
    #-------------------------------------------------------------------------
    def cosine_similarity_matrix(self,matrik_utama, matrik_pembanding, mode='persen'):
        """
        Menghitung Cosine Similarity antara dua matriks.
        Args:
            matrik_utama (list[list] | np.ndarray): Matriks referensi
            matrik_pembanding (list[list] | np.ndarray): Matriks pembanding
            mode (str): 'persen' untuk output 0–100%, 
                        'index' untuk output -1 sampai 1
        Returns:
            float: Nilai cosine similarity sesuai mode
        """
        # Konversi ke numpy array
        a = np.array(matrik_utama, dtype=float)
        b = np.array(matrik_pembanding, dtype=float)

        # Pastikan ukuran sama
        if a.shape != b.shape:
            raise ValueError(f"Ukuran matriks berbeda: {a.shape} vs {b.shape}")

        # Flatten menjadi vektor 1D
        a_flat = a.flatten()
        b_flat = b.flatten()

        # Hitung Cosine Similarity (hasil antara -1 sampai 1)
        sim = cosine_similarity(a_flat.reshape(1, -1), b_flat.reshape(1, -1))[0][0]

        # Pilih mode output
        if mode == 'persen':
            return round(sim * 100, 2)
        elif mode == 'index':
            return round(sim, 4)
        else:
            raise ValueError("Parameter 'mode' harus 'persen' atau 'index'.")
        
    
    #Pearson Correlation
    #-------------------------------------------------------------------------
    def pearson_similarity_matrix(self, matrik_utama, matrik_pembanding, mode="persen"):
        """
        Menghitung kesamaan dua matriks berdasarkan Pearson Correlation.
        Args:
            matrik_utama (list or np.ndarray): Matriks utama.
            matrik_pembanding (list or np.ndarray): Matriks pembanding.
            mode (str): 'index' untuk nilai -1 s.d. 1, 
                        'persen' untuk output 0–100%.
        Returns:
            float: Nilai korelasi atau persentase kesamaan.
        """
        a = np.array(matrik_utama).flatten()
        b = np.array(matrik_pembanding).flatten()

        if len(a) != len(b):
            raise ValueError("Kedua matriks harus memiliki jumlah elemen yang sama.")
        
        corr, _ = pearsonr(a, b)
        
        if mode == "persen":
            return round((corr + 1) / 2 * 100, 2)  # ubah -1–1 menjadi 0–100%
        else:
            return round(corr, 4)
    
    #Pattern Matcher
    #-------------------------------------------------------------------------
    def run_pattern_match(self,df:pd.DataFrame, target_norm, window_size, idx_target, numeric_cols, threshold:dict, max_lookback: int = 0):
        """
        Mencocokkan pola candlestick berdasarkan metrik kesamaan.
        
        Args:
            df (pd.DataFrame): Data sumber dengan kolom numerik dan Timestamp.
            target_norm (pd.DataFrame): Data window target yang sudah dinormalisasi.
            window_size (int): Ukuran window perbandingan.
            idx_target (int): Index posisi target terakhir.
            numeric_cols (list): Kolom numerik yang digunakan.
            threshold (dict): Batas nilai minimum/maksimum untuk filter hasil.
            max_lookback (int): Jumlah maksimum data historis yang digunakan untuk pencarian.
                Jika 0, maka menggunakan seluruh data sebelum idx_target.
        Returns:
            list[dict]: Daftar hasil yang lolos threshold.
        """
        results = []

        # Hitung batas awal jika ada max_lookback
        if max_lookback > 0:
            start_idx = max(window_size - 1, idx_target - max_lookback)
        else:
            start_idx = window_size - 1

        # progress bar
        pbar = tqdm(range(start_idx, idx_target), desc="Processing windows", unit="window")


        for i in pbar:
            past_window = df.iloc[i - window_size + 1 : i + 1]
            if len(past_window) < window_size:
                continue
            
            # normalisasi window lama
            past_norm = self.gobal_norm(past_window)
            
            # flatten untuk perbandingan
            a = target_norm[numeric_cols].values.tolist()
            b = past_norm[numeric_cols].values.tolist()
            
            # hitung semua metrik
            persentase_identik_val = self.persentase_identik(a, b)
            mse_rmse_val = self.mse_rmse(a, b)
            cosine_similarity_matrix_val = self.cosine_similarity_matrix(a, b, mode='persen')
            pearson_similarity_matrix_val = self.pearson_similarity_matrix(a, b, mode='persen')
            
            # cek apakah lolos semua threshold
            if (
                cosine_similarity_matrix_val >= threshold["cosine_similarity"] and
                persentase_identik_val >= threshold["persentase_identik"] and
                mse_rmse_val["MSE"] <= threshold["mse"] and
                pearson_similarity_matrix_val >= threshold["pearson_similarity"]
            ):
                tmp = {
                    "timestamp": past_window["Timestamp"].values.tolist(),
                    "data": b,
                    "val_persentase_identik": persentase_identik_val,
                    "val_mse": mse_rmse_val["MSE"],
                    "val_rmse": mse_rmse_val["RMSE"],
                    "val_cosine_similarity": cosine_similarity_matrix_val,
                    "val_pearson_similarity": pearson_similarity_matrix_val,
                }

                # prediksi hasil
                if i + 1 < len(df):
                    next_close = df.iloc[i + 1]["Close"]
                    current_close = df.iloc[i]["Close"]
                    if next_close > current_close:
                        tmp["result"] = "UP"
                    elif next_close < current_close:
                        tmp["result"] = "DOWN"
                    else:
                        tmp["result"] = "HOLD"
                else:
                    tmp["result"] = "HOLD"

                results.append(tmp)
            
            # update jumlah hasil di progress bar
            pbar.set_postfix({"matches": len(results)})

        pbar.close()
        print(f"\nTotal hasil lolos threshold: {len(results)}")
        return results
    
    #test
    #---------------------------------------------------------------------------
    def test(self):
        benar = 0
        data = []

        df = self.historical_data

        # Cari index target berdasarkan Timestamp
        idx_target_arr = df.index[df['Timestamp'] == self.start_time]
        if len(idx_target_arr) == 0:
            raise ValueError("Timestamp tidak ditemukan dalam data.")
        idx_target = idx_target_arr[0]

        print(f"Mulai pengujian dari timestamp: {self.start_time}")

        # Loop sample_data secara berurutan
        for i in range(self.sample_data_test):
            current_idx = idx_target + i

            # Pastikan tidak keluar dari batas DataFrame
            if current_idx + 1 >= len(df):
                break  

            # Ambil window data
            target_window = df.iloc[current_idx - self.window_size + 1 : current_idx + 1][['Timestamp','Open','High','Low','Close']]
            if len(target_window) < self.window_size:
                continue 

            # Simpan Timestamp
            timestamps = target_window['Timestamp']

            # Normalisasi hanya kolom numerik
            target_scaled = self.gobal_norm(target_window)

            # Gabungkan kembali dengan Timestamp
            target_norm = target_window.copy()
            target_norm[self.numeric_cols] = target_scaled
            
            # Hasil Next Time (gunakan nilai asli, bukan normalisasi)
            next_row = df.iloc[current_idx + 1]
            current_close = df.iloc[current_idx]['Close']
            real_out = (
                'UP' if next_row['Close'] > current_close else
                'DOWN' if next_row['Close'] < current_close else
                'HOLD'
            )

            print(f"\n{i+1}. {timestamps.iloc[0]} → {timestamps.iloc[-1]} -> Real: {real_out}")

            # Jalankan pattern matching
            results = self.run_pattern_match(
                df=df,
                target_norm=target_norm,
                window_size=self.window_size,
                idx_target=current_idx,  # gunakan current index yang sedang diuji
                numeric_cols=self.numeric_cols,
                threshold=self.threshold,
                max_lookback=self.max_lookback
            )

            tmp = {
                "test_index": i + 1,
                "timestamp_start": str(timestamps.iloc[0]),
                "timestamp_end": str(timestamps.iloc[-1]),
                "real_out": real_out,
                "pattern_match": len(results),
                "results": results
            }

            # Jika tidak ada hasil
            if not results:
                print("   ⚠️  Tidak ada pola yang cocok ditemukan.")
                continue

            # Hitung hasil kecocokan
            print("   📊 Hasil kecocokan:")
            c_up, c_down, c_hold = 0, 0, 0

            for j, row in enumerate(results):
                ts = row.get('timestamp')
                ts_start = str(pd.to_datetime(ts[0], unit='ns'))
                ts_end = str(pd.to_datetime(ts[-1], unit='ns'))
                res = row.get('result', '-')
                val_persentase_identik = row.get('val_persentase_identik', 0)
                val_mse = row.get('val_mse', 0)
                val_rmse = row.get('val_rmse', 0)
                val_cos = row.get('val_cosine_similarity', 0)
                val_pearson = row.get('val_pearson_similarity', 0)

                if res == 'UP':
                    c_up += 1
                elif res == 'DOWN':
                    c_down += 1
                elif res == 'HOLD':
                    c_hold += 1

                print(
                    f"      {j+1:02d}. {ts_start} → {ts_end} | "
                    f"Identik={val_persentase_identik:.3f} | MSE={val_mse:.3f} | "
                    f"RMSE={val_rmse:.3f} | Cosine={val_cos:.3f} | "
                    f"Pearson={val_pearson:.3f} | Result: {res}"
                )

            # Hitung total hasil
            total = c_up + c_down + c_hold
            if total > 0:
                p_up = (c_up / total) * 100
                p_down = (c_down / total) * 100
                p_hold = (c_hold / total) * 100

                # Tentukan hasil dominan
                dominant = max(
                    [('UP', p_up), ('DOWN', p_down), ('HOLD', p_hold)],
                    key=lambda x: x[1]
                )
                tmp['dominant'] = dominant

                if dominant[0] == real_out:
                    benar += 1
                    tmp['is_correct'] = True
                    print(f"   ✅ Sesuai hasil sebenarnya ({real_out})")
                else:
                    tmp['is_correct'] = False
                    print(f"   ❌ Tidak sesuai (Prediksi: {dominant[0]} | Real: {real_out})")

                print(f"   🔹 UP:   {c_up} ({p_up:.2f}%)")
                print(f"   🔹 DOWN: {c_down} ({p_down:.2f}%)")
                print(f"   🔹 HOLD: {c_hold} ({p_hold:.2f}%)")
                print(f"   🏁 Dominan: {dominant[0]} ({dominant[1]:.2f}%)")
            else:
                print("   ⚠️ Tidak ada hasil yang valid untuk dihitung.")
                tmp['is_correct'] = False

            data.append(tmp)
            print('-----------------------------------------------------------------------')

        # Ringkasan akhir
        print("\n=== 📈 Ringkasan Akurasi ===")
        total_test = len(data)
        print(f"✅ Total benar: {benar} dari {total_test} percobaan.")
        if total_test > 0:
            print(f"🎯 Akurasi: {benar / total_test * 100:.2f}%")

        return data




file_name = "btc_data_hourly.csv"
df = pd.read_csv('data/' + file_name)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

start_time = '2023-08-01 00:00:00'

matcher = CandlePatternMatcher(
    historical_data=df,
    start_time=start_time,
    max_lookback=50000,
    sample_data_test=20,
    window_size=6,
    threshold={
        "persentase_identik" : 11,
        "mse" : 0.03, #0.018,
        "rmse": 0.2, #0.1,
        "cosine_similarity" : 90,#96,
        "pearson_similarity": 90
    },
    numeric_cols=['Open', 'High', 'Low', 'Close']
)
out = matcher.test()
#save to json
with open('pattern_match_results.json', 'w') as f:
    json.dump(out, f, indent=4, default=str)    
