import torch, json, os, joblib
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from data_set import MultiLabel1DDataset
import torch.nn.functional as F

class MultiLabelPredictor:
    """
    Class untuk membuat form input dan melakukan prediksi multi-label dengan model PyTorch.
    
    Args:
        model (nn.Module): Model PyTorch yang sudah diload.
        config: Konfigurasi model (input_dim, hidden_dims, output_dim, threshold, labels, dll).
        cols (list[str]): Semua kolom input.
        label_mapping (dict): Mapping label untuk kolom kategorikal.
        scaler_folder (str): Folder tempat file scaler .pkl berada.
        col_target (str): Nama kolom target (tidak ditampilkan di input form).
    """
    
    def __init__(self, model, config, label_mapping, scaler_folder, dimension='1d'):
        self.model = model
        self.config = config
        self.label_mapping = label_mapping
        self.scaler_folder = scaler_folder
        self.dimension  = dimension
        
        self.feature_cols = []
        for key, value in vars(self.config).items():
            if key.startswith("feature") and key.endswith("_cols"):
                if isinstance(value, list):
                    self.feature_cols.extend(value)
        
        self.numeric_cols = [c for c in self.feature_cols if 'label_' not in c]
        self.radios, self.inputs = [], []
        self._build_widgets()
        self._display_form()
    
    # ==================== INTERNAL: Build widgets ====================
    def _build_widgets(self):
        ultra_compact = widgets.Layout(width='90%', margin='2px 0')
        
        for c in self.feature_cols:
            if 'label_' in c:
                label_name = c.replace('label_', '')
                options = self.label_mapping[label_name]
                radio = widgets.RadioButtons(
                    options=options,
                    description=f'🎯 {label_name}:',
                    disabled=False,
                    layout=ultra_compact,
                    style={'description_width': '120px'}
                )
                self.radios.append((c, radio))
            elif c in self.numeric_cols:
                scaler_path = os.path.join(self.scaler_folder, f'scaler_{c}.pkl')
                placeholder = f"Value for {c}"
                if os.path.exists(scaler_path):
                    scaler_col = joblib.load(scaler_path)
                    placeholder += f" (min:{scaler_col.data_min_[0]} | max:{scaler_col.data_max_[0]})"
                inputan = widgets.Text(
                    value='',
                    placeholder=placeholder,
                    description=f'📊 {c}:',
                    disabled=False,
                    layout=ultra_compact,
                    style={'description_width': '320px'}
                )
                self.inputs.append((c, inputan))
        
        # Tombol & output
        self.btn = widgets.Button(description='🚀 Predict', button_style='primary', layout=widgets.Layout(width='120px', margin='8px 0'))
        self.out_w = widgets.Output()
        self.btn.on_click(self._predict_click)
    
    # ==================== INTERNAL: Fungsi bantu ====================
    def _cekPersen(self, predictions, percentages, labels):
        print("\n📊 HASIL PREDIKSI")
        print("┌" + "─" * 48 + "┐")
        for i, label in enumerate(labels):
            pred_value = predictions[0][i]
            percentage_value = percentages[0][i]
            
            bar_length = 15
            filled_length = int(percentage_value / 100 * bar_length)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            status_icon = "✅" if pred_value == 1 else "❌"
            
            if percentage_value >= 70:
                prob_text = f"\033[92m{percentage_value:5.1f}%\033[0m"  # Hijau
            elif percentage_value >= 40:
                prob_text = f"\033[93m{percentage_value:5.1f}%\033[0m"  # Kuning
            else:
                prob_text = f"\033[91m{percentage_value:5.1f}%\033[0m"  # Merah
            
            print(f"│ {status_icon} {label:15} {bar} {prob_text} │")
        print("└" + "─" * 48 + "┘")
    
    def _cekMultiLabel(self, out, labels, threshold=None):
        if threshold is None: threshold = self.config.threshold
        return [labels[i] for i, val in enumerate(out) if val > threshold]
    
    # ==================== INTERNAL: Fungsi prediksi tombol ====================
    def _predict_click(self, b):
        with self.out_w:
            clear_output()
            print("📋 INPUT DATA")
            print("─" * 30)
            
            # Ambil semua nilai
            data_dict = {col: r.value for col, r in self.radios}
            for col, i in self.inputs:
                val = i.value.strip()
                try: val = float(val)
                except: pass
                data_dict[col] = val if val != '' else None
            
            df_input = pd.DataFrame([data_dict])
            df_input = df_input[self.feature_cols]
            display(df_input)
            
            # Scaling numerik
            df_scaled = df_input.copy()
            for col in self.numeric_cols:
                scaler_path = os.path.join(self.scaler_folder, f'scaler_{col}.pkl')
                if os.path.exists(scaler_path):
                    scaler_col = joblib.load(scaler_path)
                    df_scaled[[col]] = scaler_col.transform(df_input[[col]])
                    print(f"✅ '{col}' scaled (min:{scaler_col.data_min_[0]} | max:{scaler_col.data_max_[0]})")
            
            
            # Konversi ke tensor
            
            print("\n⚙️ DATA SETELAH SCALING")
            print("─" * 30)
            display(df_scaled)
            
            
            if(self.dimension=='1d'):
                X = torch.tensor(df_scaled.values, dtype=torch.float32)
                # Prediksi
                with torch.no_grad():
                    output = self.model(X)
                    predictions = (output > self.config.threshold).float().cpu().numpy()
                    percentages = output.cpu().numpy() * 100
                    pred = (output > self.config.threshold).float()
                    o = self._cekMultiLabel(list(pred[0]), self.config.labels)
                    
            elif(self.dimension=='2d'):
                X1 = torch.tensor(df_scaled[self.config.feature_1_cols].values, dtype=torch.float32)
                X2 = torch.tensor(df_scaled[self.config.feature_2_cols].values, dtype=torch.float32)
                # Prediksi
                with torch.no_grad():
                    output = self.model(X1,X2)
                    predictions = (output > self.config.threshold).float().cpu().numpy()
                    percentages = output.cpu().numpy() * 100
                    pred = (output > self.config.threshold).float()
                    o = self._cekMultiLabel(list(pred[0]), self.config.labels)
                    
            elif(self.dimension=='3d'):
                X1 = torch.tensor(df_scaled[self.config.feature_1_cols].values, dtype=torch.float32)
                X2 = torch.tensor(df_scaled[self.config.feature_2_cols].values, dtype=torch.float32)
                X3 = torch.tensor(df_scaled[self.config.feature_3_cols].values, dtype=torch.float32)
                # Prediksi
                with torch.no_grad():
                    output = self.model(X1,X2,X3)
                    predictions = (output > self.config.threshold).float().cpu().numpy()
                    percentages = output.cpu().numpy() * 100
                    pred = (output > self.config.threshold).float()
                    o = self._cekMultiLabel(list(pred[0]), self.config.labels)
            
            self._cekPersen(predictions, percentages, self.config.labels)
    
    # ==================== Tampilkan form ====================
    def _display_form(self):
        MINIMAL_STYLE = """
        <style>
        .minimal-container { background: #f8f9fa; padding: 12px; border-radius: 6px; border: 1px solid #dee2e6; }
        .minimal-item { margin: 3px 0; }
        .radio-label { font-weight: bold; color: #333; }
        .input-label { font-weight: bold; color: #555; }
        </style>
        """
        display(widgets.HTML(value=MINIMAL_STYLE))
        form_widgets = [w for _, w in self.radios] + [w for _, w in self.inputs] + [self.btn, self.out_w]
        minimal_form = widgets.VBox(form_widgets, layout=widgets.Layout(padding='8px'))
        display(minimal_form)



class MultiClassPredictor:
    """
    Class untuk membuat form input dan melakukan prediksi multi-class dengan model PyTorch.
    
    Args:
        model (nn.Module): Model PyTorch yang sudah diload.
        config: Konfigurasi model (input_dim, hidden_dims, output_dim, labels, dll).
        label_mapping (dict): Mapping label untuk kolom kategorikal.
        scaler_folder (str): Folder tempat file scaler .pkl berada.
        dimension (str): '1d', '2d', atau '3d' sesuai model.
    """
    
    def __init__(self, model, config, label_mapping, scaler_folder, dimension='1d'):
        self.model = model
        self.model.eval()
        self.config = config
        self.label_mapping = label_mapping
        self.scaler_folder = scaler_folder
        self.dimension  = dimension
        
        # Gabungkan semua feature_cols otomatis dari config
        self.feature_cols = []
        for key, value in vars(self.config).items():
            if key.startswith("feature") and key.endswith("_cols"):
                if isinstance(value, list):
                    self.feature_cols.extend(value)
        
        self.numeric_cols = [c for c in self.feature_cols if 'label_' not in c]
        self.radios, self.inputs = [], []
        self._build_widgets()
        self._display_form()
    
    # ==================== INTERNAL: Build widgets ====================
    def _build_widgets(self):
        ultra_compact = widgets.Layout(width='90%', margin='2px 0')
        
        for c in self.feature_cols:
            if 'label_' in c:
                label_name = c.replace('label_', '')
                options = self.label_mapping[label_name]
                radio = widgets.RadioButtons(
                    options=options,
                    description=f'🎯 {label_name}:',
                    disabled=False,
                    layout=ultra_compact,
                    style={'description_width': '120px'}
                )
                self.radios.append((c, radio))
            elif c in self.numeric_cols:
                scaler_path = os.path.join(self.scaler_folder, f'scaler_{c}.pkl')
                placeholder = f"Value for {c}"
                if os.path.exists(scaler_path):
                    scaler_col = joblib.load(scaler_path)
                    placeholder += f" (min:{scaler_col.data_min_[0]} | max:{scaler_col.data_max_[0]})"
                inputan = widgets.Text(
                    value='',
                    placeholder=placeholder,
                    description=f'📊 {c}:',
                    disabled=False,
                    layout=ultra_compact,
                    style={'description_width': '320px'}
                )
                self.inputs.append((c, inputan))
        
        # Tombol & output
        self.btn = widgets.Button(
            description='🚀 Predict',
            button_style='primary',
            layout=widgets.Layout(width='120px', margin='8px 0')
        )
        self.out_w = widgets.Output()
        self.btn.on_click(self._predict_click)
    
    # ==================== INTERNAL: Fungsi bantu ====================
    def _cekPersenMultiClass(self, probs, labels):
        print("\n📊 HASIL PREDIKSI (Multi-class)")
        print("┌" + "─" * 60 + "┐")
        for i, label in enumerate(labels):
            percentage_value = probs[0][i] * 100
            bar_length = 15
            filled_length = int(percentage_value / 100 * bar_length)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            if percentage_value >= 70:
                prob_text = f"\033[92m{percentage_value:5.1f}%\033[0m"  # Hijau
            elif percentage_value >= 40:
                prob_text = f"\033[93m{percentage_value:5.1f}%\033[0m"  # Kuning
            else:
                prob_text = f"\033[91m{percentage_value:5.1f}%\033[0m"  # Merah
            print(f"│ {labels[i]:15} {label:15} {bar} {prob_text} │")
        print("└" + "─" * 60 + "┘")
    
    # ==================== INTERNAL: Fungsi prediksi tombol ====================
    def _predict_click(self, b):
        with self.out_w:
            clear_output()
            print("📋 INPUT DATA")
            print("─" * 30)
            
            # Ambil semua nilai input
            data_dict = {col: r.value for col, r in self.radios}
            for col, i in self.inputs:
                val = i.value.strip()
                try: val = float(val)
                except: pass
                data_dict[col] = val if val != '' else None
            
            df_input = pd.DataFrame([data_dict])
            df_input = df_input[self.feature_cols]
            display(df_input)
            
            # Scaling numerik
            df_scaled = df_input.copy()
            for col in self.numeric_cols:
                scaler_path = os.path.join(self.scaler_folder, f'scaler_{col}.pkl')
                if os.path.exists(scaler_path):
                    scaler_col = joblib.load(scaler_path)
                    df_scaled[[col]] = scaler_col.transform(df_input[[col]])
                    print(f"✅ '{col}' scaled (min:{scaler_col.data_min_[0]} | max:{scaler_col.data_max_[0]})")
            
            print("\n⚙️ DATA SETELAH SCALING")
            print("─" * 30)
            display(df_scaled)
            
            # Konversi ke tensor dan prediksi
            if(self.dimension=='1d'):
                X = torch.tensor(df_scaled.values, dtype=torch.float32)
                with torch.no_grad():
                    logits = self.model(X)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                    
            elif(self.dimension=='2d'):
                X1 = torch.tensor(df_scaled[self.config.feature_1_cols].values, dtype=torch.float32)
                X2 = torch.tensor(df_scaled[self.config.feature_2_cols].values, dtype=torch.float32)
                # Prediksi
                with torch.no_grad():
                    logits = self.model(X1,X2)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                    
            elif(self.dimension=='3d'):
                X1 = torch.tensor(df_scaled[self.config.feature_1_cols].values, dtype=torch.float32)
                X2 = torch.tensor(df_scaled[self.config.feature_2_cols].values, dtype=torch.float32)
                X3 = torch.tensor(df_scaled[self.config.feature_3_cols].values, dtype=torch.float32)
                # Prediksi
                with torch.no_grad():
                    logits = self.model(X1,X2,X3)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                
            label_pred = self.config.labels[preds[0]]
            print(f"\n✅ Prediksi akhir: \033[1;32m{label_pred}\033[0m")
            print(f"\n logits: {logits}")
            print(f"\n  probs: {probs}")
            print(f"\n  preds: {preds}")
            self._cekPersenMultiClass(probs, self.config.labels)
    
    # ==================== Tampilkan form ====================
    def _display_form(self):
        MINIMAL_STYLE = """
        <style>
        .minimal-container { background: #f8f9fa; padding: 12px; border-radius: 6px; border: 1px solid #dee2e6; }
        .minimal-item { margin: 3px 0; }
        .radio-label { font-weight: bold; color: #333; }
        .input-label { font-weight: bold; color: #555; }
        </style>
        """
        display(widgets.HTML(value=MINIMAL_STYLE))
        form_widgets = [w for _, w in self.radios] + [w for _, w in self.inputs] + [self.btn, self.out_w]
        minimal_form = widgets.VBox(form_widgets, layout=widgets.Layout(padding='8px'))
        display(minimal_form)


class RegressionPredictor:
    """
    Class untuk membuat form input dan melakukan prediksi REGRESI dengan model PyTorch.
    
    Args:
        model (nn.Module): Model PyTorch yang sudah diload.
        config: Konfigurasi model (input_dim, hidden_dims, output_dim, dll).
        label_mapping (dict): Mapping label untuk kolom kategorikal (opsional).
        scaler_folder (str): Folder tempat file scaler .pkl berada.
        dimension (str): '1d', '2d', atau '3d' sesuai model.
    """
    
    def __init__(self, model, config, label_mapping=None, scaler_folder=None, dimension='1d'):
        self.model = model
        self.model.eval()
        self.config = config
        self.label_mapping = label_mapping or {}
        self.scaler_folder = scaler_folder
        self.dimension = dimension
        
        # Gabungkan semua feature_cols otomatis dari config
        self.feature_cols = []
        for key, value in vars(self.config).items():
            if key.startswith("feature") and key.endswith("_cols"):
                if isinstance(value, list):
                    self.feature_cols.extend(value)
        
        self.numeric_cols = [c for c in self.feature_cols if not c.startswith("label_")]
        self.radios, self.inputs = [], []
        self._build_widgets()
        self._display_form()

    # ==================== INTERNAL: Build widgets ====================
    def _build_widgets(self):
        ultra_compact = widgets.Layout(width='90%', margin='2px 0')
        
        for c in self.feature_cols:
            if c.startswith("label_"):
                label_name = c.replace("label_", "")
                options = self.label_mapping.get(label_name, [])
                radio = widgets.RadioButtons(
                    options=options,
                    description=f'🎯 {label_name}:',
                    disabled=False,
                    layout=ultra_compact,
                    style={'description_width': '120px'}
                )
                self.radios.append((c, radio))
            else:
                placeholder = f"Value for {c}"
                if self.scaler_folder:
                    scaler_path = os.path.join(self.scaler_folder, f'scaler_{c}.pkl')
                    if os.path.exists(scaler_path):
                        scaler_col = joblib.load(scaler_path)
                        placeholder += f" (min:{scaler_col.data_min_[0]} | max:{scaler_col.data_max_[0]})"
                inputan = widgets.Text(
                    value='',
                    placeholder=placeholder,
                    description=f'📊 {c}:',
                    disabled=False,
                    layout=ultra_compact,
                    style={'description_width': '320px'}
                )
                self.inputs.append((c, inputan))
        
        # Tombol & output
        self.btn = widgets.Button(
            description='🚀 Predict',
            button_style='primary',
            layout=widgets.Layout(width='120px', margin='8px 0')
        )
        self.out_w = widgets.Output()
        self.btn.on_click(self._predict_click)

    def denormalize_value(self,value, col_target, scaler_folder):
        """
        Mengembalikan nilai hasil normalisasi menjadi nilai asli menggunakan scaler tersimpan.
        
        Args:
            value (float | list | np.ndarray): Nilai atau array hasil normalisasi.
            col_target (str): Nama kolom target (misal 'berat', 'usia', dsb).
            scaler_folder (str): Folder tempat scaler disimpan.
    
        Returns:
            np.ndarray: Nilai asli setelah inverse transform.
        """
        scaler_path = os.path.join(scaler_folder, f'scaler_{col_target}.pkl')
    
        if not os.path.exists(scaler_path):
            return value.flatten()
            #raise FileNotFoundError(f"Scaler untuk kolom '{col_target}' tidak ditemukan di {scaler_folder}")
    
        # Load scaler untuk kolom target
        scaler = joblib.load(scaler_path)
    
        # Pastikan bentuk data sesuai (2D)
        value = np.array(value).reshape(-1, 1)
    
        # Kembalikan ke nilai asli
        original_value = scaler.inverse_transform(value)
    
        return original_value.flatten()

    # ==================== INTERNAL: Fungsi prediksi tombol ====================
    def _predict_click(self, b):
        with self.out_w:
            clear_output()
            print("📋 INPUT DATA")
            print("─" * 30)
            
            # Ambil semua nilai input
            data_dict = {col: r.value for col, r in self.radios}
            for col, i in self.inputs:
                val = i.value.strip()
                try:
                    val = float(val)
                except:
                    pass
                data_dict[col] = val if val != '' else None
            
            df_input = pd.DataFrame([data_dict])
            df_input = df_input[self.feature_cols]
            display(df_input)
            
            # Scaling numerik
            df_scaled = df_input.copy()
            if self.scaler_folder:
                for col in self.numeric_cols:
                    scaler_path = os.path.join(self.scaler_folder, f'scaler_{col}.pkl')
                    if os.path.exists(scaler_path):
                        scaler_col = joblib.load(scaler_path)
                        df_scaled[[col]] = scaler_col.transform(df_input[[col]])
                        print(f"✅ '{col}' scaled (min:{scaler_col.data_min_[0]} | max:{scaler_col.data_max_[0]})")
            
            print("\n⚙️ DATA SETELAH SCALING")
            print("─" * 30)
            display(df_scaled)
            
            # Prediksi
            with torch.no_grad():
                if self.dimension == '1d':
                    X = torch.tensor(df_scaled.values, dtype=torch.float32)
                    output = self.model(X)
                elif self.dimension == '2d':
                    X1 = torch.tensor(df_scaled[self.config.feature_1_cols].values, dtype=torch.float32)
                    X2 = torch.tensor(df_scaled[self.config.feature_2_cols].values, dtype=torch.float32)
                    output = self.model(X1, X2)
                elif self.dimension == '3d':
                    X1 = torch.tensor(df_scaled[self.config.feature_1_cols].values, dtype=torch.float32)
                    X2 = torch.tensor(df_scaled[self.config.feature_2_cols].values, dtype=torch.float32)
                    X3 = torch.tensor(df_scaled[self.config.feature_3_cols].values, dtype=torch.float32)
                    output = self.model(X1, X2, X3)
                else:
                    raise ValueError("dimension harus '1d', '2d', atau '3d'")
            
            pred_value = str(output.cpu().numpy().flatten())
            s = self.denormalize_value(output, self.config.col_target, self.scaler_folder)
            
            print("\n✅ HASIL PREDIKSI")
            print("─" * 30)
            print(f"📈 Prediksi nilai kontinu: \033[1;32m{str(s)}\033[0m")
    
    # ==================== Tampilkan form ====================
    def _display_form(self):
        MINIMAL_STYLE = """
        <style>
        .minimal-container { background: #f8f9fa; padding: 12px; border-radius: 6px; border: 1px solid #dee2e6; }
        .minimal-item { margin: 3px 0; }
        .radio-label { font-weight: bold; color: #333; }
        .input-label { font-weight: bold; color: #555; }
        </style>
        """
        display(widgets.HTML(value=MINIMAL_STYLE))
        form_widgets = [w for _, w in self.radios] + [w for _, w in self.inputs] + [self.btn, self.out_w]
        minimal_form = widgets.VBox(form_widgets, layout=widgets.Layout(padding='8px'))
        display(minimal_form)
