import numpy as np
from tqdm import tqdm
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

#List multi label
#-> params : mode, dataset, dataloader, model, criterion, optimizer, device, threshold=0.6
#-----------------------------------------------------------------------------------------------------------------------
#loop_multi_label_classification_timeseris
#   |----> untuk training dan testing model multi-label classification berbasis timeseris. (RNN, LSTM, atau GRU).
#
#loop_multi_label_classification_1D
#   |----> untuk training dan testing model multi-label classification 1D.
#
#loop_multi_label_classification_2D
#   |----> untuk training dan testing model multi-label classification dengan 2 input (features_1, features_2).
#
#loop_multi_label_classification_3D
#   |----> untuk training dan testing model multi-label classification dengan 3 input (features_1, features_2, features_3).
#-----------------------------------------------------------------------------------------------------------------------


#List multi class
#-> params : mode, dataset, dataloader, model, criterion, optimizer, device
#-----------------------------------------------------------------------------------------------------------------------
#loop_multi_class_classification_timeseris
#   |----> untuk training & testing model multi-class classification berbasis timeseris. (RNN, LSTM, atau GRU).
#
#loop_multi_class_classification_1D
#   |----> untuk training dan testing model multi-class classification 1D.
#
#loop_multi_class_classification_2D
#   |----> untuk training dan testing model multi-class classification dengan 2 input (features_1, features_2).
#
#loop_multi_class_classification_2D
#   |----> untuk training dan testing model multi-class classification dengan 3 input (features_1, features_2, features_3).
#
#-----------------------------------------------------------------------------------------------------------------------


#List Regression
#-> params : mode,dataset,dataloader,model,criterion,optimizer,device
#-----------------------------------------------------------------------------------------------------------------------
#loop_regression_timeseries
#   |----> Loop untuk training dan testing model regresi berbasis time series.
#
#loop_regression_1D
#   |---->Loop untuk training dan testing model regresi 1D.
#
#loop_regression_2D
#   |---->Loop untuk training dan testing model regresi 2D. dengan 2 input (features_1, features_2).
#
#loop_regression_3D
#   |---->Loop untuk training dan testing model regresi 2D. dengan 3 input (features_1, features_2, features_3).
#
#-----------------------------------------------------------------------------------------------------------------------

#========================================================================================================================
#                                     Classification MULTI LABEL
#========================================================================================================================

#Loop untuk training dan testing model multi-label classification berbasis timeseris.  (RNN, LSTM, atau GRU).
#========================================================================================================================
def loop_multi_label_classification_timeseris(mode, dataset, dataloader, model, criterion, optimizer, device, threshold=0.6):
    """
    Loop untuk training dan testing model multi-label classification berbasis LSTM.

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch (LSTM)
        criterion (loss function): fungsi loss
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU
        threshold (float): ambang batas sigmoid untuk menentukan label aktif (1) atau tidak (0)

    Returns:
        dict: berisi metrik evaluasi model
            - loss: nilai loss rata-rata
            - hamming_loss: rata-rata kesalahan per label
            - subset_accuracy: proporsi prediksi yang semua label-nya benar
            - micro_f1: F1-score global (berbasis seluruh label)
            - macro_f1: rata-rata F1-score antar label
            - overall_accuracy: total proporsi elemen benar
            - per_label_accuracy: akurasi per label (list)
    
    ----------------------------------------------------------------------
    📘 Penjelasan Metrik:

    Metrik                 | Arti
    -----------------------|---------------------------------------------
    Hamming Loss           | Rata-rata kesalahan per label. Semakin kecil semakin baik.
    Subset Accuracy        | Persentase sampel di mana *semua* label benar. Sangat ketat.
    Micro F1-score         | F1-score global yang menyeimbangkan presisi & recall seluruh label.
    Macro F1-score         | Rata-rata F1-score per label (tiap label bobot sama).
    Overall Accuracy       | Total prediksi benar dari seluruh label dan sampel.
    Per-label Accuracy     | Akurasi per label, berguna untuk melihat label sulit.
    ----------------------------------------------------------------------
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")

    if mode == "train":
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_batches = 0
    all_predictions, all_targets = [], []

    for features, target in tqdm(dataloader, desc=mode):
        features, target = features.to(device), target.to(device)
        
        # --- Pastikan bentuk input sesuai untuk LSTM ---
        # Jika input berupa [batch, input_size], ubah jadi [batch, seq_len=1, input_size]
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        
        # Jika dataset berbentuk [batch, seq_len, input_size], langsung pakai
        # (tidak perlu diubah)
        
        # --- Forward ---
        output = model(features)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_batches += 1
        
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # --- Threshold sigmoid output menjadi biner (multi-label) ---
        predictions = (output > threshold).float()
        
        # Simpan hasil untuk evaluasi
        all_predictions.append(predictions.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    # --- Gabungkan hasil dari semua batch ---
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # --- Hitung metrik evaluasi ---
    hamming = hamming_loss(all_targets, all_predictions)
    subset_acc = accuracy_score(all_targets, all_predictions)
    micro_f1 = f1_score(all_targets, all_predictions, average='micro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    per_label_acc = (all_predictions == all_targets).mean(axis=0)
    overall_acc = (all_predictions == all_targets).mean()
    avg_loss = total_loss / total_batches

    # --- Kembalikan hasil evaluasi ---
    return {
        "loss": avg_loss,
        "hamming_loss": hamming,
        "subset_accuracy": subset_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "overall_accuracy": overall_acc,
        "per_label_accuracy": per_label_acc.tolist()
    }
#========================================================================================================================

#Loop untuk training dan testing model multi-label classification 1D.
#========================================================================================================================
def loop_multi_label_classification_1D(mode,dataset,dataloader,model,criterion,optimizer,device,threshold=0.6):
    """
    Loop untuk training dan testing model multi-label classification 1D.

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch
        criterion (loss function): fungsi loss
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU
        threshold (float): ambang batas aktivasi sigmoid untuk menentukan label 1 atau 0

    Returns:
        dict: berisi metrik evaluasi model
            - loss: nilai loss rata-rata
            - hamming_loss: rata-rata kesalahan per label
            - subset_accuracy: proporsi prediksi yang semua label-nya benar
            - micro_f1: F1-score global (berbasis seluruh label)
            - macro_f1: rata-rata F1-score antar label
            - overall_accuracy: total proporsi elemen benar
            - per_label_accuracy: akurasi per label (list)
    
    ----------------------------------------------------------------------
    📘 Penjelasan Metrik:
    
    Metrik                 | Arti
    -----------------------|---------------------------------------------
    Hamming Loss           | Rata-rata kesalahan per label. Semakin kecil semakin baik.
    Subset Accuracy        | Persentase sampel di mana *semua* label benar. Sangat ketat.
    Micro F1-score         | F1-score global yang menyeimbangkan presisi & recall seluruh label.
    Macro F1-score         | Rata-rata F1-score per label (tiap label bobot sama).
    Overall Accuracy       | Total prediksi benar dari seluruh label dan sampel.
    Per-label Accuracy     | Akurasi per label, berguna untuk melihat label sulit.
    ----------------------------------------------------------------------
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")
    
    if mode == "train":
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    correct = 0
    total_elements = 0
    all_predictions = []
    all_targets = []

    for features, target in tqdm(dataloader, desc=mode):
        features, target = features.to(device), target.to(device)
        
        # Forward pass
        output = model(features)
        loss = criterion(output, target)
        total_loss += loss.item()

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Apply threshold sigmoid -> binary output
        predictions = (output > threshold).float()

        all_predictions.append(predictions.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

        # Hitung akurasi element-wise
        correct += (predictions == target).sum().item()
        total_elements += target.numel()

    # Gabungkan semua batch
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # === Hitung metrik ===
    hamming = hamming_loss(all_targets, all_predictions)
    subset_acc = accuracy_score(all_targets, all_predictions)
    micro_f1 = f1_score(all_targets, all_predictions, average='micro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    overall_acc = correct / total_elements

    # Akurasi per label
    per_label_acc = (all_predictions == all_targets).mean(axis=0)

    avg_loss = total_loss / len(dataloader)

    # Kembalikan semua metrik
    return {
        "loss": avg_loss,
        "hamming_loss": hamming,
        "subset_accuracy": subset_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "overall_accuracy": overall_acc,
        "per_label_accuracy": per_label_acc
    }
#========================================================================================================================

#Loop untuk training dan testing model multi-label classification dengan 2 input (features_1, features_2).
#========================================================================================================================
def loop_multi_label_classification_2D(mode,dataset,dataloader,model,criterion,optimizer,device,threshold=0.6):
    """
    Loop untuk training dan testing model multi-label classification dengan 2 input (features_1, features_2).

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch dengan dua input
        criterion (loss function): fungsi loss
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU
        threshold (float): ambang batas sigmoid untuk menentukan label aktif (1) atau tidak (0)

    Returns:
        dict: metrik evaluasi model
            - loss
            - hamming_loss
            - subset_accuracy
            - micro_f1
            - macro_f1
            - overall_accuracy
            - per_label_accuracy
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")

    if mode == "train":
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total_elements = 0
    all_predictions = []
    all_targets = []

    for features_1, features_2, target in tqdm(dataloader, desc=mode):
        features_1, features_2, target = (
            features_1.to(device),
            features_2.to(device),
            target.to(device)
        )

        # --- Forward Pass ---
        output = model(features_1, features_2)
        loss = criterion(output, target)
        total_loss += loss.item()

        # --- Backprop jika training ---
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Binarisasi output (sigmoid + threshold) ---
        predictions = (output > threshold).float()

        # --- Simpan hasil ---
        all_predictions.append(predictions.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

        correct += (predictions == target).sum().item()
        total_elements += target.numel()

    # === Gabungkan semua batch ===
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # === Hitung metrik ===
    hamming = hamming_loss(all_targets, all_predictions)
    subset_acc = accuracy_score(all_targets, all_predictions)
    micro_f1 = f1_score(all_targets, all_predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)
    overall_acc = correct / total_elements
    per_label_acc = (all_predictions == all_targets).mean(axis=0)
    avg_loss = total_loss / len(dataloader)

    # === Return hasil ===
    return {
        "loss": avg_loss,
        "hamming_loss": hamming,
        "subset_accuracy": subset_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "overall_accuracy": overall_acc,
        "per_label_accuracy": per_label_acc
    }
#========================================================================================================================

#Loop untuk training dan testing model multi-label classification dengan 3 input (features_1, features_2, features_3).
#========================================================================================================================
def loop_multi_label_classification_3D(mode,dataset,dataloader,model,criterion,optimizer,device,threshold=0.6):
    """
    Loop untuk training dan testing model multi-label classification dengan 3 input (features_1, features_2, features_3).

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch dengan dua input
        criterion (loss function): fungsi loss
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU
        threshold (float): ambang batas sigmoid untuk menentukan label aktif (1) atau tidak (0)

    Returns:
        dict: metrik evaluasi model
            - loss
            - hamming_loss
            - subset_accuracy
            - micro_f1
            - macro_f1
            - overall_accuracy
            - per_label_accuracy
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")

    if mode == "train":
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total_elements = 0
    all_predictions = []
    all_targets = []

    for features_1, features_2, features_3, target in tqdm(dataloader, desc=mode):
        features_1, features_2, features_3, target = (
            features_1.to(device),
            features_2.to(device),
            features_3.to(device),
            target.to(device)
        )

        # --- Forward Pass ---
        output = model(features_1, features_2, features_3)
        loss = criterion(output, target)
        total_loss += loss.item()

        # --- Backprop jika training ---
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Binarisasi output (sigmoid + threshold) ---
        predictions = (output > threshold).float()

        # --- Simpan hasil ---
        all_predictions.append(predictions.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

        correct += (predictions == target).sum().item()
        total_elements += target.numel()

    # === Gabungkan semua batch ===
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # === Hitung metrik ===
    hamming = hamming_loss(all_targets, all_predictions)
    subset_acc = accuracy_score(all_targets, all_predictions)
    micro_f1 = f1_score(all_targets, all_predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)
    overall_acc = correct / total_elements
    per_label_acc = (all_predictions == all_targets).mean(axis=0)
    avg_loss = total_loss / len(dataloader)

    # === Return hasil ===
    return {
        "loss": avg_loss,
        "hamming_loss": hamming,
        "subset_accuracy": subset_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "overall_accuracy": overall_acc,
        "per_label_accuracy": per_label_acc
    }
#========================================================================================================================



#========================================================================================================================
#                                     Classification MULTI CLASS
#========================================================================================================================

#Loop untuk training & testing model multi-class classification berbasis timeseris.(RNN, LSTM, atau GRU).
#========================================================================================================================
def loop_multi_class_classification_timeseris(mode, dataset, dataloader, model, criterion, optimizer, device):
    """
    Loop untuk training & testing model multi-class classification (RNN, LSTM, atau GRU).

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch (RNN/GRU/LSTM)
        criterion (loss function): CrossEntropyLoss
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU

    Returns:
        dict: berisi metrik evaluasi
            - loss: nilai loss rata-rata
            - accuracy: akurasi keseluruhan
            - macro_f1: rata-rata F1 per kelas
            - confusion_matrix: matriks kebingungan (numpy array)
    ----------------------------------------------------------------------
    📘 Penjelasan Metrik:

    Metrik               | Arti
    ----------------------|---------------------------------------------
    Accuracy              | Persentase prediksi yang benar.
    Macro F1-score        | Rata-rata F1-score per kelas (tiap kelas bobot sama).
    Confusion Matrix      | Menunjukkan jumlah prediksi per kelas aktual & prediksi.
    ----------------------------------------------------------------------
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")

    if mode == "train":
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_batches = 0
    all_predictions, all_targets = [], []

    for features, target in tqdm(dataloader, desc=mode):
        features, target = features.to(device), target.to(device)

        # --- Pastikan bentuk input sesuai untuk RNN ---
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # [batch, seq_len=1, input_size]

        # --- Forward pass ---
        output = model(features)  # [batch, num_classes]
        loss = criterion(output, target)
        total_loss += loss.item()
        total_batches += 1

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Prediksi kelas ---
        predictions = torch.argmax(output, dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())

    # --- Hitung metrik ---
    avg_loss = total_loss / total_batches
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_predictions)

    # --- Print hasil evaluasi ---
    print("\n=== Hasil Evaluasi ===")
    print(f"Loss rata-rata       : {avg_loss:.4f}")
    print(f"Akurasi              : {accuracy*100:.2f}%")
    print(f"Macro F1-score       : {macro_f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # --- Return hasil ---
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist()
    }
#========================================================================================================================

#Loop untuk training dan testing model multi-class classification 1D.
#========================================================================================================================
def loop_multi_class_classification_1D(mode,dataset,dataloader,model,criterion,optimizer,device):
    """
    Loop untuk training dan testing model multi-class classification 1D.

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch (MLP, CNN, dsb)
        criterion (loss function): biasanya nn.CrossEntropyLoss()
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU

    Returns:
        dict: berisi metrik evaluasi
            - loss: nilai loss rata-rata
            - accuracy: akurasi keseluruhan
            - macro_f1: rata-rata F1 per kelas
            - confusion_matrix: matriks kebingungan (numpy array)
    
    ----------------------------------------------------------------------
    📘 Penjelasan Metrik:

    Metrik                 | Arti
    -----------------------|---------------------------------------------
    Accuracy               | Persentase prediksi yang benar.
    Macro F1-score         | Rata-rata F1-score per kelas (tiap kelas bobot sama).
    Confusion Matrix       | Menunjukkan jumlah prediksi benar & salah per kelas.
    ----------------------------------------------------------------------
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")

    if mode == "train":
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_batches = 0
    all_predictions, all_targets = [], []

    for features, target in tqdm(dataloader, desc=mode):
        features, target = features.to(device), target.to(device)

        # --- Forward pass ---
        output = model(features)  # [batch_size, num_classes]
        loss = criterion(output, target)
        total_loss += loss.item()
        total_batches += 1

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Ambil prediksi kelas tertinggi ---
        predictions = torch.argmax(output, dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())

    # --- Hitung metrik ---
    avg_loss = total_loss / total_batches
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_predictions)

    # --- Return hasil ---
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist()
    }
#========================================================================================================================

#Loop untuk training dan testing model multi-class classification 2D.
#========================================================================================================================
def loop_multi_class_classification_2D(mode,dataset,dataloader,model,criterion,optimizer,device):
    """
    Loop untuk training dan testing model multi-class classification 2D memilki 2 input(features_1, features_2).

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch (MLP, CNN, dsb)
        criterion (loss function): biasanya nn.CrossEntropyLoss()
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU

    Returns:
        dict: berisi metrik evaluasi
            - loss: nilai loss rata-rata
            - accuracy: akurasi keseluruhan
            - macro_f1: rata-rata F1 per kelas
            - confusion_matrix: matriks kebingungan (numpy array)
    
    ----------------------------------------------------------------------
    📘 Penjelasan Metrik:

    Metrik                 | Arti
    -----------------------|---------------------------------------------
    Accuracy               | Persentase prediksi yang benar.
    Macro F1-score         | Rata-rata F1-score per kelas (tiap kelas bobot sama).
    Confusion Matrix       | Menunjukkan jumlah prediksi benar & salah per kelas.
    ----------------------------------------------------------------------
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")

    if mode == "train":
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_batches = 0
    all_predictions, all_targets = [], []

    for features_1, features_2, target in tqdm(dataloader, desc=mode):
        features_1, features_2, target = features_1.to(device), features_2.to(device), target.to(device)

        # --- Forward pass ---
        output = model(features_1, features_2)  # [batch_size, num_classes]
        loss = criterion(output, target)
        total_loss += loss.item()
        total_batches += 1

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Ambil prediksi kelas tertinggi ---
        predictions = torch.argmax(output, dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())

    # --- Hitung metrik ---
    avg_loss = total_loss / total_batches
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_predictions)

    # --- Return hasil ---
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist()
    }
#========================================================================================================================

#Loop untuk training dan testing model multi-class classification 3D.
#========================================================================================================================
def loop_multi_class_classification_3D(mode,dataset,dataloader,model,criterion,optimizer,device):
    """
    Loop untuk training dan testing model multi-class classification 3D memilki 3 input(features_1, features_2, features_3).

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch (MLP, CNN, dsb)
        criterion (loss function): biasanya nn.CrossEntropyLoss()
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU

    Returns:
        dict: berisi metrik evaluasi
            - loss: nilai loss rata-rata
            - accuracy: akurasi keseluruhan
            - macro_f1: rata-rata F1 per kelas
            - confusion_matrix: matriks kebingungan (numpy array)
    
    ----------------------------------------------------------------------
    📘 Penjelasan Metrik:

    Metrik                 | Arti
    -----------------------|---------------------------------------------
    Accuracy               | Persentase prediksi yang benar.
    Macro F1-score         | Rata-rata F1-score per kelas (tiap kelas bobot sama).
    Confusion Matrix       | Menunjukkan jumlah prediksi benar & salah per kelas.
    ----------------------------------------------------------------------
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")

    if mode == "train":
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_batches = 0
    all_predictions, all_targets = [], []

    for features_1, features_2, features_3, target in tqdm(dataloader, desc=mode):
        features_1, features_2, features_3, target = features_1.to(device), features_2.to(device), features_3.to(device), target.to(device)

        # --- Forward pass ---
        output = model(features_1, features_2, features_3)  # [batch_size, num_classes]
        loss = criterion(output, target)
        total_loss += loss.item()
        total_batches += 1

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Ambil prediksi kelas tertinggi ---
        predictions = torch.argmax(output, dim=1)

        all_predictions.extend(predictions.detach().cpu().numpy())
        all_targets.extend(target.detach().cpu().numpy())

    # --- Hitung metrik ---
    avg_loss = total_loss / total_batches
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_predictions)

    # --- Return hasil ---
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist()
    }
#========================================================================================================================



#========================================================================================================================
#                                                Regression
#========================================================================================================================

#Loop untuk training dan testing model regresi berbasis time series.
#========================================================================================================================
def loop_regression_timeseries(mode,dataset,dataloader,model,criterion,optimizer,device):
    """
    Loop untuk training dan testing model regresi berbasis time series.

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch (LSTM/GRU/MLP/CNN)
        criterion (loss function): fungsi loss, misal nn.MSELoss()
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU

    Returns:
        dict: berisi metrik evaluasi model regresi
            - loss: nilai loss rata-rata
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - r2: R-squared (koefisien determinasi)
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")

    if mode == "train":
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_batches = 0
    all_predictions, all_targets = [], []

    for features, target in tqdm(dataloader, desc=mode):
        features, target = features.to(device), target.to(device)

        # Pastikan target berbentuk float untuk regresi
        target = target.float()

        # --- Forward pass ---
        output = model(features)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_batches += 1

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Simpan hasil prediksi dan target
        all_predictions.append(output.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    # --- Gabungkan semua batch ---
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # --- Hitung metrik evaluasi ---
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    avg_loss = total_loss / total_batches

    return {
        "loss": avg_loss,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
#========================================================================================================================

#Loop untuk training dan testing model regresi 1D.
#========================================================================================================================
def loop_regression_1D(mode,dataset,dataloader,model,criterion,optimizer,device):
    """
    Loop untuk training dan testing model regresi 1D.

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch (MLP, CNN1D, dll)
        criterion (loss function): fungsi loss, misal nn.MSELoss()
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU

    Returns:
        dict: berisi metrik evaluasi model regresi
            - loss: nilai loss rata-rata
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - r2: R-squared (koefisien determinasi)
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")

    model.train() if mode == "train" else model.eval()

    total_loss = 0.0
    total_batches = 0
    all_predictions = []
    all_targets = []

    for features, target in tqdm(dataloader, desc=mode):
        # Kirim ke device
        features, target = features.to(device), target.to(device)

        # Pastikan target bertipe float
        target = target.float()

        # === Forward Pass ===
        output = model(features)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_batches += 1

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Simpan hasil
        all_predictions.append(output.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    # Gabungkan semua hasil batch
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # === Hitung metrik regresi ===
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    avg_loss = total_loss / total_batches

    return {
        "loss": avg_loss,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
#========================================================================================================================

#Loop untuk training dan testing model regresi 2D.
#========================================================================================================================
def loop_regression_2D(mode,dataset,dataloader,model,criterion,optimizer,device):
    """
    Loop untuk training dan testing model regresi 2D. dengan 2 input (features_1, features_2)

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch (MLP, CNN1D, dll)
        criterion (loss function): fungsi loss, misal nn.MSELoss()
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU

    Returns:
        dict: berisi metrik evaluasi model regresi
            - loss: nilai loss rata-rata
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - r2: R-squared (koefisien determinasi)
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")

    model.train() if mode == "train" else model.eval()

    total_loss = 0.0
    total_batches = 0
    all_predictions = []
    all_targets = []

    for features_1, features_2, target in tqdm(dataloader, desc=mode):
        # Kirim ke device
        features_1, features_2, target = features_1.to(device), features_2.to(device), target.to(device)

        # Pastikan target bertipe float
        target = target.float()

        # === Forward Pass ===
        output = model(features_1, features_2)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_batches += 1

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Simpan hasil
        all_predictions.append(output.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    # Gabungkan semua hasil batch
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # === Hitung metrik regresi ===
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    avg_loss = total_loss / total_batches

    return {
        "loss": avg_loss,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
#========================================================================================================================

#Loop untuk training dan testing model regresi 3D.
#========================================================================================================================
def loop_regression_3D(mode,dataset,dataloader,model,criterion,optimizer,device):
    """
    Loop untuk training dan testing model regresi 3D. dengan 3 input (features_1, features_2, features_3)

    Args:
        mode (str): 'train' atau 'test'
        dataset (Dataset): dataset PyTorch
        dataloader (DataLoader): loader data
        model (nn.Module): model PyTorch (MLP, CNN1D, dll)
        criterion (loss function): fungsi loss, misal nn.MSELoss()
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): CPU atau GPU

    Returns:
        dict: berisi metrik evaluasi model regresi
            - loss: nilai loss rata-rata
            - mae: Mean Absolute Error
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - r2: R-squared (koefisien determinasi)
    """
    if mode not in ["train", "test"]:
        raise ValueError("Mode harus 'train' atau 'test'")

    model.train() if mode == "train" else model.eval()

    total_loss = 0.0
    total_batches = 0
    all_predictions = []
    all_targets = []

    for features_1, features_2, features_3, target in tqdm(dataloader, desc=mode):
        # Kirim ke device
        features_1, features_2, features_3, target = features_1.to(device), features_2.to(device), features_3.to(device), target.to(device)

        # Pastikan target bertipe float
        target = target.float()

        # === Forward Pass ===
        output = model(features_1, features_2, features_3)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_batches += 1

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Simpan hasil
        all_predictions.append(output.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    # Gabungkan semua hasil batch
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # === Hitung metrik regresi ===
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    avg_loss = total_loss / total_batches

    return {
        "loss": avg_loss,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
#========================================================================================================================