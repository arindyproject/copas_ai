import torch
import pandas as pd
import numpy as np
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

import matplotlib.pyplot as plt
import seaborn as sns

console = Console()

def evaluate_multi_label(model, dataloader, target_cols, criterion, device, threshold=0.5, dimension='1d'):
    """
    Evaluasi model multi-label classification.

    Args:
        model (nn.Module): Model PyTorch
        dataloader (DataLoader): DataLoader untuk data test
        target_cols (list): Nama kolom target
        device (torch.device)
        threshold (float)
        criterion (loss function, optional): Jika diberikan, akan hitung rata-rata loss
    
    Returns:
        dict: hasil metrik dan DataFrame hasil prediksi
    """
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    total_correct = 0
    total_elements = 0

    with torch.no_grad():
        if(dimension=='1d'):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = (outputs > threshold).float()

                # Hitung loss dan akurasi elemen
                loss = criterion(outputs, y)
                total_loss += loss.item()
                total_correct += (preds == y).sum().item()
                total_elements += y.numel()

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
        elif(dimension=='2d'):
            for x1,x2, y in dataloader:
                x1,x2, y = x1.to(device),x2.to(device), y.to(device)
                outputs = model(x1,x2)
                preds = (outputs > threshold).float()

                # Hitung loss dan akurasi elemen
                loss = criterion(outputs, y)
                total_loss += loss.item()
                total_correct += (preds == y).sum().item()
                total_elements += y.numel()

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
        elif(dimension=='3d'):
            for x1,x2, x3, y in dataloader:
                x1,x2, x3, y = x1.to(device),x2.to(device),x3.to(device), y.to(device)
                outputs = model(x1,x2,x3)
                preds = (outputs > threshold).float()

                # Hitung loss dan akurasi elemen
                loss = criterion(outputs, y)
                total_loss += loss.item()
                total_correct += (preds == y).sum().item()
                total_elements += y.numel()

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # === Hitung metrik ===
    avg_loss = total_loss / len(dataloader)
    overall_acc = total_correct / total_elements
    h_loss = hamming_loss(targets, preds)
    subset_acc = accuracy_score(targets, preds)
    micro_f1 = f1_score(targets, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    per_label_acc = ((preds == targets).sum(axis=0) / len(targets))
    
    # === Tabel utama ===
    table_summary = Table(title="📊 Evaluasi Model", title_style="bold magenta")
    table_summary.add_column("Metrik", style="cyan", justify="left")
    table_summary.add_column("Nilai", style="green", justify="right")

    
    table_summary.add_row("🔹 Overall Accuracy", f"{overall_acc:.4f}")
    table_summary.add_row("✅ Subset  Accuracy", f"{subset_acc:.4f}")
    table_summary.add_row("💥 Loss (avg)", f"{avg_loss:.4f}")
    table_summary.add_row("📉 Hamming Loss", f"{h_loss:.4f}")
    table_summary.add_row("⚖️  Micro F1-score", f"{micro_f1:.4f}")
    table_summary.add_row("📈 Macro F1-score", f"{macro_f1:.4f}")

    # === Tabel per label ===
    table_label = Table(title="🎯 Per-Label Accuracy", title_style="bold blue")
    table_label.add_column("Label", justify="center", style="cyan")
    table_label.add_column("Akurasi", justify="center", style="green")

    for label, acc in zip(target_cols, per_label_acc):
        table_label.add_row(label, f"{acc:.4f}")

    # === Gabungkan tampilan (kiri-kanan) ===
    console.print(Columns([table_summary, table_label]))

    # === Hasil dataframe untuk analisis lanjut ===
    df_results = pd.DataFrame({
        "Target": targets.tolist(),
        "Prediksi": preds.tolist()
    })
    
    df_results["Benar (%)"] = [
        (sum(t == p for t, p in zip(tar, pre)) / len(tar)) * 100
        for tar, pre in zip(df_results["Target"], df_results["Prediksi"])
    ]

    return {
        "df": df_results,
        "preds": preds,
        "targets": targets,
        "loss": avg_loss,
        "overall_accuracy": overall_acc,
        "hamming_loss": h_loss,
        "subset_accuracy": subset_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_label_accuracy": dict(zip(target_cols, per_label_acc.round(4)))
    }


def evaluate_multi_class(model, dataloader, labels, criterion, device, dimension='1d', show_detail=True):
    """
    Evaluasi model multi-class classification dengan tampilan tabel & hasil prediksi.

    Args:
        model (nn.Module)
        dataloader (DataLoader)
        labels (dict | list): Mapping id→label atau list label.
        criterion: Fungsi loss (mis. nn.CrossEntropyLoss)
        device (torch.device)
        dimension (str): '1d', '2d', atau '3d'
        show_detail (bool): Jika True, tampilkan tabel dan confusion matrix.

    Returns:
        dict: hasil evaluasi dan DataFrame hasil prediksi
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            # === Input Dinamis ===
            if dimension == '1d':
                x, y = batch
                outputs = model(x.to(device))
            elif dimension == '2d':
                x1, x2, y = batch
                outputs = model(x1.to(device), x2.to(device))
            elif dimension == '3d':
                x1, x2, x3, y = batch
                outputs = model(x1.to(device), x2.to(device), x3.to(device))
            else:
                raise ValueError("dimension harus '1d', '2d', atau '3d'")

            y = y.to(device)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            total_batches += 1

            preds = torch.argmax(outputs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    preds = np.array(all_predictions)
    targets = np.array(all_targets)

    # === Hitung metrik ===
    avg_loss = total_loss / total_batches
    accuracy = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    cm = confusion_matrix(targets, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    # === Konversi labels ===
    if isinstance(labels, dict):
        label_names = [labels[k] for k in sorted(labels.keys())]
    elif isinstance(labels, (list, tuple)):
        label_names = list(labels)
    else:
        raise ValueError("`labels` harus dict atau list")

    # === Tabel utama ===
    table_summary = Table(title="📊 Evaluasi Model", title_style="bold magenta")
    table_summary.add_column("Metrik", style="cyan", justify="left")
    table_summary.add_column("Nilai", style="green", justify="right")

    table_summary.add_row("💥 Loss (avg)", f"{avg_loss:.4f}")
    table_summary.add_row("🔹 Accuracy", f"{accuracy:.4f}")
    table_summary.add_row("📈 Macro F1-score", f"{macro_f1:.4f}")

    # === Tabel per label ===
    table_label = Table(title="🎯 Per-Class Accuracy", title_style="bold blue")
    table_label.add_column("Class", justify="center", style="cyan")
    table_label.add_column("Accuracy", justify="center", style="green")

    for label_name, acc in zip(label_names, per_class_acc):
        table_label.add_row(str(label_name), f"{acc:.4f}")

    # === Print berdampingan ===
    if show_detail:
        console.print(Columns([table_summary, table_label]))

        # === Confusion Matrix (matplotlib) ===
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names,
                    yticklabels=label_names)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    # === DataFrame hasil ===
    df_results = pd.DataFrame({
        "Target": [label_names[t] for t in targets],
        "Prediksi": [label_names[p] for p in preds]
    })
    df_results["Benar"] = (df_results["Target"] == df_results["Prediksi"]).astype(int)
    df_results["Benar (%)"] = df_results["Benar"] * 100

    return {
        "df": df_results,
        "preds": preds,
        "targets": targets,
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_accuracy": dict(zip(label_names, per_class_acc.round(4)))
    }