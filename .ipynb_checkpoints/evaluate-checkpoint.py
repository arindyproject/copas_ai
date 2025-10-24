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

import torch.nn.functional as F

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
    import torch
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from rich.table import Table
    from rich.columns import Columns
    from rich.console import Console

    console = Console()

    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_predictions, all_targets = [], []
    all_probs = []

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

            probs = F.softmax(outputs, dim=1)  # tensor
            preds = torch.argmax(probs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    preds = np.array(all_predictions)
    targets = np.array(all_targets)
    probs = np.array(all_probs)

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

    if show_detail:
        console.print(Columns([table_summary, table_label]))

        # === Confusion Matrix ===
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
        "Prediksi": [label_names[p] for p in preds],
        "Prob (%)": (probs[np.arange(len(preds)), preds] * 100).round(2),  # Prob kelas prediksi
        "Benar (%)": (probs[np.arange(len(targets)), targets] * 100).round(2),  # Prob kelas target sebenarnya
    })
    df_results["Benar"] = (df_results["Target"] == df_results["Prediksi"]).astype(int)

    return {
        "df": df_results,
        "preds": preds,
        "targets": targets,
        "probs": probs,
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_accuracy": dict(zip(label_names, per_class_acc.round(4)))
    }


def evaluate_regression(model, dataloader, criterion, device, dimension='1d', show_detail=True):
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from rich.table import Table
    from rich.columns import Columns
    from rich.console import Console

    console = Console()
    model.eval()

    total_loss = 0.0
    total_batches = 0
    all_preds, all_targets = [], []

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

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # === Hitung metrik regresi ===
    avg_loss = total_loss / total_batches
    mae = mean_absolute_error(targets, preds)
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, preds)

    # === Tabel ringkasan ===
    table_summary = Table(title="📊 Evaluasi Model Regresi", title_style="bold magenta")
    table_summary.add_column("Metrik", style="cyan", justify="left")
    table_summary.add_column("Nilai", style="green", justify="right")
    table_summary.add_row("💥 Loss (avg)", f"{avg_loss:.4f}")
    table_summary.add_row("📉 MAE", f"{mae:.4f}")
    table_summary.add_row("📉 MSE", f"{mse:.4f}")
    table_summary.add_row("📉 RMSE", f"{rmse:.4f}")
    table_summary.add_row("📈 R² Score", f"{r2:.4f}")

    if show_detail:
        console.print(table_summary)

        # === Scatter Plot (Pred vs Target) ===
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=targets, y=preds, s=30, alpha=0.7)
        plt.plot([targets.min(), targets.max()],
                 [targets.min(), targets.max()],
                 'r--', lw=2, label="Ideal (y=x)")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Prediksi vs Aktual", fontsize=13, fontweight="bold")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # === Distribusi Error ===
        errors = preds - targets
        plt.figure(figsize=(6, 4))
        sns.histplot(errors, bins=30, kde=True, color='purple')
        plt.xlabel("Error (Pred - Actual)")
        plt.title("Distribusi Error", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    # === DataFrame hasil ===
    df_results = pd.DataFrame({
        "Target": targets.round(4),
        "Prediksi": preds.round(4),
        "Error": (preds - targets).round(4),
        "Error (%)": ((preds - targets) / (targets + 1e-8) * 100).round(2)
    })

    return {
        "df": df_results,
        "preds": preds,
        "targets": targets,
        "loss": avg_loss,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

