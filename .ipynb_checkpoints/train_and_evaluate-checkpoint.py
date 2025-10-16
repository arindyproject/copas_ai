import torch
from rich.console import Console
from rich.table import Table
from rich.columns import Columns
import looper as lp

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

console = Console()

def train_and_evaluate_multi_label(model, train_dataset, test_dataset, train_loader, test_loader, criterion, optim, config, device, labels, callback, dimension='1d',show_detail=True):
    """
    Fungsi untuk melatih dan menguji model multi-label classification dengan tampilan tabel Rich yang rapi.

    Args:
        model (torch.nn.Module): Model PyTorch.
        train_dataset, test_dataset: Dataset PyTorch.
        train_loader, test_loader: DataLoader PyTorch.
        criterion: Loss function.
        optim: Optimizer.
        config: Objek konfigurasi dengan atribut 'threshold'.
        device: torch.device (CPU/GPU).
        labels (list[str]): Daftar nama label untuk per-label accuracy.
        callback: Objek callback (logging, plotting, dan early stopping).
    """

    epoch_counter = 0
    while True:
        epoch_counter += 1
        console.rule(f"[bold green]🚀 EPOCH {epoch_counter}")
        
        if(dimension=='1d'):
            # === Training ===
            train_metrics = lp.loop_multi_label_classification_1D(
                "train", train_dataset, train_loader, model, criterion, optim, device, config.threshold
            )

            # === Testing ===
            with torch.no_grad():
                test_metrics = lp.loop_multi_label_classification_1D(
                    "test", test_dataset, test_loader, model, criterion, optim, device, config.threshold
                )
        elif(dimension=='2d'):
            # === Training ===
            train_metrics = lp.loop_multi_label_classification_2D(
                "train", train_dataset, train_loader, model, criterion, optim, device, config.threshold
            )

            # === Testing ===
            with torch.no_grad():
                test_metrics = lp.loop_multi_label_classification_2D(
                    "test", test_dataset, test_loader, model, criterion, optim, device, config.threshold
                )
        elif(dimension=='3d'):
            # === Training ===
            train_metrics = lp.loop_multi_label_classification_3D(
                "train", train_dataset, train_loader, model, criterion, optim, device, config.threshold
            )

            # === Testing ===
            with torch.no_grad():
                test_metrics = lp.loop_multi_label_classification_3D(
                    "test", test_dataset, test_loader, model, criterion, optim, device, config.threshold
                )
        
        if(show_detail):
            # === Tabel Utama (Epoch Summary) ===
            table_summary = Table(title=f"📊 Summary", title_style="bold magenta")
            table_summary.add_column("Metric", justify="left", style="cyan", no_wrap=True)
            table_summary.add_column("Train", justify="right", style="green")
            table_summary.add_column("Test", justify="right", style="yellow")

            metrics_keys = ["loss", "hamming_loss", "subset_accuracy", "micro_f1", "macro_f1", "overall_accuracy"]

            for key in metrics_keys:
                table_summary.add_row(
                    key.replace("_", " ").title(),
                    f"{train_metrics[key]:.4f}",
                    f"{test_metrics[key]:.4f}"
                )

            # === Tabel Per-Label Accuracy ===
            table_label = Table(title="🎯 Per-Label Accuracy", title_style="bold blue")
            table_label.add_column("Label", justify="center", style="cyan", no_wrap=True)
            table_label.add_column("Accuracy", justify="center", style="green")

            per_label_acc = test_metrics.get("per_label_accuracy", [])
            if len(per_label_acc) == len(labels):
                for label, acc in zip(labels, per_label_acc):
                    table_label.add_row(label, f"{acc:.4f}")
            else:
                for i, acc in enumerate(per_label_acc):
                    table_label.add_row(f"Label {i+1}", f"{acc:.4f}")

            # === Cetak berdampingan ===
            console.print(Columns([table_summary, table_label]))

        # === Logging & Callback ===
        train_cost, train_acc = train_metrics["loss"], train_metrics["overall_accuracy"]
        test_cost, test_acc = test_metrics["loss"], test_metrics["overall_accuracy"]

        callback.log(train_cost, test_cost, train_acc, test_acc)
        callback.save_checkpoint()
        callback.cost_runtime_plotting()
        callback.score_runtime_plotting()

        # === Early Stopping ===
        if callback.early_stopping(model, monitor="test_score"):
            callback.plot_cost()
            callback.plot_score()
            console.print("[bold red]⏹️ Training dihentikan oleh early stopping.")
            break
            

def train_and_evaluate_multi_class(
    model,
    train_dataset, test_dataset,
    train_loader, test_loader,
    criterion, optim,
    config, device, labels,
    callback,
    dimension='1d',
    show_detail=True
):
    """
    Fungsi untuk melatih & menguji model multi-class classification
    dengan tampilan tabel Rich (mirip versi multi-label).
    """

    epoch_counter = 0
    while True:
        epoch_counter += 1
        console.rule(f"[bold green]🚀 EPOCH {epoch_counter}")

        # === Loop Training & Testing ===
        if dimension == '1d':
            # Training
            train_metrics = lp.loop_multi_class_classification_1D(
                "train", train_dataset, train_loader, model, criterion, optim, device
            )

            # Testing
            with torch.no_grad():
                test_metrics = lp.loop_multi_class_classification_1D(
                    "test", test_dataset, test_loader, model, criterion, optim, device
                )

        elif dimension == '2d':
            train_metrics = lp.loop_multi_class_classification_2D(
                "train", train_dataset, train_loader, model, criterion, optim, device
            )
            with torch.no_grad():
                test_metrics = lp.loop_multi_class_classification_2D(
                    "test", test_dataset, test_loader, model, criterion, optim, device
                )

        elif dimension == '3d':
            train_metrics = lp.loop_multi_class_classification_3D(
                "train", train_dataset, train_loader, model, criterion, optim, device
            )
            with torch.no_grad():
                test_metrics = lp.loop_multi_class_classification_3D(
                    "test", test_dataset, test_loader, model, criterion, optim, device
                )

        # === Menampilkan hasil detail ===
        if show_detail:
            # === Summary Table ===
            table_summary = Table(title="Summary", title_style="bold magenta")
            table_summary.add_column("Metric", justify="left", style="cyan", no_wrap=True)
            table_summary.add_column("Train", justify="right", style="green")
            table_summary.add_column("Test", justify="right", style="yellow")

            metrics_keys = ["loss", "accuracy", "macro_f1"]
            for key in metrics_keys:
                table_summary.add_row(
                    key.replace("_", " ").title(),
                    f"{train_metrics[key]:.4f}",
                    f"{test_metrics[key]:.4f}"
                )

            # === Per-Class Accuracy ===
            cm = np.array(test_metrics["confusion_matrix"])
            per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

            table_label = Table(title="Per-Class Accuracy", title_style="bold blue")
            table_label.add_column("Class", justify="center", style="cyan")
            table_label.add_column("Accuracy", justify="center", style="green")

            for label_id, acc in zip(labels.keys(), per_class_acc):
                table_label.add_row(str(labels[label_id]), f"{acc:.4f}")

            # === Print berdampingan ===
            console.print(Columns([table_summary, table_label]))

   

            # === Confusion Matrix (visualisasi Matplotlib) ===
            plt.figure(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=[labels[k] for k in labels.keys()],
                        yticklabels=[labels[k] for k in labels.keys()])
            plt.xlabel("Predicted", fontsize=12)
            plt.ylabel("Actual", fontsize=12)
            plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show()


        # === Logging & Callback ===
        train_cost, train_acc = train_metrics["loss"], train_metrics["accuracy"]
        test_cost, test_acc = test_metrics["loss"], test_metrics["accuracy"]

        callback.log(train_cost, test_cost, train_acc, test_acc)
        callback.save_checkpoint()
        callback.cost_runtime_plotting()
        callback.score_runtime_plotting()

        # === Early Stopping ===
        if callback.early_stopping(model, monitor="test_score"):
            callback.plot_cost()
            callback.plot_score()
            console.print("[bold red]⏹️ Training dihentikan oleh early stopping.")
            break