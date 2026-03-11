# ---------------------
# IMPORTS
# ---------------------
import os
import sys
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, hamming_loss, f1_score, brier_score_loss, confusion_matrix
import copy


# from networks import MILNet
# from datasets import DatasetECG
# from utils import (
#     setup_logger, compute_best_thresholds, plot_loss_curves,
#     plot_roc_auc_per_class, create_run_folder, get_pos_weights,
#     distributions, get_class_data, plot_avg_std_loss_curve, combine_CVD_classes,
#     model_log, split_data_patient_aware, plot_attn_heatmap
# )


# load data:
ECGS = "drive/MyDrive/CMBE/ecg_CMBE_dataset.npz"
data = np.load(ECGS, allow_pickle=True)


# ---------------------
# CONFIG
# ---------------------
RUNS_BASE = {
    "VA" : "/content/drive/MyDrive/CMBE/RUNS/RUNS_VA",
    "AF" : "/content/drive/MyDrive/CMBE/RUNS/RUNS_AF",
    "IS" : "/content/drive/MyDrive/CMBE/RUNS/RUNS_IS",
    "HF" : "/content/drive/MyDrive/CMBE/RUNS/RUNS_HF",
    "COMBINED" : "/content/drive/MyDrive/CMBE/RUNS/RUNS_COMBINED"
}

CLASS_NAMES = ["VA", "AF", "IS", "HF"]
class_name = "AF"
COMBINED_CVDS = False
ONE_CVD = True
USE_SAMPLER = True
TRAIN_ENCODER = True

EPOCHS = 10
LR = 1e-4
K_FOLDS = None
BATCH_SIZE = 32         #try 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.BCEWithLogitsLoss()

using_val_test = False
weighted = True

# ---------------------
# DATA LOADING
# ---------------------
# Keep full original dataset untouched
features_full = data["features"].copy()
subject_ids_full = data["patient_ids"].copy()
study_ids_full = data["study_ids"].copy()
labels_full = data["labels"].copy()

# ---------------------
# FILTERING
# ---------------------
if ONE_CVD:
    features, labels, mask = get_class_data(class_name, CLASS_NAMES, features_full, labels_full)
    pids = subject_ids_full[mask].copy()
    study_ids = study_ids_full[mask].copy()
elif COMBINED_CVDS:
    features = features_full
    labels = combine_CVD_classes(labels_full)
    pids = subject_ids_full
    study_ids = study_ids_full
else:
    features = features_full
    labels = labels_full
    pids = subject_ids_full
    study_ids = study_ids_full


# ---------------------
# ENSURE LABELS ARE FLOAT32
# ---------------------
labels = np.asarray(labels)
if labels.dtype == object:
    labels = np.array(labels.tolist(), dtype=np.float32)
else:
    labels = labels.astype(np.float32)

print("NaNs in features:", np.isnan(features).any())
print("Infs in features:", np.isinf(features).any())
print(f"Dataset patients: {features.shape[0]}")
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)
print("PIDS shape:", len(pids))

# ---------------------
# PATIENT-AWARE SPLIT
# ---------------------
split_data = split_data_patient_aware(
    features=features,
    labels=labels,
    pids=pids,
    study_ids=study_ids,
    batch_size=BATCH_SIZE,
    use_sampler=USE_SAMPLER
)

train_loader = split_data["train_loader"]
val_loader   = split_data["val_loader"]

X_train = split_data["X_train"]
y_train = split_data["y_train"]

pids_train = split_data["pids_train"]
study_ids_train = split_data["study_ids_train"]


# ---------------------
# TRAINING FUNCTION
# ---------------------
def train_model(model, train_loader, val_loader, device, criterion, epochs, lr, run_folder):
    model.to(device)

    # optimizer = torch.optim.Adam([
    # {'params': model.encoder.parameters(), 'lr': lr },
    # {'params': model.decoder.parameters(), 'lr': lr},
    # ])

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    best_val_loss = float('inf')
    best_val_auc = 0.0
    best_model_path = os.path.join(run_folder, "best_model.pth")

    train_losses, val_losses = [], []
    val_aucs = []   # STORE AUC PER EPOCH

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = []

        # ---------------------
        # TRAINING LOOP (FIXED UNPACKING)
        # ---------------------
        for batch_idx, (X_batch, y_batch, pids_batch, study_batch) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")):

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)

            loss = criterion(logits.squeeze(-1), y_batch.squeeze(-1))
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())

        avg_train_loss = np.mean(epoch_train_loss)

        # ----------------- VALIDATION LOSS + AUC -----------------
        model.eval()
        epoch_val_loss = []
        val_probs_list, val_targets_list = [], []

        for X_batch, y_batch, pids_batch, study_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            with torch.no_grad():
                logits = model(X_batch)
                loss = criterion(logits.squeeze(-1), y_batch.squeeze(-1))

                epoch_val_loss.append(loss.item())
                val_probs_list.append(torch.sigmoid(logits).cpu().numpy())
                val_targets_list.append(y_batch.cpu().numpy())

        avg_val_loss = np.mean(epoch_val_loss)
        val_probs = np.concatenate(val_probs_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)
        val_auc = roc_auc_score(val_targets, val_probs)
        val_aucs.append(val_auc)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f}")

        # Save best model based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved Best Model at Epoch {epoch+1} | Val AUC: {val_auc:.4f}")

    return train_losses, val_losses, val_aucs

# ---------------------
# EVALUATION FUNCTION
# ---------------------
def evaluate_model(model, data_loader, device, run_folder, split_name="val", thresholds=None, one_CVD=True, combined_CVDs=False):
    model.eval()
    probs_list, targets_list = [], []

    with torch.no_grad():
        for batch_idx, (X_batch, y_batch, pids_batch, study_batch) in enumerate(data_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)

            probs_list.append(probs.cpu().numpy())
            targets_list.append(y_batch.cpu().numpy())

    # --------------------- METRICS ---------------------
    probs = np.concatenate(probs_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    preds = (probs >= thresholds).astype(int) if thresholds is not None else (probs > 0.5).astype(int)

    if thresholds is not None:
        np.save(os.path.join(run_folder, f"{split_name}_best_thresholds.npy"), thresholds)
        print("Using optimized thresholds:", thresholds)
    else:
        print("Using default threshold: 0.5")

    auc = roc_auc_score(targets, probs)
    f1 = f1_score(targets, preds, zero_division=0)
    acc = accuracy_score(targets, preds)
    hamm = hamming_loss(targets, preds)
    brier = float(np.mean([brier_score_loss(targets[:, i], probs[:, i]) for i in range(targets.shape[1])]))

    confusion_matrices = []
    sensitivities = []
    specificities = []

    for i in range(targets.shape[1]):
        cm = confusion_matrix(targets[:, i], preds[:, i])
        confusion_matrices.append(cm)
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            sensitivities.append(tp / (tp + fn + 1e-8))
            specificities.append(tn / (tn + fp + 1e-8))
        else:
            sensitivities.append(np.nan)
            specificities.append(np.nan)

    metrics_dict = {
        "accuracy": acc, "roc_auc": auc, "hamming": hamm, "brier": brier,
        "f1": f1, "sensitivities": sensitivities,"specificities": specificities, "confusion_matrices": confusion_matrices
    }

    print("\n======= FINAL TEST METRICS =======")
    print(f"{split_name}_ACCURACY : {acc:.4f}")
    print(f"{split_name}_ROC AUC  : {auc:.4f}")
    print(f"{split_name}_F1 SCORE : {f1:.4f}")
    print(f"{split_name}_HAMMING  : {hamm:.4f}")
    print(f"{split_name}_BRIER    : {brier:.4f}")
    print("==================================================")

    np.save(os.path.join(run_folder, f"{split_name}_test_probs.npy"), probs)
    np.save(os.path.join(run_folder, f"{split_name}_test_preds.npy"), preds)
    np.save(os.path.join(run_folder, f"{split_name}_test_targets.npy"), targets)

    plot_roc_auc_per_class(targets, probs, CLASS_NAMES, run_folder)

    return {"preds": preds, "probs": probs, "targets": targets, "metrics": metrics_dict}

# ---------------------
# MAIN FUNCTION
# ---------------------
def main(model, X, y, pids, epochs, lr, batch_size, device, criterion, weighted, using_val_test, k_folds, run_folder, one_CVD, combined_CVDS, use_sampler):
    print("PRINTING WORKING")
    os.makedirs(run_folder, exist_ok=True)

    out_path = run_folder
    model_file = os.path.join(out_path, "model.txt")
    with open(model_file, "w") as f:
        f.write(f"MODEL:\n{str(model)}\n\n")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"TOTAL TRAINABLE PARAMETERS:\t{total_params}\n\n")
        f.write(f"OPTIMIZER:\nAdam (lr={lr})\n\n")
        f.write(f"CRITERION:\n{str(criterion)}\n\n")
        f.write(f"BATCH SIZE:\t{batch_size}\n")
        f.write(f"EPOCHS:\t{epochs}\n\n")
        f.write(f"DEVICE:\t{device}\n\n")
        f.write(f"ONE CVD:\t{one_CVD}\n")
        f.write(f"COMBINED CVDS:\t{combined_CVDS}\n")
        f.write(f"USE SAMPLER:\t{use_sampler}\n")
        f.write(f"WEIGHTED LOSS:\t{weighted}\n\n")
        f.write(f"CLASS NAME:\t{class_name}\n")
        f.write(f"Using instance time embedding attention")
        f.write(f"Retraining pretrained encoder:\t{TRAIN_ENCODER}\n")
        # f.write(f"Getting attention weights for all ecgs (HF)")
        f.write(f"Using TEMPORAL ATTENTION")


    model_log(model, epochs, lr, batch_size, device, criterion, weighted, k_folds, one_CVD, class_name, combined_CVDS, use_sampler)

    if k_folds is None:
        split_data = split_data_patient_aware(X, y, pids, study_ids, batch_size=batch_size, use_sampler=use_sampler)
        train_loader, val_loader = split_data["train_loader"], split_data["val_loader"]
        y_train, y_val = split_data["y_train"], split_data["y_val"]
        distributions(y_train, y_val)
        criterion = nn.BCEWithLogitsLoss()

        train_loss, val_loss, val_aucs = train_model(model, train_loader, val_loader, device, criterion, epochs, lr, run_folder)
        plot_loss_curves(train_loss, val_loss, run_folder)
        plot_auc_curve_per_epoch(val_aucs, run_folder)

        model.load_state_dict(torch.load(os.path.join(run_folder, "best_model.pth"), map_location=device))

        val_probs_list, val_targets_list = [], []
        with torch.no_grad():
            for X_val_batch, y_val_batch, pids_batch, study_batch in val_loader:
                X_val_batch, y_val_batch,  = X_val_batch.to(device), y_val_batch.to(device)
                logits= model(X_val_batch)
                val_probs_list.append(torch.sigmoid(logits).cpu().numpy())
                val_targets_list.append(y_val_batch.cpu().numpy())

        val_probs = np.concatenate(val_probs_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)
        best_thresholds = compute_best_thresholds(val_targets, val_probs)

        # Evaluate and get metrics
        eval_results = evaluate_model(model, val_loader, device, run_folder, thresholds=best_thresholds, one_CVD=ONE_CVD, combined_CVDs=COMBINED_CVDS)
        metrics = eval_results["metrics"]

        # Append metrics to model.txt
        with open(model_file, "a") as f:
            f.write("\n======= EVALUATION METRICS =======\n")
            for k, v in metrics.items():
                if isinstance(v, list):
                    f.write(f"{k}: {v}\n")
                else:
                    f.write(f"{k}: {v:.4f}\n")

# ---------------------
# RUN
# ---------------------
if __name__ == "__main__":
    if COMBINED_CVDS and not ONE_CVD:
        RUN_FOLDER = create_run_folder(RUNS_BASE["COMBINED"])
    elif ONE_CVD and not COMBINED_CVDS:
        RUN_FOLDER = create_run_folder(RUNS_BASE[class_name])

    main(
        model = ECGTransformer(),
        X=features,
        y=labels,
        pids=pids,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        criterion=CRITERION,
        weighted=weighted,
        using_val_test=using_val_test,
        k_folds=K_FOLDS,
        run_folder=RUN_FOLDER,
        one_CVD=ONE_CVD,
        combined_CVDS=COMBINED_CVDS,
        use_sampler=USE_SAMPLER
    )
