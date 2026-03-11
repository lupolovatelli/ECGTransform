import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve,
    f1_score, confusion_matrix, precision_score, recall_score
)
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def create_run_folder(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]
    run_indices = []
    for d in existing_runs:
        try:
            run_indices.append(int(d.split("_")[1]))
        except:
            pass
    next_idx = max(run_indices) + 1 if run_indices else 1
    run_folder = os.path.join(base_dir, f"run_{next_idx:03d}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder



def setup_logger(log_file_path):
    """
    Redirect stdout and stderr to both terminal and a log file.
    """
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.__stdout__
            self.log = open(filename, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    logger = Logger(log_file_path)
    sys.stdout = logger
    sys.stderr = logger  #  capture errors too

    return logger  # allows clean shutdown later






def plot_loss_curves(train_losses, val_losses, run_folder=None):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss", marker="")
    plt.plot(val_losses, label="Validation Loss", marker="")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    if run_folder:
        plt.savefig(os.path.join(run_folder, "loss_curve.png"), dpi=300)
    plt.show()


# edit to only binary (pos class vs neg class)
def plot_roc_auc_per_class(y_true, y_prob, class_names=None, run_folder=None):
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.3f})")
        except ValueError:
            plt.plot([], [], label=f"{class_names[i]} (AUC = N/A)")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Class")
    plt.legend(loc="lower right")
    plt.grid(True)
    if run_folder:
        plt.savefig(os.path.join(run_folder, "roc_auc_per_class.png"), dpi=300)
    plt.show()




# edit to only binary (pos class vs neg class)
def compute_best_thresholds(y_true, y_prob):
    """Compute per-class optimal threshold using Youden's J (tpr - fpr)."""

    n_classes = y_true.shape[1]
    thresholds = np.zeros(n_classes)

    for i in range(n_classes):
        # If only one class present, fallback
        if len(np.unique(y_true[:, i])) < 2:
            thresholds[i] = 0.5
            continue

        fpr, tpr, thr = roc_curve(y_true[:, i], y_prob[:, i])
        youden = tpr - fpr
        idx = np.nanargmax(youden)

        best_thr = thr[idx]   # scalar value

        # Proper numerical safety check
        if np.isinf(best_thr) or np.isnan(best_thr):
            best_thr = 0.5

        thresholds[i] = best_thr

    return thresholds




# ---------------------
# Compute class counts (useful info and for optional weighting)
# ---------------------
def get_pos_weights(labels_train,device):
    y_train_np = labels_train.numpy().astype(int)
    pos_counts = np.sum(y_train_np, axis=0).astype(int)
    neg_counts = y_train_np.shape[0] - pos_counts
    # print("Positive counts  :", pos_counts)
    # print("Negative counts  :", neg_counts)
    prevalence = pos_counts / y_train_np.shape[0] * 100
    # print("Prevalence (pct) :", np.round(prevalence, 4))

    # compute pos_weight but clip to avoid extremely large weights (stabilizes training)
    pos_weight = (neg_counts / (pos_counts + 1e-6)).astype(float)
    # clip
    pos_weight = np.clip(pos_weight, 1.0, 10.0)
    pos_weight_torch = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    print("Using pos_weight (clipped) per class:", pos_weight_torch.cpu().numpy())

    return pos_weight_torch



def distributions(labels_train, labels_test):
    # labels are already numpy arrays
    y_train_np = labels_train.astype(int)
    pos_counts_train = np.sum(y_train_np, axis=0).astype(int)
    neg_counts_train = y_train_np.shape[0] - pos_counts_train
    prevalence_train = pos_counts_train / y_train_np.shape[0] * 100

    y_test_np = labels_test.astype(int)
    pos_counts_test = np.sum(y_test_np, axis=0).astype(int)
    neg_counts_test = y_test_np.shape[0] - pos_counts_test
    prevalence_test = pos_counts_test / y_test_np.shape[0] * 100

    print(f"\nTraining set pos counts: {pos_counts_train}")
    print(f"Training set neg counts: {neg_counts_train}")
    print("Training label Prevalence in %:", np.round(prevalence_train, 4))

    print(f"\nTesting set pos counts: {pos_counts_test}")
    print(f"Testing set neg counts: {neg_counts_test}")
    print(f"Testing label Prevalence in %:", np.round(prevalence_test, 4))





def plot_avg_std_loss_curve(fold_train_losses, fold_val_losses, run_folder):
    """
    Plots the average training and validation loss across folds,
    with shaded area showing standard deviation.

    fold_train_losses: list of lists, each inner list is training loss per epoch for a fold
    fold_val_losses: list of lists, each inner list is validation loss per epoch for a fold
    save_path: optional path to save the figure
    """
    # Convert to numpy arrays for easier calculations
    train_losses_arr = np.array([np.array(l) for l in fold_train_losses])
    val_losses_arr = np.array([np.array(l) for l in fold_val_losses])

    # Pad sequences if folds have different epochs
    max_epochs = max(train_losses_arr.shape[1], val_losses_arr.shape[1])

    def pad_losses(loss_arr, max_len):
        padded = []
        for l in loss_arr:
            if len(l) < max_len:
                l = np.pad(l, (0, max_len - len(l)), mode='edge')
            padded.append(l)
        return np.array(padded)

    train_losses_arr = pad_losses(train_losses_arr, max_epochs)
    val_losses_arr = pad_losses(val_losses_arr, max_epochs)

    # Compute mean and std
    train_mean = np.mean(train_losses_arr, axis=0)
    train_std = np.std(train_losses_arr, axis=0)
    val_mean = np.mean(val_losses_arr, axis=0)
    val_std = np.std(val_losses_arr, axis=0)

    epochs = np.arange(1, max_epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mean, label='Train Loss', color='blue')
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
    plt.plot(epochs, val_mean, label='Validation Loss', color='orange')
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Train/Validation Loss Across Folds')
    plt.legend()
    plt.grid(True)
    if run_folder:
        plt.savefig(os.path.join(run_folder, "avg_k_fold_loss_curve.png"), dpi=300)
    plt.show()




def plot_auc_curve_per_epoch(val_aucs, run_folder):
    import matplotlib.pyplot as plt
    epochs = range(1, len(val_aucs) + 1)

    plt.figure()
    plt.plot(epochs, val_aucs)
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC")
    plt.title("Validation AUC per Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "val_auc_curve.png"), dpi=300)
    plt.close()




def get_class_data(class_name, class_names, X, y):
    """
    Inputs:
        class_names: ["VA", "AF", "IS", "HF"]
        X --> raw features: (N, 5000, 12)
        y --> raw labels:   (N, 4)

    Outputs:
        X_class --> features of chosen class + controls (N_selected, 5000, 12)
        y_class --> labels for the chosen class (N_selected, 1)
        selected_mask --> boolean mask applied to original dataset
    """

    assert class_name in class_names, "Invalid class name"
    class_idx = class_names.index(class_name)

    # Positive samples: the chosen class
    pos_mask = y[:, class_idx] == 1

    # Controls: patients with ALL zeros (healthy)
    neg_mask = np.all(y == 0, axis=1)

    # Combined mask
    selected_mask = pos_mask | neg_mask

    # Apply selection
    X_class = X[selected_mask]
    y_class = y[selected_mask][:, class_idx:class_idx+1]  # keep 2D (N,1)

    # FIXED: correct class counts (must use selected_mask)
    pos_count = np.sum(pos_mask)
    neg_count = np.sum(neg_mask)

    print("Selected class features shapes:", X_class.shape)
    print("Selected class labels shapes:", y_class.shape)
    print("Positive class counts:", pos_count)
    print("Negative class counts:", neg_count)
    print("Total selected:", np.sum(selected_mask))

    return X_class, y_class, selected_mask




def combine_CVD_classes(y):
    """ inputs:
            X --> raw features: (N,leads=12,timesteps=5000)
            y --> raw labels: (N,4)

        outputs:
            y_combined --> labels of chosen class: (N_class,1)
        """
    y_combined = []

    for row in y:
        if any(row):
            y_combined.append(1)
        else:
            y_combined.append(0)

    y_combined = np.array(y_combined, dtype=np.float32).reshape(-1, 1)

    return y_combined



# preds_path = "./RUNS/run_023/test_preds.npy"
# probs_path = "./RUNS/run_023/test_probs.npy"

# preds = np.load(preds_path)
# probs = np.load(probs_path)

# print("Probs:",probs)
# print("\nPreds:",preds)

# def split_data_by_patient(df,test_size=0.2,seed=42):

#     # get patient id:
#     patients = df["subject_id"].unique()

#     # split:
#     train_patients, test_patients = train_test_split(patients,
#                                                      test_size=test_size,
#                                                      random_state=seed,
#                                                      shuffle=True)

#     train_df = df[df["subject_id"].isin(train_patients)]
#     test_df = df[df["subject_id"].isin(test_patients)]

#     print(f"Train patients: {len(train_patients)} → rows: {len(train_df)}")
#     print(f"Test patients:  {len(test_patients)} → rows: {len(test_df)}")

#     return train_df, test_df

def model_log(model,epochs,lr,batch_size,device,criterion,weighted,k_folds,one_CVD,class_name,combined_CVDS,use_sampler):
    print("================  MODEL CONFIF  =================")
    print("\nMODEL:",model.name)
    print("EPOCHS:",epochs)
    print("LR:",lr)
    print("BATCH SIZE:",batch_size)
    print("DEVICE",device)
    print("LOSS FUNCTION:",criterion)


    if weighted:
        print("Using class weighted BCE Loss function")

    if k_folds is not None:
        print(f"Running cross validation with {k_folds} folds")

    if one_CVD:
        print("Running binary classification for class:",class_name)

    if combined_CVDS:
        print("Running binary prediction on combined labels for all CVDs")

    if use_sampler:
        print("Using Sampler in training set")

    print("\n===============================================")



def split_data_patient_aware(features, labels, pids, study_ids,
                             batch_size=8, use_sampler=False, test_size=0.2, random_state=42):

    # Unique patients
    unique_pids = np.unique(pids)
    train_pids, val_pids = train_test_split(unique_pids, test_size=test_size, random_state=random_state)

    # Masks
    train_mask = np.isin(pids, train_pids)
    val_mask = np.isin(pids, val_pids)

    # Fix labels
    labels = np.asarray(labels, dtype=np.float32)

    # Extract splits
    X_train, X_val = features[train_mask], features[val_mask]
    y_train, y_val = labels[train_mask], labels[val_mask]
    pid_train, pid_val = pids[train_mask], pids[val_mask]
    study_ids_train, study_ids_val = study_ids[train_mask], study_ids[val_mask]

    # Build datasets
    train_dataset = ECGDataset(X_train, y_train, pid_train, study_ids_train, train=True)
    val_dataset = ECGDataset(X_val, y_val, pid_val, study_ids_val, train=False)

    # Optional sampler
    if use_sampler:
        y_flat = torch.tensor(y_train, dtype=torch.int).flatten()
        class_counts = torch.bincount(y_flat)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[y_flat]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "X_train": X_train, "y_train": y_train,
        "pids_train": pid_train, "study_ids_train": study_ids_train,
        "X_val": X_val, "y_val": y_val,
        "pids_val": pid_val, "study_ids_val": study_ids_val
    }






# def plot_attn_heatmap(attn_weights, patient_ids=None, ecg_ids=None, mask=None, save_path=None):
#     """
#     Plot attention weights as a heatmap: patients (y-axis) vs ECG instances (x-axis).
#     Only the first 32 patients are plotted.
#     Masked positions (padded ECGs) will not be shown.

#     mask: boolean array same shape as attn_weights (True for valid ECGs, False for padded)
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import LinearSegmentedColormap

#     # LIMIT TO FIRST 32 PATIENTS
#     max_patients = 32
#     attn_weights = attn_weights[:max_patients]
#     if mask is not None:
#         mask = mask[:max_patients]

#     num_patients, num_ecgs = attn_weights.shape

#     if patient_ids is None:
#         patient_ids = [f"P{i}" for i in range(num_patients)]
#     else:
#         patient_ids = patient_ids[:max_patients]  # slice IDs

#     if ecg_ids is None:
#         ecg_ids = [f"ECG{j+1}" for j in range(num_ecgs)]

#     # Mask padded ECGs
#     if mask is not None:
#         attn_weights_plot = np.ma.masked_where(~mask, attn_weights)
#     else:
#         attn_weights_plot = attn_weights

#     # Dark red to yellow colormap
#     darkred_yellow = LinearSegmentedColormap.from_list(
#         "darkred_yellow", ["darkred", "red", "orange", "yellow"]
#     )

#     fig, ax = plt.subplots(figsize=(10, 6))
#     cax = ax.imshow(attn_weights_plot, aspect='auto', cmap=darkred_yellow)

#     ax.set_yticks(range(num_patients))
#     ax.set_yticklabels(patient_ids)
#     ax.set_xticks(range(num_ecgs))
#     ax.set_xticklabels(ecg_ids, rotation=90)

#     fig.colorbar(cax, ax=ax)
#     plt.title("Attention Heatmap: Patients vs. ECGs (Dark Red to Yellow)")
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300)
#         print(f"Heatmap saved to {save_path}")
#         plt.close(fig)  # Close the figure to free memory
#     else:
#         plt.show()
def plot_attn_heatmap(attn_weights, patient_ids=None, ecg_ids=None, mask=None, labels=None, save_path=None):
    """
    Plot attention weights as a heatmap: patients (y-axis) vs ECG instances (x-axis).
    Now includes patient labels (0/1) next to patient IDs.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    max_patients = 32
    attn_weights = attn_weights[:max_patients]
    if mask is not None:
        mask = mask[:max_patients]
    if labels is not None:
        labels = labels[:max_patients]

    num_patients, num_ecgs = attn_weights.shape

    # Add "(label)" to patient ID
    if labels is not None:
        patient_ids = [f"{pid} ({lbl})" for pid, lbl in zip(patient_ids, labels)]
    else:
        patient_ids = patient_ids

    if ecg_ids is None:
        ecg_ids = [f"ECG{j+1}" for j in range(num_ecgs)]

    # Mask padded ECGs
    if mask is not None:
        attn_weights_plot = np.ma.masked_where(~mask, attn_weights)
    else:
        attn_weights_plot = attn_weights

    darkred_yellow = LinearSegmentedColormap.from_list(
        "darkred_yellow", ["darkred", "red", "orange", "yellow"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(attn_weights_plot, aspect='auto', cmap=darkred_yellow)

    ax.set_yticks(range(num_patients))
    ax.set_yticklabels(patient_ids)
    ax.set_xticks(range(num_ecgs))
    ax.set_xticklabels(ecg_ids, rotation=90)

    fig.colorbar(cax, ax=ax)
    plt.title("Attention Heatmap with Labels")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()



def check_model_gradients(model):
    """
    Print mean, max, and min absolute gradient of all model parameters.
    """
    print("\n=== MODEL PARAMETER GRADIENTS ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_abs = param.grad.abs()
                print(f"{name:50s} | mean: {grad_abs.mean():.3e} | max: {grad_abs.max():.3e} | min: {grad_abs.min():.3e}")
            else:
                print(f"{name:50s} | grad is None")
    print("=================================\n")



from torch.utils.data import Dataset

# ---------------------
# DATA AUGMENTATION
# ---------------------
class ECGAugmenter:
    """
    Simple ECG data augmentation: adds random noise and shifts waveform.
    """
    def __init__(self, p_noise=0.3, noise_std=0.005, p_shift=0.2, max_shift=100):
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.p_shift = p_shift
        self.max_shift = max_shift

    def __call__(self, ecg):
        # ecg: [C, L] tensor
        if torch.rand(1) < self.p_noise:
            ecg = ecg + torch.randn_like(ecg) * self.noise_std

        if torch.rand(1) < self.p_shift:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
            ecg = torch.roll(ecg, shifts=shift, dims=1)

        return ecg


# ---------------------
# Dataset
# ---------------------
class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECGs.
    Returns one ECG per sample: [C, L].
    """
    def __init__(self, X, y, pids, study_ids, train=True):
        """
        X : np.ndarray or torch.tensor [N, C, L]
        y : np.ndarray or torch.tensor [N, n_classes]
        pids : np.ndarray [N] patient IDs
        study_ids : np.ndarray [N] study IDs
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.pids = torch.tensor(pids)
        self.study_ids = torch.tensor(study_ids)

        self.train = train
        self.augment = ECGAugmenter() if train else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # [C, L]
        y = self.y[idx]

        if self.train and self.augment is not None:
            x = self.augment(x)

        return x, y, self.pids[idx], self.study_ids[idx]



