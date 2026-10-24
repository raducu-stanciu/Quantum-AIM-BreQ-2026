import time
from pathlib import Path
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from scipy.stats import spearmanr
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- CONSTANTE GLOBALE ---
SEED = 42
SAMPLE_FRACTION = 1.0
N_QUBITS = 4
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.05
NOISE_LEVELS = [0.001, 0.01, 0.05]
NATIVE_GATES = {"RX", "RY", "RZ", "Rot", "Hadamard", "CNOT", "CZ"}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# --- PREGĂTIREA DATELOR ---
def undersample(X_part, y_part, seed):
    rng = np.random.default_rng(seed)
    fraud = np.flatnonzero(y_part == 1)
    legit = rng.choice(np.flatnonzero(y_part == 0), size=len(fraud), replace=False)
    keep = rng.permutation(np.concatenate([fraud, legit]))
    return X_part.iloc[keep], y_part[keep]

class AngleEncoder:
    raw_columns = ["Time", "Amount"]

    def __init__(self, n_components, seed):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.angles = MinMaxScaler(feature_range=(0, np.pi))

    def _scaled(self, frame):
        out = frame.copy()
        out[self.raw_columns] = self.scaler.transform(frame[self.raw_columns])
        return out.to_numpy()

    def fit(self, frame):
        self.scaler.fit(frame[self.raw_columns])
        pca_out = self.pca.fit_transform(self._scaled(frame))
        self.angles.fit(pca_out)
        return self

    def transform(self, frame):
        arr = self._scaled(frame)
        arr = self.pca.transform(arr)
        arr = self.angles.transform(arr)
        arr = np.clip(arr, 0, np.pi)
        return arr

# --- ÎNCĂRCAREA DATELOR ȘI ÎMPĂRȚIREA COMUNĂ (COMMON SPLIT) ---
DATA_PATH = Path("data/creditcard.csv")
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    df = fetch_openml(data_id=42175, as_frame=True, parser="auto").frame
df["Class"] = df["Class"].astype(int)

# Utilizarea împărțirii (split-ului) standard specificată în template-ul Task 2:
if SAMPLE_FRACTION == 1.0:
    # Dacă fișierul există deja, încarcă-l, altfel refă împărțirile:
    try:
        split = np.load("split_indices.npz")
        X_all, y_all = df.drop(columns="Class"), df["Class"].to_numpy()
        X_train, y_train = X_all.loc[split["train"]], y_all[split["train"]]
        X_val, y_val = X_all.loc[split["val"]], y_all[split["val"]]
        X_test, y_test = X_all.loc[split["test"]], y_all[split["test"]]
    except FileNotFoundError:
        X = df.drop(columns="Class")
        y = df["Class"].to_numpy()
        X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=SEED)
        np.savez("split_indices.npz", train=X_train.index.to_numpy(), val=X_val.index.to_numpy(), test=X_test.index.to_numpy())

X_bal, y_bal = undersample(X_train, y_train, SEED)
encoder = AngleEncoder(N_QUBITS, SEED).fit(X_bal)
Z_bal = encoder.transform(X_bal)
Z_val = encoder.transform(X_val)
Z_test = encoder.transform(X_test)

# --- FUNCȚII DE ANTRENAMENT ---
def predict(Z, weights, model, chunk=20000):
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([model(X_t[k:k + chunk], weights) for k in range(0, len(X_t), chunk)]).numpy()

def train_vqc(Z_train, y_train_part, Z_valid, y_valid, seed, model, init, epochs=EPOCHS, verbose=True):
    set_seed(seed)
    weights = init(seed)
    optimizer = torch.optim.Adam([weights], lr=LEARNING_RATE)
    loss_fn = torch.nn.BCELoss()
    X_t = torch.tensor(Z_train)
    y_t = torch.tensor(y_train_part, dtype=torch.float64)
    history, best_ap, best_weights = [], -1.0, None
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(X_t))
        losses = []
        for k in range(0, len(X_t), BATCH_SIZE):
            idx = order[k:k + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[idx], weights), y_t[idx])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_ap = average_precision_score(y_valid, predict(Z_valid, weights, model))
        history.append({"epoch": epoch, "train loss": np.mean(losses), "validation AUPRC": val_ap})
        if val_ap > best_ap:
            best_ap, best_weights = val_ap, weights.detach().clone()
        if verbose:
            print(f"epoch {epoch:2d}   train loss {np.mean(losses):.4f}   validation AUPRC {val_ap:.3f}")
    return best_weights, pd.DataFrame(history)

# --- INSTRUMENTE PENTRU ZGOMOT ȘI METRICI (TASK 2) ---
def noisy_qnode(circuit_fn, p, n_wires=N_QUBITS):
    noisy_device = qml.noise.insert(qml.device("default.mixed", wires=n_wires), qml.DepolarizingChannel, p, position="all")
    qnode = qml.QNode(circuit_fn, noisy_device, interface="torch", diff_method="backprop")
    return qml.transforms.decompose(qnode, gate_set=NATIVE_GATES)

def noisy_predict(Z, weights, p, circuit_fn, readout=lambda expval: (1 - expval) / 2, n_wires=N_QUBITS, chunk=2000):
    noisy_circuit = noisy_qnode(circuit_fn, p, n_wires)
    X_t = torch.tensor(Z)
    with torch.no_grad():
        return torch.cat([readout(noisy_circuit(X_t[k:k + chunk], weights)) for k in range(0, len(X_t), chunk)]).numpy()

def noise_locations(circuit_fn, *args, n_wires=N_QUBITS):
    qnode = qml.transforms.decompose(qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires)), gate_set=NATIVE_GATES)
    return sum(size * count for size, count in qml.specs(qnode)(*args)["resources"].gate_sizes.items())

def spsa_gradient(loss_of, weights, c=0.1, generator=None):
    delta = 2 * torch.randint(0, 2, weights.shape, generator=generator).to(weights.dtype) - 1
    with torch.no_grad():
        difference = loss_of(weights + c * delta) - loss_of(weights - c * delta)
    return difference / (2 * c) * delta

def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * (precision * recall) / np.clip(precision + recall, 1e-12, None)
    return thresholds[np.argmax(f1[:-1])]

def evaluate(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "AUPRC": average_precision_score(y_true, scores),
        "AUROC": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }

def evaluate_under_noise(circuit_fn, weights, Z_valid, Z_test_part, y_valid=y_val, y_test_part=y_test,
                         readout=lambda expval: (1 - expval) / 2, noise_levels=NOISE_LEVELS, n_wires=N_QUBITS):
    clean = qml.QNode(circuit_fn, qml.device("default.qubit", wires=n_wires), interface="torch")
    with torch.no_grad():
        clean_val = readout(clean(torch.tensor(Z_valid), weights)).numpy()
        clean_test = readout(clean(torch.tensor(Z_test_part), weights)).numpy()
    rows = []
    for p in [0.0, *noise_levels]:
        if p == 0.0:
            val_scores, test_scores = clean_val, clean_test
        else:
            val_scores = noisy_predict(Z_valid, weights, p, circuit_fn, readout, n_wires)
            test_scores = noisy_predict(Z_test_part, weights, p, circuit_fn, readout, n_wires)
        threshold = choose_threshold(y_valid, val_scores)
        rows.append({"p": p, **evaluate(y_test_part, test_scores, threshold), "Spearman rho": spearmanr(clean_test, test_scores).statistic, "threshold": threshold})
    return pd.DataFrame(rows).set_index("p")

# ==========================================
# EXEMPLU UTILIZARE / TEMPLATE TASK 2
# ==========================================
# def my_circuit(x, weights):
#     ... # codificarea și ansatz-ul tău
#     return qml.expval(qml.PauliZ(0))
#
# device = qml.device("default.qubit", wires=N_QUBITS)
# my_qnode = qml.QNode(my_circuit, device, interface="torch", diff_method="backprop")
# my_model = lambda x, w: (1 - my_qnode(x, w)) / 2
# my_init = lambda seed: torch.nn.Parameter(...) # ponderi inițializate corespunzător
#
# my_weights, my_history = train_vqc(Z_bal, y_bal, Z_val, y_val, SEED, model=my_model, init=my_init)
# print("noise locations:", noise_locations(my_circuit, torch.tensor(Z_bal[0]), my_weights))
# evaluare = evaluate_under_noise(my_circuit, my_weights, Z_val, Z_test)
# print(evaluare)