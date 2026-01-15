# dkn_model.py
# Запуск: pip install pandas numpy scikit-learn torch shap matplotlib
# python dkn_model.py --data patients.csv --target label

import argparse
import os
import random
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Определяем устройство
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ CUDA доступна. Используется: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠ CUDA не найдена. Используется CPU.")


# Optional: SHAP explainability
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))

DKN = SimpleMLP

numeric_cols = [
    "age",                # возраст
    "blood_pressure",     # давление
    "glucose",            # глюкоза
    "bmi",                # индекс массы тела
    "diabetes_duration",  # стаж диабета
    "creatinine",         # креатинин
    "albumin"             # альбумин в моче
]

categorical_cols = ["sex"]  # в базе нет категориальных признаков


def preprocess(df, numeric_cols, categorical_cols, fit_objects=None):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    # --- Числовые признаки ---
    X_num = df[numeric_cols]
    if fit_objects is None:
        num_imputer = SimpleImputer(strategy="mean")
        X_num_imputed = num_imputer.fit_transform(X_num)
        num_scaler = StandardScaler()
        X_num_scaled = num_scaler.fit_transform(X_num_imputed)
    else:
        num_imputer = fit_objects["num_imputer"]
        num_scaler = fit_objects["num_scaler"]
        X_num_imputed = num_imputer.transform(X_num)
        X_num_scaled = num_scaler.transform(X_num_imputed)

    # --- Категориальные признаки ---
    X_cat = df[categorical_cols] if categorical_cols else pd.DataFrame(index=df.index)

    if not X_cat.empty:
        if fit_objects is None:
            
            cat_imputer = SimpleImputer(strategy="constant", fill_value=-1)
            X_cat_imputed = cat_imputer.fit_transform(X_cat)

            ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
            X_cat_enc = ohe.fit_transform(X_cat_imputed)
        else:
            cat_imputer = fit_objects["cat_imputer"]
            ohe = fit_objects["ohe"]
            X_cat_imputed = cat_imputer.transform(X_cat)
            X_cat_enc = ohe.transform(X_cat_imputed)
    else:
        X_cat_enc = np.zeros((len(df), 0))
        cat_imputer = None
        ohe = None

    # --- объединяем ---
    X_full = np.hstack([X_num_scaled, X_cat_enc])

    # --- сэйвим объекты ---
    if fit_objects is None:
        fit_objects = {
            "num_imputer": num_imputer,
            "num_scaler": num_scaler,
            "cat_imputer": cat_imputer,
            "ohe": ohe
        }

    return X_full, fit_objects



def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    return running_loss / len(dataloader.dataset)


def eval_loop(model, dataloader, device):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            p = model(X_batch).cpu().numpy().ravel()
            ps.append(p)
            ys.append(y_batch.numpy().ravel())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return y_true, y_pred


def fit_model(X_train, y_train, X_val, y_val, input_dim, device='cpu',
              epochs=100, batch_size=128, lr=1e-3, pos_weight=None):
    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SimpleMLP(input_dim).to(device)
    if pos_weight is not None:
        # numeric pos_weight for BCEWithLogitsLoss (we use sigmoid output so use BCELoss)
        criterion = nn.BCELoss()
    else:
        criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_auc = -1
    best_state = None
    patience = 10
    waiting = 0

    for epoch in range(1, epochs + 1):
        loss = train_loop(model, train_loader, optimizer, criterion, device)
        yv, pv = eval_loop(model, val_loader, device)
        try:
            auc = roc_auc_score(yv, pv)
        except Exception:
            auc = 0.0
        print(f"Epoch {epoch:03d} loss={loss:.4f} val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                print("Early stopping")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def main(args):
    df = pd.read_csv(args.data)

    # Используем глобальные списки признаков
    global numeric_cols, categorical_cols  

    # Если в данных нет колонки — автоматически убираем из списка
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # Label
    y = df[args.target].astype(int).values

    # Train/val/test split
    X_full, fit_objs = preprocess(df, numeric_cols, categorical_cols, fit_objects=None)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_full, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=SEED
    )


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device, "Input dim:", X_train.shape[1])

    # Optionally calculate class weights
    pos_frac = y_train.mean()
    print(f"Positive fraction in train: {pos_frac:.3f}")

    model = fit_model(X_train, y_train, X_val, y_val, input_dim=X_train.shape[1], device=device,
                      epochs=200, batch_size=128, lr=1e-3)

    # Evaluate on test
    test_ds = TabularDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    ytrue, ypred = eval_loop(model, test_loader, device)
    auc = roc_auc_score(ytrue, ypred)
    ap = average_precision_score(ytrue, ypred)
    print(f"Test AUROC: {auc:.4f} AP: {ap:.4f}")

    # Calibration plot (reliability diagram)
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(ytrue, ypred, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.title('Calibration')
    plt.grid(True)
    plt.show()

    torch.save(model, "dkn_model_full.pth")
    print("Модель полностью сохранена в dkn_model_full.pth")


    # SHAP explainability (optional)
    if HAS_SHAP:
        explainer = shap.DeepExplainer(model, torch.tensor(X_train[:200], dtype=torch.float32).to(device))
        shap_values = explainer.shap_values(
            torch.tensor(X_test[:100], dtype=torch.float32).to(device),
            check_additivity=False
)

        # Простейший вывод
        shap.summary_plot(np.array(shap_values).squeeze(), X_test[:100], feature_names=[*numeric_cols, *list(fit_objs['ohe'].get_feature_names_out(categorical_cols))] if categorical_cols else numeric_cols)
    else:
        print("SHAP not installed — пропускаю explainability.")

    # Save model
    torch.save({"model_state": model.state_dict(), "fit_objs": fit_objs, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}, args.output or "dkn_model.pt")
    print("Saved model to", args.output or "dkn_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='CSV with features + target')
    parser.add_argument('--target', type=str, default='label', help='Target column name (0/1)')
    parser.add_argument('--output', type=str, default='dkn_model.pt', help='Output model path')
    args = parser.parse_args()
    main(args)

