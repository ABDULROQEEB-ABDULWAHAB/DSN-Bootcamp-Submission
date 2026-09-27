"""
PyTorch Deep Tabular Neural Network with Categorical Entity Embeddings and Residual Connections.
Provides smooth continuous manifold modeling to complement GBDT tree predictions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        return self.X_num[idx], self.X_cat[idx]

class TabularResNet(nn.Module):
    def __init__(self, num_features, cat_dims, emb_drop=0.1):
        super().__init__()
        
        # Categorical Entity Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, min(50, (cardinality + 1) // 2))
            for cardinality in cat_dims
        ])
        emb_total_dim = sum(emb.embedding_dim for emb in self.embeddings)
        self.emb_drop = nn.Dropout(emb_drop)
        
        total_in_dim = num_features + emb_total_dim
        
        # Deep Residual Architecture
        self.in_proj = nn.Sequential(
            nn.Linear(total_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2)
        )
        
        # ResNet Block 1
        self.block1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )
        self.act1 = nn.SiLU()
        
        # Bottleneck Layer
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x_num, x_cat):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_outs, dim=1)
        x_emb = self.emb_drop(x_emb)
        
        x = torch.cat([x_num, x_emb], dim=1)
        x = self.in_proj(x)
        
        # Residual connection
        res = x
        x = self.act1(self.block1(x) + res)
        
        out = self.head(x)
        return out

class PyTorchTabularWrapper:
    def __init__(self, batch_size=256, lr=2e-3, epochs=100, patience=15, device=None):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val, cat_cols):
        # Separate numerical and categorical columns
        num_cols = [c for c in X_train.columns if c not in cat_cols]
        
        # Clean numericals & scale
        X_tr_num = self.scaler.fit_transform(np.nan_to_num(X_train[num_cols].astype(float).values, nan=0.0))
        X_va_num = self.scaler.transform(np.nan_to_num(X_val[num_cols].astype(float).values, nan=0.0))
        
        # Categoricals
        X_tr_cat = X_train[cat_cols].astype(int).values
        X_va_cat = X_val[cat_cols].astype(int).values
        
        cat_dims = [max(int(X_train[c].astype(int).max()), int(X_val[c].astype(int).max())) + 10 for c in cat_cols]
        
        train_ds = TabularDataset(X_tr_num, X_tr_cat, y_train)
        val_ds = TabularDataset(X_va_num, X_va_cat, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        
        self.model = TabularResNet(len(num_cols), cat_dims).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        best_weights = None
        patience_count = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            for x_num, x_cat, y in train_loader:
                x_num, x_cat, y = x_num.to(self.device), x_cat.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(x_num, x_cat)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_num, x_cat, y in val_loader:
                    x_num, x_cat, y = x_num.to(self.device), x_cat.to(self.device), y.to(self.device)
                    preds = self.model(x_num, x_cat)
                    val_loss += criterion(preds, y).item() * len(y)
                    
            val_rmse = np.sqrt(val_loss / len(val_ds))
            
            if val_rmse < best_loss:
                best_loss = val_rmse
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break
                    
        self.model.load_state_dict(best_weights)
        self.model.to(self.device)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        return self

    def predict(self, X):
        X_num = self.scaler.transform(np.nan_to_num(X[self.num_cols].astype(float).values, nan=0.0))
        X_cat = X[self.cat_cols].astype(int).values
        
        ds = TabularDataset(X_num, X_cat)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for x_num, x_cat in loader:
                x_num, x_cat = x_num.to(self.device), x_cat.to(self.device)
                p = self.model(x_num, x_cat).cpu().numpy()
                preds.append(p)
                
        preds = np.vstack(preds).flatten()
        return np.clip(preds, 0, None)
