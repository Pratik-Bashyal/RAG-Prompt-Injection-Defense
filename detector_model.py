"""Dual-branch MLP detector, ported from the official CustomMLP (classification.py).

Original (8B): gradient branch Linear(400000->4096)+BN, hidden branch BN(4096),
concat -> 1024 -> 256 -> 64 -> 16 -> 2.

Here it is generalized so it works for hidden-only, gradient-only, or fused
features, and for any model hidden size. Input layout is [gradient || hidden].
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DualMLP(nn.Module):
    def __init__(self, grad_dim, hidden_dim, grad_proj=4096):
        super().__init__()
        self.grad_dim = grad_dim
        self.hidden_dim = hidden_dim
        comb = 0
        if grad_dim > 0:
            self.fc_g = nn.Linear(grad_dim, grad_proj)
            self.bn_g = nn.BatchNorm1d(grad_proj)
            comb += grad_proj
        if hidden_dim > 0:
            self.bn_h = nn.BatchNorm1d(hidden_dim)
            comb += hidden_dim
        self.head = nn.Sequential(
            nn.Linear(comb, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        parts = []
        if self.grad_dim > 0:
            g = torch.relu(self.fc_g(x[:, :self.grad_dim]))
            parts.append(self.bn_g(g))
        if self.hidden_dim > 0:
            parts.append(self.bn_h(x[:, self.grad_dim:]))
        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.head(x)


class Detector:
    def __init__(self, grad_dim, hidden_dim, device=None):
        self.grad_dim = grad_dim
        self.hidden_dim = hidden_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualMLP(grad_dim, hidden_dim).to(self.device)

    def fit(self, X, y, epochs=300, batch_size=32, lr=1e-4):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.long)
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            running = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item()
            avg = running / len(loader)
            if (epoch + 1) % 25 == 0:
                print(f"  epoch {epoch+1}/{epochs}  loss {avg:.4f}")
            if avg <= 1e-4 and epoch > 60:
                break
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        t = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return torch.softmax(self.model(t), dim=1).cpu().numpy()

    def save(self, path):
        torch.save({"grad_dim": self.grad_dim, "hidden_dim": self.hidden_dim,
                    "state": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path, device=None):
        ckpt = torch.load(path, map_location="cpu")
        d = cls(ckpt["grad_dim"], ckpt["hidden_dim"], device=device)
        d.model.load_state_dict(ckpt["state"])
        d.model.to(d.device)
        return d
