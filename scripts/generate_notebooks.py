#!/usr/bin/env python3
"""Generate 10 practical Jupyter notebooks for the ML & DL course."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "practicals"


def _uid() -> str:
    return str(uuid.uuid4())


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text.strip(), id=_uid())


def code(text: str) -> nbformat.NotebookNode:
    return new_code_cell(text.strip(), id=_uid())


def exercise_header(num: str, title: str, job: str) -> list[nbformat.NotebookNode]:
    return [
        md(f"### Exercise {num}: {title}"),
        md(f"**Your job**: {job}"),
    ]


def reflection_cell(day: int, topic: str) -> nbformat.NotebookNode:
    return md(
        f"""## Reflection (Day {day})

Take a few minutes to answer in your own words:

1. What was the most important concept you practiced today ({topic})?
2. Where did you get stuck, and how did you resolve it?
3. How would you explain today's material to a classmate in two sentences?
4. What would you like to explore further?"""
    )


def week1_intro(day: int, title: str, blurb: str) -> list[nbformat.NotebookNode]:
    return [
        md(f"# Day {day:02d}: {title}"),
        md(
            f"Welcome to practical {day}! {blurb}\n\n"
            "Work through each exercise in order. Look for `# YOUR CODE HERE` markers "
            "— replace the reference implementations with your own solutions."
        ),
    ]


def week2_intro(day: int, title: str, blurb: str) -> list[nbformat.NotebookNode]:
    return [
        md(f"# Day {day:02d}: {title}"),
        md(
            f"Welcome to practical {day}! {blurb}\n\n"
            "This notebook follows the MIT lab style: abstract base classes, PyTorch, "
            "and LaTeX in markdown. Fill in methods marked `# YOUR CODE HERE`."
        ),
    ]


def torch_setup() -> nbformat.NotebookNode:
    return code(
        """import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

device = torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)"""
    )


# ---------------------------------------------------------------------------
# Day 1 — Gradients & MLE (numpy)
# ---------------------------------------------------------------------------
def build_day01() -> nbformat.NotebookNode:
    cells = week1_intro(
        1,
        "Gradients & Maximum Likelihood",
        "We review multivariate calculus and MLE using NumPy on small synthetic data.",
    )
    cells.append(
        code(
            """import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)"""
        )
    )
    cells += exercise_header(
        "1.1",
        "Numerical gradient check",
        "Implement `numerical_gradient` using central finite differences and compare to the analytical gradient of $f(w) = w^\\top A w$.",
    )
    cells.append(
        md(
            "For $f(\\mathbf{w}) = \\mathbf{w}^\\top A \\mathbf{w}$ with symmetric $A$, "
            "the gradient is $\\nabla f = 2 A \\mathbf{w}$."
        )
    )
    cells.append(
        code(
            """def f(w, A):
    return w @ A @ w

def analytical_gradient(w, A):
    # YOUR CODE HERE
    return 2 * A @ w

def numerical_gradient(w, A, eps=1e-5):
    # YOUR CODE HERE
    grad = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        e = np.zeros_like(w, dtype=float)
        e[i] = eps
        grad[i] = (f(w + e, A) - f(w - e, A)) / (2 * eps)
    return grad

A = np.array([[2.0, 0.5], [0.5, 1.0]])
w = np.array([1.0, -0.5])
g_num = numerical_gradient(w, A)
g_ana = analytical_gradient(w, A)
print("Numerical:", g_num)
print("Analytical:", g_ana)
print("Max |diff|:", np.max(np.abs(g_num - g_ana)))"""
        )
    )
    cells += exercise_header(
        "1.2",
        "Gradient descent on a quadratic",
        "Implement `gradient_descent` to minimize $L(\\mathbf{w}) = \\|\\mathbf{X}\\mathbf{w} - \\mathbf{y}\\|_2^2$.",
    )
    cells.append(
        code(
            """def mse_loss(w, X, y):
    # YOUR CODE HERE
    residual = X @ w - y
    return np.mean(residual ** 2)

def mse_gradient(w, X, y):
    # YOUR CODE HERE
    n = X.shape[0]
    return (2 / n) * X.T @ (X @ w - y)

def gradient_descent(X, y, lr=0.1, steps=50):
    # YOUR CODE HERE
    w = np.zeros(X.shape[1])
    losses = []
    for _ in range(steps):
        losses.append(mse_loss(w, X, y))
        w = w - lr * mse_gradient(w, X, y)
    return w, losses

n, d = 40, 3
X = np.random.randn(n, d)
w_true = np.array([1.5, -2.0, 0.5])
y = X @ w_true + 0.1 * np.random.randn(n)

w_hat, losses = gradient_descent(X, y)
print("Estimated w:", w_hat)
print("True w:", w_true)

plt.plot(losses)
plt.xlabel("step")
plt.ylabel("MSE")
plt.title("Gradient descent on quadratic loss")
plt.show()"""
        )
    )
    cells += exercise_header(
        "1.3",
        "MLE for a Gaussian mean",
        "Derive and implement the MLE for the mean of $\\mathcal{N}(\\mu, \\sigma^2 I)$ with known $\\sigma$.",
    )
    cells.append(
        md(
            "Given i.i.d. samples $x_1,\\ldots,x_n \\sim \\mathcal{N}(\\mu, \\sigma^2)$, "
            "the MLE of $\\mu$ is $\\hat{\\mu} = \\frac{1}{n}\\sum_i x_i$."
        )
    )
    cells.append(
        code(
            """def mle_gaussian_mean(samples):
    # YOUR CODE HERE
    return np.mean(samples)

samples = np.random.randn(100) * 0.5 + 2.0
print("MLE mean:", mle_gaussian_mean(samples))
print("Sample mean:", samples.mean())"""
        )
    )
    cells += exercise_header(
        "1.4",
        "MLE for Bernoulli coin flips",
        "Implement the MLE for Bernoulli($p$) and compare to the method-of-moments estimator.",
    )
    cells.append(
        code(
            """def mle_bernoulli_p(flip_counts, n_flips):
    # YOUR CODE HERE — flip_counts is number of heads
    return flip_counts / n_flips

flip_counts, n_flips = 37, 50
p_hat = mle_bernoulli_p(flip_counts, n_flips)
print(f"MLE p-hat = {p_hat:.3f}")"""
        )
    )
    cells.append(reflection_cell(1, "gradients and MLE"))
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


# ---------------------------------------------------------------------------
# Day 2 — sklearn regression & classification
# ---------------------------------------------------------------------------
def build_day02() -> nbformat.NotebookNode:
    cells = week1_intro(
        2,
        "Statistical Learning with scikit-learn",
        "Linear models, regularization, and classification on built-in datasets.",
    )
    cells.append(
        code(
            """import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)"""
        )
    )
    cells += exercise_header(
        "2.1",
        "Linear regression baseline",
        "Fit `LinearRegression` on synthetic data and report train/test MSE.",
    )
    cells.append(
        code(
            """X, y = make_regression(n_samples=200, n_features=5, noise=4.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# YOUR CODE HERE
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, y_pred))"""
        )
    )
    cells += exercise_header(
        "2.2",
        "Ridge regression",
        "Compare Ridge ($\\alpha=1$) vs OLS on the same split.",
    )
    cells.append(
        code(
            """# YOUR CODE HERE
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print("Ridge test MSE:", mean_squared_error(y_test, ridge.predict(X_test)))
print("OLS  test MSE:", mean_squared_error(y_test, model.predict(X_test)))"""
        )
    )
    cells += exercise_header(
        "2.3",
        "Logistic regression",
        "Train a classifier on Breast Cancer Wisconsin and report accuracy.",
    )
    cells.append(
        code(
            """data = load_breast_cancer()
Xc, yc = data.data, data.target
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.25, random_state=0, stratify=yc)

# YOUR CODE HERE
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
clf.fit(Xc_tr, yc_tr)
yc_pred = clf.predict(Xc_te)
print("Accuracy:", accuracy_score(yc_te, yc_pred))
print(classification_report(yc_te, yc_pred, target_names=data.target_names))"""
        )
    )
    cells += exercise_header(
        "2.4",
        "Bias–variance intuition",
        "Plot train vs test MSE as a function of Ridge $\\alpha$.",
    )
    cells.append(
        code(
            """alphas = np.logspace(-3, 3, 15)
train_mse, test_mse = [], []
for a in alphas:
    # YOUR CODE HERE
    r = Ridge(alpha=a)
    r.fit(X_train, y_train)
    train_mse.append(mean_squared_error(y_train, r.predict(X_train)))
    test_mse.append(mean_squared_error(y_test, r.predict(X_test)))

plt.semilogx(alphas, train_mse, label="train")
plt.semilogx(alphas, test_mse, label="test")
plt.xlabel("alpha")
plt.ylabel("MSE")
plt.legend()
plt.title("Ridge regularization path")
plt.show()"""
        )
    )
    cells.append(reflection_cell(2, "statistical learning with sklearn"))
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


# ---------------------------------------------------------------------------
# Day 3 — MNIST MLP
# ---------------------------------------------------------------------------
def build_day03() -> nbformat.NotebookNode:
    cells = week2_intro(
        3,
        "Deep Neural Networks — MNIST MLP",
        "Build and train a multi-layer perceptron on a small MNIST subset.",
    )
    cells.append(torch_setup())
    cells.append(
        md(
            "We classify digits using a feed-forward network. "
            "Cross-entropy loss for one-hot labels $y$ and logits $z$ is "
            "$\\mathcal{L} = -\\sum_c y_c \\log \\mathrm{softmax}(z)_c$."
        )
    )
    cells.append(
        code(
            """from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root="/tmp/mnist", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)
train_set = Subset(mnist_train, range(2000))
test_set = Subset(mnist_test, range(500))
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)"""
        )
    )
    cells += exercise_header(
        "3.1",
        "Define the MLP",
        "Implement `MLP` with one hidden layer and ReLU activations.",
    )
    cells.append(
        code(
            """class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # YOUR CODE HERE
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
print(model)"""
        )
    )
    cells += exercise_header(
        "3.2",
        "Training loop",
        "Implement one epoch of SGD with cross-entropy loss.",
    )
    cells.append(
        code(
            """def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # YOUR CODE HERE
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
history = []
for epoch in range(3):
    loss, acc = train_epoch(model, train_loader, optimizer)
    history.append((loss, acc))
    print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.3f}")"""
        )
    )
    cells += exercise_header(
        "3.3",
        "Evaluation",
        "Compute test accuracy.",
    )
    cells.append(
        code(
            """@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # YOUR CODE HERE
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total

print("Test accuracy:", evaluate(model, test_loader))"""
        )
    )
    cells += exercise_header(
        "3.4",
        "Loss curve",
        "Plot training loss per epoch.",
    )
    cells.append(
        code(
            """losses = [h[0] for h in history]
plt.plot(losses, marker="o")
plt.xlabel("epoch")
plt.ylabel("train loss")
plt.title("MNIST MLP training")
plt.show()"""
        )
    )
    cells.append(reflection_cell(3, "deep neural networks and MNIST"))
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


# ---------------------------------------------------------------------------
# Day 4 — Fashion-MNIST CNN
# ---------------------------------------------------------------------------
def build_day04() -> nbformat.NotebookNode:
    cells = week2_intro(
        4,
        "Convolutional Networks — Fashion-MNIST",
        "Learn spatial feature extraction with a small CNN.",
    )
    cells.append(torch_setup())
    cells.append(
        md(
            "A 2-D convolution applies a learned filter across spatial locations: "
            "$(f * k)[i,j] = \\sum_{u,v} f[i+u,j+v]\\, k[u,v]$. "
            "Stacking conv–ReLU–pool blocks builds translation-equivariant representations."
        )
    )
    cells.append(
        code(
            """from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])
fashion_train = datasets.FashionMNIST("/tmp/fashion", train=True, download=True, transform=transform)
fashion_test = datasets.FashionMNIST("/tmp/fashion", train=False, download=True, transform=transform)
train_loader = DataLoader(Subset(fashion_train, range(3000)), batch_size=64, shuffle=True)
test_loader = DataLoader(Subset(fashion_test, range(1000)), batch_size=256)"""
        )
    )
    cells += exercise_header("4.1", "Small CNN", "Build `TinyCNN` with two conv blocks.")
    cells.append(
        code(
            """class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # YOUR CODE HERE
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = TinyCNN().to(device)"""
        )
    )
    cells += exercise_header("4.2", "Train for 2 epochs", "Use Adam and cross-entropy.")
    cells.append(
        code(
            """def run_training(model, loader, optimizer, epochs=2):
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # YOUR CODE HERE
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"Epoch {epoch+1} avg loss: {running/len(loader):.4f}")

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
run_training(model, train_loader, opt)"""
        )
    )
    cells += exercise_header("4.3", "Visualize filters", "Plot the first-layer conv filters.")
    cells.append(
        code(
            """w = model.features[0].weight.detach().cpu()
fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for i, ax in enumerate(axes.flat):
    if i < w.size(0):
        ax.imshow(w[i, 0], cmap="gray")
    ax.axis("off")
plt.suptitle("First conv layer filters")
plt.tight_layout()
plt.show()"""
        )
    )
    cells += exercise_header("4.4", "Test accuracy", "Evaluate on the held-out subset.")
    cells.append(
        code(
            """@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    ok, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ok += (model(x).argmax(1) == y).sum().item()
        n += y.size(0)
    return ok / n

print("Test accuracy:", accuracy(model, test_loader))"""
        )
    )
    cells.append(reflection_cell(4, "convolutional networks"))
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


# ---------------------------------------------------------------------------
# Day 5 — Tiny attention / RNN sentiment
# ---------------------------------------------------------------------------
def build_day05() -> nbformat.NotebookNode:
    cells = week2_intro(
        5,
        "Sequence Models — Tiny Sentiment Classifier",
        "Compare a bag-of-words baseline with a tiny RNN on a synthetic sentiment corpus.",
    )
    cells.append(torch_setup())
    cells.append(
        md(
            "Attention computes a weighted sum of value vectors: "
            "$\\mathrm{Attention}(Q,K,V) = \\mathrm{softmax}(QK^\\top / \\sqrt{d_k}) V$. "
            "RNNs maintain a hidden state $h_t = \\phi(W_x x_t + W_h h_{t-1})$."
        )
    )
    cells.append(
        code(
            """# Tiny sentiment corpus (label: 1=positive, 0=negative)
corpus = [
    ("i love this course", 1),
    ("great lecture today", 1),
    ("amazing explanation", 1),
    ("wonderful material", 1),
    ("i hate this topic", 0),
    ("bad lecture today", 0),
    ("terrible explanation", 0),
    ("awful material", 0),
    ("not bad at all", 1),
    ("not great really", 0),
]
vocab = sorted(set(w for s, _ in corpus for w in s.split()))
word2idx = {w: i + 1 for i, w in enumerate(vocab)}  # 0 = pad
print("Vocab size:", len(vocab) + 1)"""
        )
    )
    cells += exercise_header("5.1", "Encode sequences", "Convert sentences to padded index tensors.")
    cells.append(
        code(
            """def encode(sentence, max_len=6):
    # YOUR CODE HERE
    ids = [word2idx.get(w, 0) for w in sentence.split()]
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

X = torch.stack([encode(s) for s, _ in corpus])
y = torch.tensor([lbl for _, lbl in corpus], dtype=torch.long)
print(X.shape, y.shape)"""
        )
    )
    cells += exercise_header("5.2", "Tiny RNN classifier", "Implement `TinyRNN` with `nn.Embedding` + `nn.GRU`.")
    cells.append(
        code(
            """class TinyRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden=32):
        super().__init__()
        # YOUR CODE HERE
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        emb = self.embed(x)
        _, h = self.gru(emb)
        return self.fc(h[-1])

model = TinyRNN(len(word2idx) + 1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
for step in range(100):
    model.train()
    opt.zero_grad()
    loss = F.cross_entropy(model(X), y)
    loss.backward()
    opt.step()
print("Final loss:", loss.item())"""
        )
    )
    cells += exercise_header("5.3", "Scaled dot-product attention", "Implement attention over GRU outputs.")
    cells.append(
        code(
            """def scaled_dot_product_attention(Q, K, V):
    # YOUR CODE HERE — shapes (batch, seq, dim)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

with torch.no_grad():
    emb = model.embed(X)
    out, _ = model.gru(emb)
    attn_out, w = scaled_dot_product_attention(out, out, out)
print("Attention output shape:", attn_out.shape)"""
        )
    )
    cells += exercise_header("5.4", "Predictions", "Show predicted labels vs ground truth.")
    cells.append(
        code(
            """@torch.no_grad()
def predict(model, X):
    return model(X).argmax(1)

preds = predict(model, X)
for (sent, lbl), p in zip(corpus, preds.tolist()):
    print(f"{sent!r:30s} true={lbl} pred={p}")"""
        )
    )
    cells.append(reflection_cell(5, "RNNs and attention"))
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


# ---------------------------------------------------------------------------
# Day 6 — KL / ELBO + 1D VAE
# ---------------------------------------------------------------------------
def build_day06() -> nbformat.NotebookNode:
    cells = week2_intro(
        6,
        "Generative Modeling — 1D VAE",
        "Implement the ELBO and train a variational autoencoder on 1-D synthetic data.",
    )
    cells.append(torch_setup())
    cells.append(
        md(
            "The ELBO for a VAE is "
            "$\\mathcal{L} = \\mathbb{E}_{q_\\phi(z|x)}[\\log p_\\theta(x|z)] "
            "- D_{\\mathrm{KL}}(q_\\phi(z|x) \\| p(z))$. "
            "For Gaussian $q=\\mathcal{N}(\\mu,\\mathrm{diag}(\\sigma^2))$ and "
            "$p=\\mathcal{N}(0,I)$, "
            "$D_{\\mathrm{KL}} = \\tfrac{1}{2}\\sum_j (\\mu_j^2 + \\sigma_j^2 - \\log\\sigma_j^2 - 1)$."
        )
    )
    cells.append(
        code(
            """# 1-D mixture of Gaussians as "data"
data = torch.cat([
    torch.randn(500) * 0.3 + 2.0,
    torch.randn(500) * 0.3 - 2.0,
]).unsqueeze(1)
loader = DataLoader(data, batch_size=64, shuffle=True)"""
        )
    )
    cells += exercise_header("6.1", "KL divergence", "Implement `kl_gaussian` for diagonal Gaussians vs standard normal.")
    cells.append(
        code(
            """def kl_gaussian(mu, logvar):
    # YOUR CODE HERE
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

mu = torch.randn(8, 4)
logvar = torch.randn(8, 4)
print("KL (sample):", kl_gaussian(mu, logvar).item())"""
        )
    )
    cells += exercise_header("6.2", "VAE architecture", "Build encoder and decoder MLPs.")
    cells.append(
        code(
            """class VAE1D(nn.Module):
    def __init__(self, latent=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.mu = nn.Linear(32, latent)
        self.logvar = nn.Linear(32, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, 1))

    def reparameterize(self, mu, logvar):
        # YOUR CODE HERE
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

vae = VAE1D().to(device)"""
        )
    )
    cells += exercise_header("6.3", "ELBO loss", "Combine reconstruction MSE and KL term.")
    cells.append(
        code(
            """def elbo_loss(recon, x, mu, logvar):
    # YOUR CODE HERE
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl = kl_gaussian(mu, logvar)
    return recon_loss + kl, recon_loss.item(), kl.item()

opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
for epoch in range(20):
    vae.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        recon, mu, logvar = vae(batch)
        loss, _, _ = elbo_loss(recon, batch, mu, logvar)
        loss.backward()
        opt.step()
        total += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: ELBO loss {total/len(loader):.4f}")"""
        )
    )
    cells += exercise_header("6.4", "Sample from the prior", "Decode $z \\sim \\mathcal{N}(0,I)$ and histogram.")
    cells.append(
        code(
            """@torch.no_grad()
def sample_vae(model, n=500):
    z = torch.randn(n, 2)
    return model.decoder(z).squeeze().numpy()

samples = sample_vae(vae)
plt.hist(data.numpy().ravel(), bins=40, alpha=0.5, label="data", density=True)
plt.hist(samples, bins=40, alpha=0.5, label="VAE samples", density=True)
plt.legend()
plt.title("1D VAE: data vs samples")
plt.show()"""
        )
    )
    cells.append(reflection_cell(6, "VAEs and the ELBO"))
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


# ---------------------------------------------------------------------------
# Day 7 — Flow matching 1D
# ---------------------------------------------------------------------------
def build_day07() -> nbformat.NotebookNode:
    cells = week2_intro(
        7,
        "Flow Matching in 1D",
        "Learn a time-dependent vector field that transports noise to data (inspired by MIT Lab Two).",
    )
    cells.append(torch_setup())
    cells.append(
        md(
            "Conditional flow matching uses paths $x_t = (1-t) x_0 + t x_1$ with "
            "target velocity $u_t(x_t|x_1) = x_1 - x_0$. "
            "We regress a network $v_\\theta(t, x_t)$ onto this target."
        )
    )
    cells.append(
        code(
            """class Sampleable(ABC):
    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        pass

class GaussianMixture1D(Sampleable):
    def __init__(self, means, stds, weights):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.multinomial(self.weights, n, replacement=True)
        return self.means[idx] + self.stds[idx] * torch.randn(n)

data_dist = GaussianMixture1D([[-2.0], [2.0]], [0.3, 0.3], [0.5, 0.5])
noise_dist = GaussianMixture1D([[0.0]], [1.0], [1.0])"""
        )
    )
    cells += exercise_header("7.1", "Conditional path", "Sample $(x_0, x_1, t)$ and compute $x_t$ and target velocity.")
    cells.append(
        code(
            """def sample_conditional_path(n, t=None):
    # YOUR CODE HERE
    x0 = noise_dist.sample(n).unsqueeze(1)
    x1 = data_dist.sample(n).unsqueeze(1)
    if t is None:
        t = torch.rand(n, 1)
    xt = (1 - t) * x0 + t * x1
    velocity = x1 - x0
    return x0, x1, t, xt, velocity

x0, x1, t, xt, v = sample_conditional_path(8)
print("xt shape:", xt.shape, "velocity shape:", v.shape)"""
        )
    )
    cells += exercise_header("7.2", "Velocity network", "MLP taking concatenated $(t, x_t)$.")
    cells.append(
        code(
            """class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        # YOUR CODE HERE
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, t, xt):
        inp = torch.cat([t, xt], dim=-1)
        return self.net(inp)

vel_net = VelocityNet().to(device)
opt = torch.optim.Adam(vel_net.parameters(), lr=1e-3)"""
        )
    )
    cells += exercise_header("7.3", "Train flow matching", "Minimize MSE between predicted and target velocity.")
    cells.append(
        code(
            """for step in range(300):
    vel_net.train()
    opt.zero_grad()
    _, _, t, xt, target_v = sample_conditional_path(256)
    t, xt, target_v = t.to(device), xt.to(device), target_v.to(device)
    # YOUR CODE HERE
    pred_v = vel_net(t, xt)
    loss = F.mse_loss(pred_v, target_v)
    loss.backward()
    opt.step()
    if (step + 1) % 100 == 0:
        print(f"Step {step+1}: loss={loss.item():.4f}")"""
        )
    )
    cells += exercise_header("7.4", "Integrate ODE from noise", "Euler steps from $t=0$ to $t=1$.")
    cells.append(
        code(
            """@torch.no_grad()
def integrate_flow(model, x_init, steps=50):
    x = x_init.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.size(0), 1), i * dt)
        # YOUR CODE HERE
        x = x + dt * model(t, x)
    return x

x_init = noise_dist.sample(1000).unsqueeze(1)
generated = integrate_flow(vel_net, x_init).squeeze().numpy()
data_samples = data_dist.sample(1000).numpy()

plt.hist(data_samples, bins=40, alpha=0.5, label="data", density=True)
plt.hist(generated, bins=40, alpha=0.5, label="flow samples", density=True)
plt.legend()
plt.title("1D flow matching")
plt.show()"""
        )
    )
    cells.append(reflection_cell(7, "flow matching"))
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


# ---------------------------------------------------------------------------
# Day 8 — SDE vs ODE 2D (simplified lab_one)
# ---------------------------------------------------------------------------
def build_day08() -> nbformat.NotebookNode:
    cells = week2_intro(
        8,
        "SDE vs ODE Simulation in 2D",
        "Compare deterministic Euler integration with Euler–Maruyama (simplified MIT Lab One).",
    )
    cells.append(torch_setup())
    cells.append(
        md(
            "An ODE is $dX_t = u_t(X_t)\\,dt$. An SDE adds noise: "
            "$dX_t = u_t(X_t)\\,dt + \\sigma_t\\, dW_t$. "
            "Euler: $X_{t+h} = X_t + h\\, u_t(X_t)$. "
            "Euler–Maruyama: $X_{t+h} = X_t + h\\, u_t(X_t) + \\sqrt{h}\\,\\sigma_t z$."
        )
    )
    cells.append(
        code(
            """class ODE(ABC):
    @abstractmethod
    def drift(self, x, t):
        pass

class SDE(ABC):
    @abstractmethod
    def drift(self, x, t):
        pass

    @abstractmethod
    def diffusion(self, x, t):
        pass

class RotatingODE(ODE):
    def drift(self, x, t):
        # YOUR CODE HERE — rotate velocity field
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return torch.cat([-x2, x1], dim=1)

class RotatingSDE(SDE):
    def drift(self, x, t):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return torch.cat([-x2, x1], dim=1)

    def diffusion(self, x, t):
        # YOUR CODE HERE
        return 0.3 * torch.ones_like(x)"""
        )
    )
    cells += exercise_header("8.1", "Euler simulator", "Implement `euler_step` for an ODE.")
    cells.append(
        code(
            """def euler_simulate(ode, x0, steps=100):
    xs = [x0.clone()]
    x = x0.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.tensor(i * dt)
        # YOUR CODE HERE
        x = x + dt * ode.drift(x, t)
        xs.append(x.clone())
    return torch.stack(xs, dim=0)

x0 = torch.tensor([[2.0, 0.0]])
traj = euler_simulate(RotatingODE(), x0)
plt.plot(traj[:, 0, 0].numpy(), traj[:, 0, 1].numpy())
plt.title("ODE trajectory (rotation)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.show()"""
        )
    )
    cells += exercise_header("8.2", "Euler–Maruyama", "Implement stochastic integration.")
    cells.append(
        code(
            """def euler_maruyama_simulate(sde, x0, steps=100, n_paths=200):
    dt = 1.0 / steps
    x = x0.repeat(n_paths, 1)
    for i in range(steps):
        t = torch.tensor(i * dt)
        # YOUR CODE HERE
        z = torch.randn_like(x)
        x = x + dt * sde.drift(x, t) + (dt ** 0.5) * sde.diffusion(x, t) * z
    return x

final = euler_maruyama_simulate(RotatingSDE(), torch.zeros(1, 2), steps=200, n_paths=500)
plt.scatter(final[:, 0].numpy(), final[:, 1].numpy(), s=8, alpha=0.5)
plt.title("SDE endpoints (cloud due to noise)")
plt.axis("equal")
plt.show()"""
        )
    )
    cells += exercise_header("8.3", "Compare spread", "Print std of final positions for ODE vs SDE.")
    cells.append(
        code(
            """ode_final = euler_simulate(RotatingODE(), torch.zeros(1, 2), steps=200)[-1]
sde_final = euler_maruyama_simulate(RotatingSDE(), torch.zeros(1, 2), steps=200, n_paths=500)
print("ODE final (single path):", ode_final.numpy())
print("SDE final std:", sde_final.std(dim=0).numpy())"""
        )
    )
    cells += exercise_header("8.4", "Vector field visualization", "Plot drift arrows on a grid.")
    cells.append(
        code(
            """grid = torch.linspace(-2, 2, 15)
X, Y = torch.meshgrid(grid, grid, indexing="xy")
pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
ode = RotatingODE()
v = ode.drift(pts, torch.tensor(0.0))
plt.quiver(pts[:, 0], pts[:, 1], v[:, 0], v[:, 1], alpha=0.7)
plt.title("Drift field")
plt.axis("equal")
plt.show()"""
        )
    )
    cells.append(reflection_cell(8, "ODEs vs SDEs"))
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


# ---------------------------------------------------------------------------
# Day 9 — Tiny char-level GPT
# ---------------------------------------------------------------------------
def build_day09() -> nbformat.NotebookNode:
    cells = week2_intro(
        9,
        "Autoregressive Models — Tiny Char-Level GPT",
        "Train a small transformer on a short text snippet.",
    )
    cells.append(torch_setup())
    cells.append(
        md(
            "Autoregressive models factorize $p(x) = \\prod_t p(x_t \\mid x_{<t})$. "
            "A causal transformer masks future positions so each token only attends to the past."
        )
    )
    cells.append(
        code(
            """text = "to be or not to be that is the question "
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
print("Corpus length:", len(data), "Vocab:", len(chars))"""
        )
    )
    cells += exercise_header("9.1", "Build training sequences", "Sliding window of length `block_size`.")
    cells.append(
        code(
            """block_size = 16

def get_batch(data, block_size, batch_size=32):
    # YOUR CODE HERE
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch(data, block_size)
print(xb.shape, yb.shape)"""
        )
    )
    cells += exercise_header("9.2", "Tiny GPT", "Causal self-attention block + LM head.")
    cells.append(
        code(
            """class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, n_layers=2, block_size=16):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        x = self.tr(x, mask=mask)
        return self.lm_head(x)

model = TinyGPT(len(chars), block_size=block_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)"""
        )
    )
    cells += exercise_header("9.3", "Train", "Minimize next-token cross-entropy.")
    cells.append(
        code(
            """for step in range(200):
    model.train()
    xb, yb = get_batch(data, block_size)
    xb, yb = xb.to(device), yb.to(device)
    # YOUR CODE HERE
    opt.zero_grad()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, len(chars)), yb.view(-1))
    loss.backward()
    opt.step()
    if (step + 1) % 50 == 0:
        print(f"Step {step+1}: loss={loss.item():.3f}")"""
        )
    )
    cells += exercise_header("9.4", "Generate", "Sample characters autoregressively.")
    cells.append(
        code(
            """@torch.no_grad()
def generate(model, prompt="t", max_new=40):
    idx = torch.tensor([[stoi[c] for c in prompt]], device=device)
    for _ in range(max_new):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)
    return "".join(itos[i.item()] for i in idx[0])

print(generate(model, "to"))"""
        )
    )
    cells.append(reflection_cell(9, "autoregressive language modeling"))
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


# ---------------------------------------------------------------------------
# Day 10 — KV cache timing
# ---------------------------------------------------------------------------
def build_day10() -> nbformat.NotebookNode:
    cells = week2_intro(
        10,
        "Efficient Inference — KV Cache Timing",
        "Measure latency with and without key/value caching during autoregressive decoding.",
    )
    cells.append(torch_setup())
    cells.append(
        md(
            "During decoding, past keys and values can be cached so each new token only "
            "computes attention for the latest query. Without caching, cost grows "
            "$O(T^2)$ per step; with caching, $O(T)$ per step."
        )
    )
    cells.append(
        code(
            """class TinyDecoder(nn.Module):
    def __init__(self, vocab=64, d_model=64, nhead=4, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(512, d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, tgt, memory, tgt_mask=None, past_kv=None):
        B, T = tgt.shape
        pos = torch.arange(T, device=tgt.device)
        x = self.emb(tgt) + self.pos(pos)
        out = self.decoder(x, memory, tgt_mask=tgt_mask)
        return self.head(out)

vocab_size = 64
model = TinyDecoder(vocab=vocab_size).to(device)
memory = torch.randn(1, 8, 64)  # fake encoder output"""
        )
    )
    cells += exercise_header("10.1", "Naive autoregressive loop", "Re-run full sequence each step.")
    cells.append(
        code(
            """import time

@torch.no_grad()
def decode_naive(model, memory, prompt_len=4, gen_len=32):
    seq = torch.randint(0, vocab_size, (1, prompt_len), device=device)
    t0 = time.perf_counter()
    for _ in range(gen_len):
        T = seq.size(1)
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        logits = model(seq, memory, tgt_mask=mask)
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_tok], dim=1)
    return time.perf_counter() - t0

naive_time = decode_naive(model, memory)
print(f"Naive decode time: {naive_time*1000:.2f} ms")"""
        )
    )
    cells += exercise_header("10.2", "Manual KV cache", "Store keys/values and append one token at a time.")
    cells.append(
        code(
            """class CachedDecoder(nn.Module):
    \"\"\"Wrapper that caches projected K/V per layer (simplified).\"\"\"
    def __init__(self, base: TinyDecoder):
        super().__init__()
        self.base = base

    @torch.no_grad()
    def decode_step(self, token, memory, cache=None):
        # YOUR CODE HERE — single-token forward; cache stores full tgt embeddings so far
        if cache is None:
            cache = token
        else:
            cache = torch.cat([cache, token], dim=1)
        T = cache.size(1)
        pos = torch.arange(T, device=token.device)
        x = self.base.emb(cache) + self.base.pos(pos)
        mask = torch.triu(torch.ones(T, T, device=token.device), diagonal=1).bool()
        out = self.base.decoder(x, memory, tgt_mask=mask)
        logits = self.base.head(out[:, -1:])
        return logits, cache

cached = CachedDecoder(model)

@torch.no_grad()
def decode_cached(model, memory, prompt_len=4, gen_len=32):
    seq = torch.randint(0, vocab_size, (1, prompt_len), device=device)
    cache = None
    t0 = time.perf_counter()
    for _ in range(gen_len):
        logits, cache = model.decode_step(seq[:, -1:], memory, cache)
        next_tok = logits.argmax(dim=-1)
        seq = torch.cat([seq, next_tok], dim=1)
    return time.perf_counter() - t0

cached_time = decode_cached(cached, memory)
print(f"Cached decode time: {cached_time*1000:.2f} ms")"""
        )
    )
    cells += exercise_header("10.3", "Benchmark sweep", "Compare times across generation lengths.")
    cells.append(
        code(
            """lengths = [8, 16, 32, 64]
naive_times, cached_times = [], []
for g in lengths:
    # YOUR CODE HERE
    naive_times.append(decode_naive(model, memory, gen_len=g))
    cached_times.append(decode_cached(cached, memory, gen_len=g))

plt.plot(lengths, naive_times, marker="o", label="naive")
plt.plot(lengths, cached_times, marker="o", label="cached")
plt.xlabel("tokens generated")
plt.ylabel("seconds")
plt.legend()
plt.title("KV cache timing (simplified)")
plt.show()
print(f"Speedup at 64 tokens: {naive_times[-1]/cached_times[-1]:.2f}x")"""
        )
    )
    cells += exercise_header("10.4", "Discuss complexity", "Fill in the big-O comparison below.")
    cells.append(
        md(
            "**Without cache**: each step recomputes attention over all $T$ tokens → "
            "$\\sum_{T} O(T^2) = O(G^3)$ for $G$ generated tokens.\n\n"
            "**With cache**: each step is $O(T)$ → $\\sum_T O(T) = O(G^2)$.\n\n"
            "Modern LLMs also use FlashAttention and fused kernels; caching is necessary but not sufficient."
        )
    )
    cells.append(reflection_cell(10, "efficient autoregressive inference"))
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


BUILDERS = [
    build_day01,
    build_day02,
    build_day03,
    build_day04,
    build_day05,
    build_day06,
    build_day07,
    build_day08,
    build_day09,
    build_day10,
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, builder in enumerate(BUILDERS, start=1):
        nb = builder()
        path = OUT_DIR / f"day{i:02d}.ipynb"
        with path.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
