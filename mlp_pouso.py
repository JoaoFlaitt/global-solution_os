"""
FIAP - Global Solution 2026/1 - Sistemas Operacionais
Prof. Dr. Jose Gomes Salim Neto
Tema: Industria aero-espacial

Quantificacao da influencia de padroes numericos (int8 vs float64)
em uma MLP embarcada para inferencia da distancia de parada (pouso)
em jatos comerciais com SO pre-emptivo de Kernel 64 bits.

Integrantes:
    - Miguel Leal           RM553009
    - Joao Victor Flaitt    RM553888
    - Lucas Bertolassi      RM553183
    - Lucca Calsolari       RM553678

Repositorio: https://github.com/JoaoFlaitt/global-solution_os
"""

from __future__ import annotations

import time
import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Reprodutibilidade
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 2. Geracao do dataset sintetico de pouso (5 atributos, ~6M linhas)
#    Modelo: equacoes do movimento variado nao uniformemente (jerk constante)
#            x(t) = x0 + v0 t + 1/2 a0 t^2 + 1/6 j t^3
#    A distancia de parada e estimada via integracao numerica (RK4) da EDO
#            m dV/dt = -1/2 rho V^2 S Cd - mu (m g - 1/2 rho V^2 S Cl)
#    A saida (target) e a distancia ate V=0.
# ---------------------------------------------------------------------------

@dataclass
class LandingPhysics:
    rho: float = 1.225        # densidade do ar (kg/m3) ao nivel do mar
    S: float = 124.6          # area da asa (m2) - referencia A320
    Cd: float = 0.085         # coeficiente de arrasto na corrida de pouso
    Cl: float = 0.65          # coeficiente de sustentacao residual
    g: float = 9.80665        # aceleracao da gravidade (m/s2)


def stop_distance(V0: float, mass: float, mu: float, slope_deg: float,
                  altitude: float, phys: LandingPhysics = LandingPhysics(),
                  dt: float = 0.05, t_max: float = 120.0) -> float:
    """Integra a EDO nao linear da corrida de pouso ate V=0 (RK4).

    Atributos de entrada do problema (5):
        V0          velocidade de toque (m/s)
        mass        massa de pouso (kg)
        mu          coef. de atrito da pista (-)
        slope_deg   gradiente da pista (graus)
        altitude    altitude do aerodromo (m)
    """
    # ajuste simplificado de densidade com altitude (modelo isotermico)
    rho = phys.rho * np.exp(-altitude / 8500.0)
    slope = np.deg2rad(slope_deg)

    def accel(V: float) -> float:
        drag = 0.5 * rho * V * V * phys.S * phys.Cd
        lift = 0.5 * rho * V * V * phys.S * phys.Cl
        # peso efetivo na pista (gradiente positivo => aclive => freia mais)
        N = mass * phys.g * np.cos(slope) - lift
        N = max(N, 0.0)
        # arrasto + atrito + componente do peso
        return -(drag + mu * N) / mass - phys.g * np.sin(slope)

    V = V0
    x = 0.0
    t = 0.0
    while V > 0.1 and t < t_max:
        k1 = accel(V)
        k2 = accel(V + 0.5 * dt * k1)
        k3 = accel(V + 0.5 * dt * k2)
        k4 = accel(V + dt * k3)
        a = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        V_new = V + a * dt
        x += 0.5 * (V + max(V_new, 0.0)) * dt
        V = V_new
        t += dt
    return x


def build_dataset(n: int = 6_000_000, save_path: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Gera um dataset (n, 5) e o vetor de targets (n,).

    Para n = 6e6 a integracao RK4 ponto-a-ponto e proibitiva. Por isso
    geramos um sub-conjunto (~50k pontos) com fisica completa e ajustamos
    uma superficie polinomial para popular o restante via amostragem
    + ruido Gaussiano controlado, mantendo o vinculo fisico.
    """
    # 1) sub-amostragem fisica
    n_phys = 50_000
    V0 = rng.uniform(55.0, 85.0, size=n_phys)            # m/s
    mass = rng.uniform(55_000.0, 78_000.0, size=n_phys)  # kg
    mu = rng.uniform(0.25, 0.55, size=n_phys)
    slope_deg = rng.uniform(-1.5, 1.5, size=n_phys)
    altitude = rng.uniform(0.0, 2500.0, size=n_phys)

    X_phys = np.column_stack([V0, mass, mu, slope_deg, altitude])
    y_phys = np.fromiter(
        (stop_distance(*row) for row in X_phys),
        dtype=np.float64,
        count=n_phys,
    )

    # 2) ajuste polinomial grau 2 (regressao OLS) para extrapolar p/ n linhas
    def design(X: np.ndarray) -> np.ndarray:
        v, m, u, s, h = X.T
        return np.column_stack([
            np.ones_like(v), v, m, u, s, h,
            v * v, m * m, u * u, s * s, h * h,
            v * m, v * u, v * s, v * h,
            m * u, m * s, m * h, u * s, u * h, s * h,
        ])

    A = design(X_phys)
    coef, *_ = np.linalg.lstsq(A, y_phys, rcond=None)

    # 3) amostragem completa (n linhas)
    V0 = rng.uniform(55.0, 85.0, size=n)
    mass = rng.uniform(55_000.0, 78_000.0, size=n)
    mu = rng.uniform(0.25, 0.55, size=n)
    slope_deg = rng.uniform(-1.5, 1.5, size=n)
    altitude = rng.uniform(0.0, 2500.0, size=n)
    X = np.column_stack([V0, mass, mu, slope_deg, altitude]).astype(np.float64)

    y = design(X) @ coef
    # ruido sensorial (~3% rel.) representando incertezas embarcadas
    y *= 1.0 + rng.normal(0.0, 0.03, size=n)
    y = np.clip(y, 250.0, 5_000.0)

    if save_path:
        np.savez_compressed(save_path, X=X, y=y)
    return X, y.astype(np.float64)


# ---------------------------------------------------------------------------
# 3. Pre-processamento
# ---------------------------------------------------------------------------

def preprocess(X: np.ndarray, y: np.ndarray):
    """Tratamento + normalizacao Min-Max para [0,1] (compativel com sigmoid)."""
    # tratamento de outliers (winsorize 1%)
    lo, hi = np.percentile(y, [1, 99])
    mask = (y >= lo) & (y <= hi)
    X, y = X[mask], y[mask]

    # split 70/15/15
    n = len(y)
    idx = rng.permutation(n)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    x_min = X[tr].min(axis=0)
    x_max = X[tr].max(axis=0)
    y_min = y[tr].min()
    y_max = y[tr].max()

    def norm_x(M): return (M - x_min) / (x_max - x_min + 1e-9)
    def norm_y(v): return (v - y_min) / (y_max - y_min + 1e-9)

    return (
        norm_x(X[tr]), norm_y(y[tr]),
        norm_x(X[va]), norm_y(y[va]),
        norm_x(X[te]), norm_y(y[te]),
        (y_min, y_max),
    )


# ---------------------------------------------------------------------------
# 4. MLP (1 camada oculta + 1 saida, sigmoide em ambas)
# ---------------------------------------------------------------------------

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def dsigmoid(a): return a * (1.0 - a)


@dataclass
class TrainHistory:
    epochs: list
    train_mse: list
    val_mse: list
    val_mae: list
    val_r2: list
    seconds: float


def train_mlp(Xtr, ytr, Xva, yva, dtype: np.dtype,
              hidden: int = 16, epochs: int = 200,
              batch: int = 4096, lr: float = 0.05) -> TrainHistory:
    """Treina a MLP em um padrao numerico especifico (int8 ou float64).

    Para int8, todas as entradas, pesos e ativacoes sao quantizados
    em 8 bits sem sinal (escala [0,1] -> [0,255]) a cada passo, simulando
    o pipeline de inferencia embarcada quantizada.
    """
    rng_local = np.random.default_rng(SEED)

    # pesos iniciais Xavier
    W1 = rng_local.normal(0, np.sqrt(1.0 / Xtr.shape[1]), (Xtr.shape[1], hidden))
    b1 = np.zeros(hidden)
    W2 = rng_local.normal(0, np.sqrt(1.0 / hidden), (hidden, 1))
    b2 = np.zeros(1)

    def quantize(a):
        if dtype == np.int8:
            q = np.clip(np.round(a * 127.0), -128, 127).astype(np.int8)
            return q.astype(np.float32) / 127.0
        return a.astype(dtype)

    history = TrainHistory([], [], [], [], [], 0.0)
    t0 = time.perf_counter()

    n = len(Xtr)
    for ep in range(1, epochs + 1):
        idx = rng_local.permutation(n)
        Xs = Xtr[idx]
        ys = ytr[idx]

        for i in range(0, n, batch):
            xb = quantize(Xs[i:i + batch])
            yb = ys[i:i + batch].reshape(-1, 1)

            # forward
            z1 = xb @ quantize(W1) + b1
            a1 = sigmoid(z1)
            a1q = quantize(a1)
            z2 = a1q @ quantize(W2) + b2
            a2 = sigmoid(z2)

            # backprop (loss = MSE)
            d2 = (a2 - yb) * dsigmoid(a2)
            d1 = (d2 @ W2.T) * dsigmoid(a1)

            W2 -= lr * a1q.T @ d2 / xb.shape[0]
            b2 -= lr * d2.mean(axis=0)
            W1 -= lr * xb.T @ d1 / xb.shape[0]
            b1 -= lr * d1.mean(axis=0)

        # avaliacao por epoca
        def predict(M):
            a1 = sigmoid(quantize(M) @ quantize(W1) + b1)
            return sigmoid(quantize(a1) @ quantize(W2) + b2).ravel()

        ptr = predict(Xtr)
        pva = predict(Xva)
        mse_tr = float(np.mean((ptr - ytr) ** 2))
        mse_va = float(np.mean((pva - yva) ** 2))
        mae_va = float(np.mean(np.abs(pva - yva)))
        ss_res = np.sum((pva - yva) ** 2)
        ss_tot = np.sum((yva - yva.mean()) ** 2) + 1e-12
        r2_va = float(1.0 - ss_res / ss_tot)

        history.epochs.append(ep)
        history.train_mse.append(mse_tr)
        history.val_mse.append(mse_va)
        history.val_mae.append(mae_va)
        history.val_r2.append(r2_va)

    history.seconds = time.perf_counter() - t0
    return history


# ---------------------------------------------------------------------------
# 5. Visualizacao
# ---------------------------------------------------------------------------

def plot_history(h64: TrainHistory, h8: TrainHistory, out_dir: str = "figs"):
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h64.epochs, h64.train_mse, label="float64 - treino")
    ax[0].plot(h64.epochs, h64.val_mse, label="float64 - val.")
    ax[0].plot(h8.epochs, h8.train_mse, label="int8 - treino", linestyle="--")
    ax[0].plot(h8.epochs, h8.val_mse, label="int8 - val.", linestyle="--")
    ax[0].set_xlabel("Epoca")
    ax[0].set_ylabel("MSE")
    ax[0].set_title("Curva de treinamento - MSE")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(h64.epochs, h64.val_r2, label="float64")
    ax[1].plot(h8.epochs, h8.val_r2, label="int8", linestyle="--")
    ax[1].set_xlabel("Epoca")
    ax[1].set_ylabel("R^2 (validacao)")
    ax[1].set_title("Qualidade do ajuste por epoca")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curvas_treinamento.png"), dpi=180)
    plt.close(fig)

    # tempo total
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.bar(["float64", "int8"], [h64.seconds / 60.0, h8.seconds / 60.0],
            color=["#1f77b4", "#d62728"])
    ax2.set_ylabel("Tempo de processamento (min)")
    ax2.set_title("Tempo total de treinamento")
    for i, v in enumerate([h64.seconds / 60.0, h8.seconds / 60.0]):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "tempo_total.png"), dpi=180)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# 6. Pipeline principal
# ---------------------------------------------------------------------------

def main():
    print(">> Gerando dataset sintetico (6M linhas, 5 atributos) ...")
    X, y = build_dataset(n=6_000_000)
    print(f"   X.shape={X.shape}  y.shape={y.shape}")

    print(">> Tratando e particionando ...")
    Xtr, ytr, Xva, yva, Xte, yte, scale = preprocess(X, y)
    print(f"   train={len(ytr):,}  val={len(yva):,}  test={len(yte):,}")

    print(">> Treinamento float64 ...")
    h64 = train_mlp(Xtr, ytr, Xva, yva, dtype=np.float64)
    print(f"   tempo: {h64.seconds/60:.2f} min  R2={h64.val_r2[-1]:.4f}")

    print(">> Treinamento int8 ...")
    h8 = train_mlp(Xtr, ytr, Xva, yva, dtype=np.int8)
    print(f"   tempo: {h8.seconds/60:.2f} min  R2={h8.val_r2[-1]:.4f}")

    plot_history(h64, h8)
    with open("history.json", "w", encoding="utf-8") as f:
        json.dump({"float64": asdict(h64), "int8": asdict(h8)}, f, indent=2)
    print(">> Concluido. Figuras em ./figs e historico em history.json")


if __name__ == "__main__":
    main()