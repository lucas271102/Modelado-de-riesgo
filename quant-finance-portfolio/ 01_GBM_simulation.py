import numpy as np
import matplotlib.pyplot as plt

# Parámetros
S0 = 100      # precio inicial
mu = 0.10     # retorno esperado anual
sigma = 0.20  # volatilidad anual
T = 1         # horizonte en años
N = 252       # pasos (días hábiles)
paths = 1000  # trayectorias

dt = T / N
t = np.linspace(0, T, N)

# Simulación GBM
np.random.seed(42)
W = np.random.randn(paths, N)
increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
S = S0 * np.exp(np.cumsum(increments, axis=1))

# Gráfico
plt.figure(figsize=(12, 5))
plt.plot(t, S[:50].T, alpha=0.3, linewidth=0.8, color='steelblue')
plt.plot(t, S0 * np.exp(mu * t), color='red', linewidth=2, label='Valor esperado')
plt.title('Geometric Brownian Motion — 1000 trayectorias')
plt.xlabel('Tiempo (años)')
plt.ylabel('Precio')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Precio medio final: ${S[:, -1].mean():.2f}")
print(f"Desviación estándar: ${S[:, -1].std():.2f}")
print(f"Precio mínimo: ${S[:, -1].min():.2f}")
print(f"Precio máximo: ${S[:, -1].max():.2f}")