import numpy as np
import matplotlib.pyplot as plt
from math import factorial, exp

lam = 4

# Exact computation
poisson_terms = [lam**k * exp(-lam) / factorial(k) for k in range(5)]
S = sum(poisson_terms)
M = 1.0 / (S + 2.0)

# PMF values
pmf = {}
for k in range(5):
    pmf[k] = M * lam**k * exp(-lam) / factorial(k)
pmf[5] = M
pmf[6] = M

# Expected value
E_X = sum(k * pmf[k] for k in range(7))

# Monte Carlo
np.random.seed(33)
values = np.array(list(range(7)))
probs = np.array([pmf[k] for k in range(7)])
probs = probs / probs.sum()
N_max = 10000
samples = np.random.choice(values, size=N_max, p=probs)
ns = np.arange(100, N_max + 1, 100)
estimates = [np.mean(samples[:n]) for n in ns]
# print(f"N\t |Estimate")
# for (n, e) in zip(ns, estimates):
#     print(f"{n}\t |{e}")
    
# print(f"Expected: {E_X}")

plt.figure(figsize=(10, 5))
plt.plot(ns, estimates, label="MC estimate")
plt.axhline(y=E_X, color='r', linestyle='--', label=f"Exact E[X] = {E_X:.4f}")
plt.xlabel("Number of simulations")
plt.ylabel("Estimated E[X]")
plt.title("Task 2a: Monte Carlo Convergence for E[X]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("visualizations/task2a_convergence.png", dpi=150)
plt.show()