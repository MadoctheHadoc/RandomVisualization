import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams["font.family"] = "serif"

colors = {
    "Value Iteration": "#AA2222",
    "SARSA": "#56A520",
    "Monte Carlo On-Policy": "#2B46CE",
}

markers = {
    "Value Iteration": "s",
    "SARSA": "^",
    "Monte Carlo On-Policy": "o",
}

algos = ["Value Iteration", "SARSA", "Monte Carlo On-Policy"]

data = [
    # (algo, performance, convergence_episodes)
    # --- A1 Grid ---
    ("Value Iteration", 1.1929, 18.0),
    ("SARSA", 1.2214, 221.0),
    ("Monte Carlo On-Policy", 2.2536, 2354.0),
    ("Value Iteration", 1.3857, 18.0),
    ("SARSA", 1.4500, 175.0),
    ("Monte Carlo On-Policy", 2.0607, 1848.0),
    ("Value Iteration", 1.1107, 18.0),
    ("SARSA", 1.1071, 188.0),
    ("Monte Carlo On-Policy", 2.2179, 1358.0),
    ("Value Iteration", 1.1107, 18.0),
    ("SARSA", 1.1143, 213.0),
    ("Monte Carlo On-Policy", 2.9821, 219.0),
    ("Value Iteration", 1.1107, 18.0),
    ("SARSA", 1.1143, 176.0),
    ("Monte Carlo On-Policy", 1.3000, 601.0),
    ("Value Iteration", 1.1107, 17.0),
    ("SARSA", 1.2143, 168.0),
    ("Monte Carlo On-Policy", 2.0393, 693.0),
    ("Value Iteration", 1.1107, 18.0),
    ("SARSA", 1.2464, 757.0),
    ("Value Iteration", 1.0000, 18.0),
    ("SARSA", 1.0000, 156.0),
    ("Value Iteration", 1.1107, 19.0),
    ("SARSA", 1.1179, 184.0),
    ("Monte Carlo On-Policy", 1.5786, 4254.0),
    ("Value Iteration", 1.1107, 18.0),
    ("SARSA", 1.1321, 166.0),
    ("Monte Carlo On-Policy", 1.6929, 727.0),
    ("Value Iteration", 1.1107, 18.0),
    ("Value Iteration", 1.1107, 18.0),
    ("SARSA", 1.2071, 205.0),
    ("Monte Carlo On-Policy", 1.3357, 1034.0),
    ("Value Iteration", 1.1107, 18.0),
    ("SARSA", 1.1071, 143.0),
    ("Monte Carlo On-Policy", 1.8179, 939.0),
    ("Value Iteration", 1.1107, 18.0),
    ("SARSA", 1.1107, 186.0),
    ("Monte Carlo On-Policy", 1.9250, 429.0),
    ("Value Iteration", 1.1107, 18.0),
    ("SARSA", 1.1143, 157.0),
    ("Monte Carlo On-Policy", 1.3929, 213.0),
    ("Value Iteration", 1.1107, 18.0),
    ("SARSA", 1.3607, 355.0),
    ("Monte Carlo On-Policy", 1.8714, 1429.0),
    ("Value Iteration", 1.1107, 17.0),
    ("SARSA", 1.1357, 173.0),
    ("Monte Carlo On-Policy", 2.3036, 4962.0),
    ("Value Iteration", 1.0921, 18.0),
    ("SARSA", 1.0986, 178.6),
    ("Monte Carlo On-Policy", 1.6032, 2190.7),
    ("Value Iteration", 1.1107, 18.0),
    ("SARSA", 1.1179, 218.0),
    ("Value Iteration", 1.1107, 16.0),
    ("SARSA", 1.1143, 183.0),
    ("Monte Carlo On-Policy", 3.5304, 10000.0),
    ("Value Iteration", 1.1107, 18.0),
    ("Monte Carlo On-Policy", 1.9750, 2474.0),
    # --- Large Grid ---
    ("Value Iteration", 1.0000, 18.0),
    ("SARSA", 1.0000, 132.0),
    ("Monte Carlo On-Policy", 1.0524, 203.0),
    ("Value Iteration", 1.1190, 18.0),
    ("SARSA", 1.1238, 136.0),
    ("Monte Carlo On-Policy", 1.5476, 246.0),
    ("Value Iteration", 1.2048, 18.0),
    ("SARSA", 1.2000, 136.0),
    ("Monte Carlo On-Policy", 1.5000, 274.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.0619, 150.0),
    ("Monte Carlo On-Policy", 1.2714, 162.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.0619, 217.0),
    ("Monte Carlo On-Policy", 1.7143, 175.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.0619, 150.0),
    ("Monte Carlo On-Policy", 1.1238, 860.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.0619, 256.0),
    ("Monte Carlo On-Policy", 1.2619, 189.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.1381, 137.0),
    ("Monte Carlo On-Policy", 1.8857, 133.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.0619, 144.0),
    ("Monte Carlo On-Policy", 1.4095, 239.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.0619, 148.0),
    ("Monte Carlo On-Policy", 1.1333, 880.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 8.4262, 692.0),
    ("Monte Carlo On-Policy", 1.1190, 207.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.0619, 136.0),
    ("Monte Carlo On-Policy", 1.5143, 271.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.0619, 177.0),
    ("Monte Carlo On-Policy", 1.5143, 406.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.2476, 222.0),
    ("Monte Carlo On-Policy", 1.2524, 125.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.2429, 221.0),
    ("Monte Carlo On-Policy", 1.1238, 175.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.0762, 149.0),
    ("Monte Carlo On-Policy", 1.1238, 341.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.1952, 148.0),
    ("Monte Carlo On-Policy", 1.1619, 121.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.0619, 140.0),
    ("Monte Carlo On-Policy", 1.3238, 314.0),
    ("Value Iteration", 1.0590, 18.0),
    ("SARSA", 1.0738, 145.4),
    ("Monte Carlo On-Policy", 1.3776, 507.1),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 1.0619, 137.0),
    ("Monte Carlo On-Policy", 1.7238, 10000.0),
    ("Value Iteration", 1.0619, 18.0),
    ("SARSA", 4.8286, 10000.0),
    # --- Super Hard ---
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.1276, 912.0),
    ("Monte Carlo On-Policy", 2.6224, 1237.0),
    ("Value Iteration", 1.4397, 49.0),
    ("SARSA", 2.3776, 513.0),
    ("Monte Carlo On-Policy", 2.5569, 2203.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.5534, 852.0),
    ("Monte Carlo On-Policy", 1.9362, 4324.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.1724, 443.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.2638, 597.0),
    ("Value Iteration", 1.0000, 49.0),
    ("SARSA", 1.0345, 409.0),
    ("Value Iteration", 1.1293, 46.0),
    ("SARSA", 1.1690, 442.0),
    ("Value Iteration", 1.1293, 47.0),
    ("SARSA", 1.1828, 458.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.1672, 359.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.6793, 711.0),
    ("Value Iteration", 1.1293, 47.0),
    ("SARSA", 1.1172, 385.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.1586, 441.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 13.9069, 5385.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.1741, 459.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.1345, 542.0),
    ("Value Iteration", 1.2414, 49.0),
    ("SARSA", 1.5448, 446.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.1483, 496.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.1603, 471.0),
    ("Value Iteration", 1.1293, 49.0),
    ("SARSA", 1.1293, 398.0),
    ("Value Iteration", 1.1179, 49.0),
    ("SARSA", 1.3458, 444.4),
    ("SARSA", 1.4983, 608.0),
]

algo_data = {a: {"x": [], "y": []} for a in algos}
for algo, perf, conv in data:
    algo_data[algo]["x"].append(conv)
    algo_data[algo]["y"].append(perf)

fig, ax = plt.subplots(figsize=(7, 10))

# Plot scatter points for each algorithm
for algo in algos:
    ax.scatter(
        algo_data[algo]["x"],
        algo_data[algo]["y"],
        marker=markers[algo],
        color=colors[algo],
        s=70,
        alpha=0.7,
        label=algo,
        edgecolors="black",
        linewidths=0.0,
    )

# Aggregate all data for the combined trend line
all_x = []
all_y = []
for algo in algos:
    all_x.extend(algo_data[algo]["x"])
    all_y.extend(algo_data[algo]["y"])

# Compute the combined trend line in log-log space
log_x = np.log10(all_x)
log_y = np.log10(all_y)
coeffs = np.polyfit(log_x, log_y, 1)  # Linear fit in log-log space
trend_line = 10 ** np.polyval(coeffs, np.log10(np.sort(all_x)))

# Sort x for smooth trend line
x_sorted = np.sort(all_x)
ax.plot(
    x_sorted,
    trend_line,
    color="black",  # Use a distinct color for the combined trend line
    linestyle="-",
    alpha=1,
    linewidth=2,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Convergence (Episodes)", fontsize=18)
ax.set_ylabel("Optimality Ratio", fontsize=18)
ax.tick_params(axis="both", labelsize=15)
ax.grid(True, which="both", linestyle="--", alpha=1)

ax.legend(
    fontsize=18,
)

fig.suptitle(
    "Convergence Speed vs. Performance",
    fontsize=24,
)


plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("visualizations/ConvergenceVsPerformance.png", dpi=300, bbox_inches="tight")