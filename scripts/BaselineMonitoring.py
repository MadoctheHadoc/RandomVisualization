import matplotlib.pyplot as plt
import numpy as np

# Generate example data
np.random.seed(26)
months = np.arange(1, 25)
baseline_diagnoses = np.random.randint(5, 16, size=12)
normal_monitoring = np.random.randint(5, 16, size=10)
outbreak_monitoring = [18, 22]
monitoring_diagnoses = np.append(normal_monitoring, outbreak_monitoring)


reference_value = 17

diagnoses = np.concatenate([baseline_diagnoses, monitoring_diagnoses])

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(months[:12], baseline_diagnoses, 'o-', color='deepskyblue', alpha=1.0)
ax.plot(months[12:], monitoring_diagnoses, 'o-', color='mediumseagreen', alpha=1.0)

# Add highlight circles only when values exceed the reference value
exceeding_indices = months[12:][monitoring_diagnoses > reference_value]
exceeding_values = monitoring_diagnoses[monitoring_diagnoses > reference_value]
ax.scatter(exceeding_indices, exceeding_values, s=200, edgecolors='orangered', facecolors='none', linewidths=2)

# Add horizontal orange line at reference value
ax.axhline(y=reference_value, color='orangered', linestyle='-')

# Formatting
ax.axvline(x=12.5, color='gray', linestyle='--')
ax.text(3, 22, 'Baseline period', fontsize=14, color='deepskyblue', fontweight='bold')
ax.text(14, 22, 'Monitoring period', fontsize=14, color='mediumseagreen', fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Number of diagnoses')
ax.set_xticks(months[::2])
ax.set_ylim(0, 25)
plt.show()