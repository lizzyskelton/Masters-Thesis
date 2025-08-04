import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Multi-year', 'March', 'May', 'August']
ci = [4.92, 2.75, 2.56, 1.8]
sipl = [3.74, 2.98, 2.19, 0.93]

x = np.arange(len(categories))

plt.figure(figsize=(8, 5))
plt.plot(x, ci, marker='o', label='CI', color='#8EDDFC')
plt.plot(x, sipl, marker='o', label='SIPL', color='darkorange')

# Add data labels
for i, val in enumerate(ci):
    plt.text(x[i]+0.1, ci[i], f"{ci[i]:.2f}", color='#8EDDFC', va='bottom', fontsize=11)
for i, val in enumerate(sipl):
    plt.text(x[i]+0.1, sipl[i], f"{sipl[i]:.2f}", color='darkorange', va='bottom', fontsize=11)

plt.xticks(x, categories, fontsize=12)
plt.ylabel('Ice Thickness (m)', fontsize=14)
plt.legend(fontsize=12)
plt.ylim(0)
plt.tight_layout()

# Remove right and top border
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()