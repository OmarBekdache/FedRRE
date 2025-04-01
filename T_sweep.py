import matplotlib.pyplot as plt
import numpy as np

T = np.array([1,3,5,7])


ResNet18_AC = np.array([93.48, 93.89, 93.91, 93.82])
MobileNetV2_AC = np.array([92.56, 93.09, 92.98, 93.01])
VGG16_AC = np.array([92.20, 92.35, 92.44, 92.42])

ResNet18_AR = np.array([82.55, 81.97, 81.11, 80.44])
MobileNetV2_AR = np.array([80.28, 80.24, 79.27, 78.25])
VGG16_AR = np.array([80.14, 80.10, 79.50, 78.77])

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
#fig.suptitle("Model Performance vs T", fontsize=16)

# Plot Accuracy
l1, = axs[0].plot(T, ResNet18_AC, marker='o', label='ResNet18', color = "#4682B4")
l2, = axs[0].plot(T, MobileNetV2_AC, marker='o', label='MobileNetV2', color = "#B22222")
l3, = axs[0].plot(T, VGG16_AC, marker='o', label='VGG16', color = "#228B22")
#axs[0].set_title("Accuracy (AC) vs T")
axs[0].set_xlabel(r"$\tau = \tau_c = \tau_d$", fontsize = 24)
axs[0].set_ylabel(r"$\mathcal{A}_\text{cln}(\%)$", fontsize = 24)
axs[0].tick_params(axis='both', labelsize=20)
axs[0].grid(True)

# Plot Adversarial Robustness
axs[1].plot(T, ResNet18_AR, marker='o', color = "#4682B4")
axs[1].plot(T, MobileNetV2_AR, marker='o', color = "#B22222")
axs[1].plot(T, VGG16_AR, marker='o', color = "#228B22")
#axs[1].set_title("Adversarial Robustness (AR) vs T")
axs[1].set_xlabel(r"$\tau = \tau_c = \tau_d$", fontsize = 24)
axs[1].set_ylabel(r"$\mathcal{A}_\text{rob}(\%)$", fontsize = 24)
axs[1].tick_params(axis='both', labelsize=20)
axs[1].grid(True)

# Shared legend above plots
fig.legend([l1, l2, l3], ['ResNet-18', 'MobileNetV2', 'VGG-16'],
           loc='upper center', ncol=3, fontsize=16)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("T_sweep.pdf", dpi=300)
plt.show()

