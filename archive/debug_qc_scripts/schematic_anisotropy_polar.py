import numpy as np
import matplotlib.pyplot as plt

# --- angles for polar plots ---
theta = np.linspace(0, 2 * np.pi, 200)

# Cartoon single-grain anisotropy: slightly modulated circle
r_single = 1.0 + 0.03 * np.sin(3 * theta) + 0.02 * np.cos(5 * theta)

# Cartoon ensemble average profiles for FAPI / FAPI–TEMPO
r_fapi = 1.0 + 0.02 * np.sin(4 * theta)
r_tempo = 1.0 + 0.015 * np.sin(4 * theta + 0.3)

# Make figure with two polar panels (single grain + ensemble)
fig = plt.figure(figsize=(10, 4))

# (a) Single-grain polar profile (cartoon)
ax2 = fig.add_subplot(1, 2, 1, projection="polar")
ax2.plot(theta, np.ones_like(theta), "--", lw=1, label="Isotropic (ideal)")
ax2.plot(theta, r_single, lw=2, label="Example grain")
ax2.set_title("Single-grain polar profile\n(normalised radius)", pad=20)
ax2.set_rticks([0.9, 1.0, 1.1])
ax2.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))

# (b) Ensemble-averaged polar profiles (cartoon FAPI vs FAPI–TEMPO)
ax3 = fig.add_subplot(1, 2, 2, projection="polar")
ax3.plot(theta, r_fapi, lw=2, label="FAPI")
ax3.plot(theta, r_tempo, lw=2, label="FAPI–TEMPO")
ax3.set_title("Average normalised grain shape", pad=20)
ax3.set_rticks([0.9, 1.0, 1.1])
ax3.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))

plt.tight_layout()
plt.savefig("schematic_anisotropy_polar.png", dpi=300)
plt.close()

print("Saved schematic_anisotropy_polar.png")
