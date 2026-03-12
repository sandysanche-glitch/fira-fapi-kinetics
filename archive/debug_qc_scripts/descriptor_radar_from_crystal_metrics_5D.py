#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use("Agg")  # optional: disable GUI / Tkinter
import matplotlib.pyplot as plt

# --- raw values (put your real numbers here) ---
labels = [
    r"Entropy $H$ (bits)",
    "Circularity distortion",
    r"CV(area)",
    r"Mean grain area ($\mu$m$^2$)",
    r"Defect fraction $\phi$",
]

fapi_raw  = np.array([0.79, 0.15, 0.35, 150.0, 0.12])
tempo_raw = np.array([0.72, 0.13, 0.48,  90.0, 0.05])

# --- normalize by per-descriptor mean ---
mean_vals = 0.5 * (fapi_raw + tempo_raw)
mean_vals = np.where(mean_vals == 0, 1.0, mean_vals)

fapi  = (fapi_raw  / mean_vals)
tempo = (tempo_raw / mean_vals)

fapi  = np.concatenate([fapi,  fapi[:1]])
tempo = np.concatenate([tempo, tempo[:1]])

num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
angles = np.concatenate([angles, angles[:1]])

plt.rcParams.update({"font.size": 10})

fig = plt.figure(figsize=(4, 4), dpi=200)
ax = plt.subplot(111, polar=True)

ax.plot(angles, fapi, label="FAPI")
ax.fill(angles, fapi, alpha=0.25)

ax.plot(angles, tempo, "--", label="FAPI–TEMPO")
ax.fill(angles, tempo, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

ax.set_ylim(0.5, 1.5)
ax.set_yticks([0.75, 1.0, 1.25])
ax.set_yticklabels(["0.75", "1.0", "1.25"])

ax.set_title("Normalized descriptors", pad=20)

# --- move legend to the right of the plot ---
leg = ax.legend(
    loc="center left",
    bbox_to_anchor=(1.15, 0.5),  # (x, y) in axes fraction coords
    frameon=True
)

# leave room on the right for the legend
fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])

fig.savefig("descriptor_radar_from_crystal_metrics_5D.png",
            bbox_inches="tight", dpi=300)
