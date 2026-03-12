import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = r"sam\segfreeze_v1_fapi_vs_tempo\counts_shifted.csv"
SMOOTH_WIN = 7                # odd-ish window; try 5,7,9
USE_TCOL = "t_shifted_ms"     # shifted timeline

df = pd.read_csv(CSV)

required = {"dataset", USE_TCOL, "n_kept"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")

for ds, g in df.groupby("dataset"):
    g = g.sort_values(USE_TCOL).reset_index(drop=True)

    # Often you want only t>=0 after shifting
    g = g[g[USE_TCOL] >= 0].copy().reset_index(drop=True)
    t = g[USE_TCOL].to_numpy(dtype=float)
    n = g["n_kept"].to_numpy(dtype=float)

    if len(t) < 3:
        print(ds, "not enough points")
        continue

    dt = np.diff(t)
    dn = np.diff(n)
    rate = dn / dt
    tmid = 0.5 * (t[:-1] + t[1:])

    # simple smoothing on rate (moving average)
    if SMOOTH_WIN and SMOOTH_WIN > 1:
        k = SMOOTH_WIN
        rate_sm = pd.Series(rate).rolling(k, center=True, min_periods=1).mean().to_numpy()
    else:
        rate_sm = rate

    # Plot n(t)
    plt.figure()
    plt.plot(t, n)
    plt.title(f"{ds}: n_kept vs time (shifted)")
    plt.xlabel("t_shifted (ms)")
    plt.ylabel("n_kept")
    plt.grid(True)

    # Plot dn/dt
    plt.figure()
    plt.plot(tmid, rate_sm)
    plt.title(f"{ds}: dn/dt (smoothed, win={SMOOTH_WIN})")
    plt.xlabel("t_shifted (ms)")
    plt.ylabel("dn/dt (masks per ms)")
    plt.axhline(0, linewidth=1)
    plt.grid(True)

    # Plot dn (integer change per frame)
    plt.figure()
    plt.plot(tmid, dn)
    plt.title(f"{ds}: Δn per frame")
    plt.xlabel("t_shifted (ms)")
    plt.ylabel("Δn (masks/frame)")
    plt.axhline(0, linewidth=1)
    plt.grid(True)

plt.show()