def load_dndt(dndt_csv):
    """
    Load dn/dt CSV with flexible header detection.
    Accepts time in ms under any of: t_ms, time_ms, ms, t
    Accepts dn/dt under any of: dn_dt, dn_dt_total, dn_dt_field, rate, density
    """
    d = pd.read_csv(dndt_csv)

    # Try possible time column names
    time_candidates = ["t_ms", "time_ms", "ms", "t"]
    tcol = next((c for c in time_candidates if c in d.columns), None)

    # Try possible dn/dt column names (nuclei per ms across field)
    dndt_candidates = ["dn_dt", "dn_dt_total", "dn_dt_field", "rate", "density"]
    rcol = next((c for c in dndt_candidates if c in d.columns), None)

    if tcol is None or rcol is None:
        raise ValueError(
            f"{dndt_csv} must contain a time column (one of {time_candidates}) "
            f"and a rate column (one of {dndt_candidates}). Found columns: {list(d.columns)}"
        )

    t_ms  = d[tcol].astype(float).to_numpy()
    dn_dt = d[rcol].astype(float).to_numpy()
    return t_ms, dn_dt, d
