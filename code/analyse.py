import json, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

TEST = 75
YLIM = {"AAPL": (105, 195)}
MODELS = [("rnn", "Basic RNN"), ("lstm", "LSTM"), ("bilstm", "BiLSTM"), ("gru", "GRU")]
COL = {"rnn": "#4C72B0", "lstm": "#DD8452", "bilstm": "#55A868", "gru": "#8172B3", "mc": "#937860"}

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.6, "savefig.dpi": 300,
})


def metrics(y, f):
    e = y - f
    mape = np.mean(np.abs(e / y)) * 100
    rmse = np.sqrt(np.mean(e ** 2))
    mae = np.mean(np.abs(e))
    acc = (1 - np.sqrt(np.mean(np.square(((y + 1) - (f + 1)) / (y + 1))))) * 100  # repo metric
    da = np.mean(np.sign(np.diff(y)) == np.sign(np.diff(f))) * 100
    return dict(MAPE=mape, RMSE=rmse, MAE=mae, ACC=acc, DA=da)


def monte_carlo(train_close, n_paths=10000, horizon=TEST, seed=42):
    rng = np.random.default_rng(seed)
    vol = train_close.pct_change().std()
    shocks = 1 + rng.normal(0, vol, size=(n_paths, horizon))
    paths = train_close.iloc[-1] * np.cumprod(shocks, axis=1)
    return paths, vol


def analyse(stock, tag="", outdir="figs"):
    df = pd.read_csv(f"{stock}.csv", parse_dates=["Date"])
    y = df.Close.values[-TEST:]
    dates = df.Date.values[-TEST:]
    train = df.Close.iloc[:-TEST]
    out = {"stock": stock}
    runs = {}
    for k, name in MODELS:
        F = np.load(f"res/{stock}{tag}_{k}_forecasts.npy")
        T = np.load(f"res/{stock}{tag}_{k}_times.npy")
        per = pd.DataFrame([metrics(y, f) for f in F])
        ens = metrics(y, F.mean(0))
        runs[k] = (F, per)
        out[k] = {
            "n": len(F),
            "per_run_mean": per.mean().round(3).to_dict(),
            "per_run_sd": per.std().round(3).to_dict(),
            "per_run_min_mape": round(per.MAPE.min(), 3), "per_run_max_mape": round(per.MAPE.max(), 3),
            "per_run_median_mape": round(per.MAPE.median(), 3),
            "ensemble": {a: round(b, 3) for a, b in ens.items()},
            "time_mean": round(T.mean(), 2), "time_sd": round(T.std(), 2), "time_total": round(T.sum(), 1),
        }
    naive = np.repeat(train.iloc[-1], TEST)
    out["naive"] = {a: round(b, 3) for a, b in metrics(y, naive).items()}
    paths, vol = monte_carlo(train)
    med = np.median(paths, 0); lo, hi = np.percentile(paths, [5, 95], axis=0)
    out["mc"] = {a: round(b, 3) for a, b in metrics(y, med).items()}
    for k, _ in MODELS:
        out[k]["beat_naive_pct"] = round(float(np.mean(runs[k][1].MAPE < out["naive"]["MAPE"]) * 100), 1)
        out[k]["beat_mc_pct"] = round(float(np.mean(runs[k][1].MAPE < out["mc"]["MAPE"]) * 100), 1)
    out["last_train_close"] = round(float(train.iloc[-1]), 2); out["first_test_close"] = round(float(y[0]), 2)
    out["mc_vol_daily_pct"] = round(vol * 100, 3)
    out["mc_coverage_90"] = round(np.mean((y >= lo) & (y <= hi)) * 100, 1)
    out["mc_band_width_end"] = [round(lo[-1], 2), round(hi[-1], 2)]
    for k, _ in MODELS:
        m = runs[k][0].mean(0)
        out[k]["ens_inside_mc90_pct"] = round(np.mean((m >= lo) & (m <= hi)) * 100, 1)
        out[k]["runs_inside_mc90_pct"] = round(np.mean((runs[k][0] >= lo) & (runs[k][0] <= hi)) * 100, 1)
    best = min(MODELS, key=lambda m: out[m[0]]["per_run_mean"]["MAPE"])[0]
    out["best"] = best
    out["mwu_vs_best"] = {}
    for k, _ in MODELS:
        if k != best:
            p = mannwhitneyu(runs[best][1].MAPE, runs[k][1].MAPE, alternative="less").pvalue
            out["mwu_vs_best"][k] = float(f"{p:.3g}")
    out["test_start"] = str(df.Date.iloc[-TEST].date()); out["test_end"] = str(df.Date.iloc[-1].date())

    # Figure: forecasts, one panel per model, identical axes
    if tag == "":
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True, sharey=True)
        allv = np.concatenate([y] + [runs[k][0].ravel() for k, _ in MODELS])
        ylo, yhi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
        for ax, (k, name) in zip(axes.ravel(), MODELS):
            F, per = runs[k]
            p5, p95 = np.percentile(F, [5, 95], axis=0)
            ax.fill_between(dates, p5, p95, color=COL[k], alpha=0.18, lw=0, label="5th to 95th percentile of 50 runs")
            ax.plot(dates, F.mean(0), color=COL[k], lw=1.4, label="Mean forecast")
            ax.plot(dates, y, color="black", lw=1.0, label="Actual close")
            ax.set_title(f"({'abcd'[[m for m,_ in MODELS].index(k)]}) {name}: mean MAPE {per.MAPE.mean():.2f}%")
            ax.set_ylim(*(YLIM.get(stock) or (ylo - 3, yhi + 3)))
            if k == "rnn": ax.legend(loc="lower left", frameon=False)
        for ax in axes[:, 0]: ax.set_ylabel("Close price (USD)")
        for ax in axes[1]:
            ax.tick_params(axis="x", rotation=30)
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        fig.tight_layout()
        fig.savefig(f"{outdir}/{stock}_forecasts.png"); plt.close(fig)

        # Figure: Monte Carlo
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
        for i in range(60):
            a1.plot(dates, paths[i], color=COL["mc"], lw=0.35, alpha=0.35)
        a1.fill_between(dates, lo, hi, color=COL["mc"], alpha=0.15, lw=0, label="MC 5th to 95th pct.")
        a1.plot(dates, med, color=COL["mc"], lw=1.4, ls="--", label="MC median")
        a1.plot(dates, runs[best][0].mean(0), color=COL[best], lw=1.4, label=f"{dict(MODELS)[best]} mean forecast")
        a1.plot(dates, y, color="black", lw=1.0, label="Actual close")
        a1.set_ylabel("Close price (USD)"); a1.set_title("(a) Simulated paths over the test window")
        import matplotlib.dates as mdates
        a1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); a1.tick_params(axis="x", rotation=30)
        a1.legend(loc="upper left", frameon=False, fontsize=6.8)
        bins = np.linspace(min(paths.min(), y.min()), max(np.percentile(paths, 99.5), y.max()), 40)
        a2.hist(paths.ravel(), bins=bins, density=True, color=COL["mc"], alpha=0.45, label="MC samples")
        a2.hist(y, bins=bins, density=True, color="black", alpha=0.55, label="Actual closes")
        a2.set_xlabel("Close price (USD)"); a2.set_ylabel("Density"); a2.set_title("(b) Price distributions")
        a2.legend(frameon=False)
        fig.tight_layout(); fig.savefig(f"{outdir}/{stock}_montecarlo.png"); plt.close(fig)

        # Figure: error and time comparison
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
        data = [runs[k][1].MAPE.values for k, _ in MODELS]
        bp = a1.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="black"))
        for patch, (k, _) in zip(bp["boxes"], MODELS):
            patch.set_facecolor(COL[k]); patch.set_alpha(0.55)
        a1.axhline(out["naive"]["MAPE"], color="grey", ls=":", lw=1, label="Naive (last close)")
        a1.axhline(out["mc"]["MAPE"], color=COL["mc"], ls="--", lw=1, label="Monte Carlo median")
        a1.set_xticks(range(1, 5)); a1.set_xticklabels([n for _, n in MODELS])
        a1.set_ylabel("Test MAPE (%)"); a1.set_title("(a) Error across 50 runs"); a1.legend(frameon=False, fontsize=6.8)
        tm = [out[k]["time_mean"] for k, _ in MODELS]; ts = [out[k]["time_sd"] for k, _ in MODELS]
        a2.bar([n for _, n in MODELS], tm, yerr=ts, color=[COL[k] for k, _ in MODELS], alpha=0.75, capsize=3)
        a2.set_ylabel("Seconds per run"); a2.set_title("(b) Training and forecast time per run")
        fig.tight_layout(); fig.savefig(f"{outdir}/{stock}_comparison.png"); plt.close(fig)
    return out


if __name__ == "__main__":
    import os
    os.makedirs("figs", exist_ok=True)
    res = {}
    for s, tag in [("AAPL", ""), ("JPM", ""), ("AAPL", "tr")]:
        try:
            res[s + tag] = analyse(s, tag)
        except FileNotFoundError as e:
            print("missing", e)
    json.dump(res, open("results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
