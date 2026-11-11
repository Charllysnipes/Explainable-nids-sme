"""Paper and supporting-material figures, generated only from the result files."""
import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))  # repository root, so the code runs from any folder
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figs")
DS = ["CIC-IDS2017", "UNSW-NB15", "NSL-KDD"]
MODELS = ["NaiveBayes", "RandomForest", "XGBoost", "LightGBM", "MLP", "IsolationForest"]
LBL = {"NaiveBayes": "NB", "RandomForest": "RF", "XGBoost": "XGBoost", "LightGBM": "LightGBM", "MLP": "MLP", "IsolationForest": "IF"}
# categorical palette (validated reference order), fixed per model; markers as secondary encoding
COL = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
MRK = dict(zip(MODELS, ["o", "s", "D", "^", "v", "X"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "font.size": 8,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300})
J = lambda n: json.load(open(f"{RES}/{n}"))
main = {d: J(f"{d}_main.json") for d in DS}


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)


# ---- Fig: robustness (F1 vs noise), one panel per dataset
def robustness():
    sig = ["0.0", "0.01", "0.05", "0.1", "0.2"]
    fig, axs = plt.subplots(1, 3, figsize=(3.45, 1.55), sharey=True)
    for ax, d in zip(axs, DS):
        for m in MODELS:
            y = [main[d]["models"][m]["robustness"][s]["f1"] for s in sig]
            ax.plot([float(s) for s in sig], y, color=COL[m], marker=MRK[m], ms=3, lw=1.2, label=LBL[m])
        ax.set_title(d, fontsize=7.5, color=INK, pad=2)
        ax.set_xticks([0, 0.1, 0.2]); ax.tick_params(labelsize=6.5, length=2)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Noise σ", fontsize=7, labelpad=1)
    axs[0].set_ylabel("Test F1-score", fontsize=7)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, fontsize=6.3, bbox_to_anchor=(0.5, 1.13), handlelength=1.4, columnspacing=0.8)
    fig.tight_layout(pad=0.3, w_pad=0.4)
    save(fig, "robustness")


# ---- Fig: generalisation (a) NSL seen vs unseen recall, (b) cross-network F1
def generalisation():
    cr = J("cross_dataset.json")["results"]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    sup = [m for m in MODELS]
    x = np.arange(len(sup)); w = 0.38
    seen = [main["NSL-KDD"]["models"][m]["recall_seen_attacks"] for m in sup]
    nov = [main["NSL-KDD"]["models"][m]["recall_novel_attacks"] for m in sup]
    ax.bar(x - w / 2, seen, w, color="#2a78d6", label="Seen types")
    ax.bar(x + w / 2, nov, w, color="#eb6834", label="Unseen types")
    ax.set_xticks(x); ax.set_xticklabels([LBL[m].replace("LightGBM", "LGBM").replace("XGBoost", "XGB") for m in sup], fontsize=5.6, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Attack recall", fontsize=7); ax.tick_params(labelsize=6.3, length=2)
    ax.set_title("(a) NSL-KDD unseen attacks", fontsize=7.2, pad=2)
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=1)
    ax = axs[1]
    pairs = [("CIC-IDS2017", "CIC-IDS2017", "CIC\n→CIC"), ("CIC-IDS2017", "UNSW-NB15", "CIC\n→UNSW"),
             ("UNSW-NB15", "UNSW-NB15", "UNSW\n→UNSW"), ("UNSW-NB15", "CIC-IDS2017", "UNSW\n→CIC")]
    ms = ["RandomForest", "XGBoost", "LightGBM", "MLP"]
    x = np.arange(len(pairs)); w = 0.2
    for k, m in enumerate(ms):
        ax.bar(x + (k - 1.5) * w, [cr[s][t][m]["roc_auc"] for s, t, _ in pairs], w, color=COL[m], label=LBL[m])
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--"); ax.text(3.55, 0.51, "chance", fontsize=5.5, color=MUTED, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([p[2] for p in pairs], fontsize=5.3)
    ax.set_ylim(0, 1.05); ax.set_ylabel("ROC-AUC", fontsize=7); ax.tick_params(labelsize=6.3, length=2)
    ax.set_title("(b) Cross-network transfer (7 features)", fontsize=7.2, pad=2)
    ax.legend(fontsize=5.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.26), handlelength=1, columnspacing=0.7)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    save(fig, "generalisation")


def abbr(n):
    r = n
    for a, b in [("Init_Win_bytes_forward", "Init win fwd"), ("Init_Win_bytes_backward", "Init win bwd"),
                 ("Packet Length", "Pkt len"), ("Header Length", "Hdr len"), ("dst_host_", "dh_"),
                 ("min_seg_size_forward", "Min seg fwd"), ("_rate", " rate"), ("_", " ")]:
        r = r.replace(a, b)
    return r[:24]


# ---- Fig: SHAP global importance (XGBoost), top-8 per dataset
def shap_global():
    sh = J("shap.json")
    fig, axs = plt.subplots(1, 3, figsize=(3.45, 1.9))
    for ax, d in zip(axs, DS):
        g = list(sh[d]["global_xgb"].items())[:8][::-1]
        names = [abbr(n) for n, _ in g]
        ax.barh(range(len(g)), [v for _, v in g], color="#2a78d6", height=0.65)
        ax.set_yticks(range(len(g))); ax.set_yticklabels(names, fontsize=5.6)
        ax.tick_params(axis="x", labelsize=5.5, length=2); ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_title(d, fontsize=7, pad=2)
        ax.set_xlabel("mean |SHAP|", fontsize=6, labelpad=1)
    fig.tight_layout(pad=0.2, w_pad=0.25)
    save(fig, "shap_global")


# ---- Fig: local explanation of one alert (CIC-IDS2017)
def shap_local():
    sh = J("shap.json")["CIC-IDS2017"]["local_example"]
    items = list(sh["contributions"].items())[::-1]
    fig, ax = plt.subplots(figsize=(3.45, 1.55))
    vals = [v for _, v in items]
    ax.barh(range(len(items)), vals, color=["#e34948" if v > 0 else "#2a78d6" for v in vals], height=0.6)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([k for k, _ in items], fontsize=6)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("SHAP contribution to log-odds of attack", fontsize=6.5)
    ax.tick_params(labelsize=6, length=2); ax.grid(axis="y", visible=False)
    ax.set_title(f"Alert explanation: {sh['family']} flow, p(attack) = {sh['prob']:.3f}", fontsize=7, pad=2)
    fig.tight_layout(pad=0.2)
    save(fig, "shap_local")


# ---- Fig: Nemenyi critical difference diagram
def cd_diagram():
    cv = J("cv_stats.json")
    ranks = cv["avg_rank"]; cd = cv["nemenyi_cd"]
    order = sorted(MODELS, key=lambda m: ranks[m])
    k = len(MODELS)
    fig, ax = plt.subplots(figsize=(3.45, 1.35))
    ax.set_xlim(-1.3, k + 2.3); ax.set_ylim(-0.15, 1.45); ax.axis("off")
    Y = 1.0
    ax.plot([1, k], [Y, Y], color=INK, lw=0.8)
    for r in range(1, k + 1):
        ax.plot([r, r], [Y, Y + 0.06], color=INK, lw=0.8); ax.text(r, Y + 0.1, str(r), ha="center", fontsize=6.5)
    ax.plot([1, 1 + cd], [Y + 0.33, Y + 0.33], color="#c0392b", lw=1.4)
    for xx in (1, 1 + cd):
        ax.plot([xx, xx], [Y + 0.29, Y + 0.37], color="#c0392b", lw=1.0)
    ax.text(1 + cd / 2, Y + 0.38, f"CD = {cd:.2f}", ha="center", fontsize=6, color="#a4231c")
    half = k // 2
    rows = [0.55, 0.32, 0.09]
    for i, m in enumerate(order):
        r = ranks[m]
        if i < half:
            yy = rows[i]; xe = 0.4
            ax.plot([r, r], [Y, yy], color=MUTED, lw=0.7); ax.plot([xe, r], [yy, yy], color=MUTED, lw=0.7)
            ax.text(xe - 0.05, yy, f"{LBL[m]} ({r:.2f})", ha="right", va="center", fontsize=6.2)
        else:
            yy = rows[k - 1 - i]; xe = k + 0.6
            ax.plot([r, r], [Y, yy], color=MUTED, lw=0.7); ax.plot([r, xe], [yy, yy], color=MUTED, lw=0.7)
            ax.text(xe + 0.05, yy, f"{LBL[m]} ({r:.2f})", ha="left", va="center", fontsize=6.2)
    rs = [ranks[m] for m in order]; cl = []
    for i in range(k):
        j = max(jj for jj in range(k) if rs[jj] - rs[i] < cd)
        if j > i and not any(a_ <= i and j <= b_ for a_, b_ in cl):
            cl.append((i, j))
    for n_, (a_, b_) in enumerate(cl):
        y = Y - 0.12 - n_ * 0.08
        ax.plot([rs[a_] - 0.05, rs[b_] + 0.05], [y, y], color=INK, lw=2.0, solid_capstyle="round")
    save(fig, "cd_diagram")


# ---- Supporting: ROC-free summary bar of test F1 & FPR across models
def f1_fpr():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    x = np.arange(len(DS)); w = 0.13
    for k, m in enumerate(MODELS):
        axs[0].bar(x + (k - 2.5) * w, [main[d]["models"][m]["f1"] for d in DS], w, color=COL[m], label=LBL[m])
        axs[1].bar(x + (k - 2.5) * w, [main[d]["models"][m]["fpr"] for d in DS], w, color=COL[m], label=LBL[m])
    for ax, t in zip(axs, ["Test F1-score", "Test false positive rate"]):
        ax.set_xticks(x); ax.set_xticklabels(DS); ax.set_title(t, fontsize=9); ax.set_ylim(0, 1.05)
    axs[1].legend(ncol=3, fontsize=7, loc="upper left")
    fig.tight_layout(); save(fig, "f1_fpr")


def family_heatmap():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [7, 10, 5]})
    for ax, d in zip(axs, DS):
        fams = sorted(main[d]["models"]["XGBoost"]["family_detection"].keys())
        M = np.array([[main[d]["models"][m]["family_detection"][f] for f in fams] for m in MODELS])
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([LBL[m] for m in MODELS], fontsize=7)
        ax.grid(False); ax.set_title(d, fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5.2, color="white" if M[i, j] > 0.6 else INK)
    fig.tight_layout(); save(fig, "family_heatmap")


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:] or ["robustness", "generalisation", "shap_global", "shap_local", "cd_diagram", "f1_fpr", "family_heatmap"]:
        try:
            globals()[f](); print("ok", f)
        except Exception as e:
            print("skip", f, type(e).__name__, e)
