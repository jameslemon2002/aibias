from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import colormaps


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = ROOT / "source_data" / "supplement" / "robust100_supplement_mpnet_siebert_v1"
GROUP_ROOT = ANALYSIS_ROOT / "group_summaries"
SUPP_FIG_DATA = ROOT / "source_data" / "supplement" / "supplement_figures"
OUT_ROOT = ROOT / "figures" / "supplement"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


MODEL_LABELS = {
    "macro5_gpt52": "GPT-5.2",
    "macro5_claude": "Claude Sonnet 4",
    "macro5_gemini": "Gemini 2.5 Flash",
    "micro20_gpt52": "GPT-5.2",
    "micro20_claude": "Claude Sonnet 4",
    "micro20_gemini": "Gemini 2.5 Flash",
    "meso7_gpt52_original": "GPT-5.2",
    "meso7_gpt52_weak": "GPT-5.2 weak",
    "meso7_qwen": "Qwen2.5-VL-72B",
    "meso7_llama": "Llama 3.2 11B Vision",
    "meso7_gemma": "Gemma 3 27B",
}

MODEL_ORDER_3 = ["GPT-5.2", "Claude Sonnet 4", "Gemini 2.5 Flash"]
MODEL_ORDER_4 = ["GPT-5.2", "Qwen2.5-VL-72B", "Llama 3.2 11B Vision", "Gemma 3 27B"]
MODEL_ORDER_4_SHORT = ["GPT-5.2", "Qwen 2.5-VL", "Llama 3.2 11B", "Gemma 3 27B"]
MODEL_COLORS = {
    "GPT-5.2": "#4d80bc",
    "Claude Sonnet 4": "#76789b",
    "Gemini 2.5 Flash": "#c9758e",
    "Qwen2.5-VL-72B": "#76B7B2",
    "Llama 3.2 11B Vision": "#E15759",
    "Gemma 3 27B": "#9C755F",
}

MACRO_LABELS = {
    "europe_northern_america": "Europe and\nNorthern America",
    "asia": "Asia",
    "africa": "Africa",
    "oceania": "Oceania",
    "latin_america_caribbean": "Latin America and\nthe Caribbean",
}

MICRO_LABELS = {
    "northern_america": "Northern America",
    "western_europe": "Western Europe",
    "northern_europe": "Northern Europe",
    "southern_europe": "Southern Europe",
    "eastern_europe": "Eastern Europe",
    "central_asia": "Central Asia",
    "southern_asia": "Southern Asia",
    "northern_africa": "Northern Africa",
    "western_asia": "Western Asia",
    "eastern_asia": "Eastern Asia",
    "south_eastern_asia": "South-eastern Asia",
    "eastern_africa": "Eastern Africa",
    "middle_africa": "Middle Africa",
    "western_africa": "Western Africa",
    "southern_africa": "Southern Africa",
    "caribbean": "Caribbean",
    "central_america": "Central America",
    "south_america": "South America",
    "australia_new_zealand": "Australia + NZ",
    "pacific_islands": "Pacific islands",
}

MESO_LABELS = {
    "europe_northern_america": "Europe and\nNorthern America",
    "central_southern_asia": "Central and\nSouthern Asia",
    "northern_africa_western_asia": "Northern Africa\nand Western Asia",
    "east_southeast_asia": "Eastern and\nSouth-eastern Asia",
    "sub_saharan_africa": "Sub-Saharan\nAfrica",
    "latin_america_caribbean": "Latin America and\nthe Caribbean",
    "oceania": "Oceania",
}

MICRO_TO_MACRO = {
    "northern_america": "Europe + N. America",
    "western_europe": "Europe + N. America",
    "northern_europe": "Europe + N. America",
    "southern_europe": "Europe + N. America",
    "eastern_europe": "Europe + N. America",
    "central_asia": "Asia",
    "southern_asia": "Asia",
    "western_asia": "Asia",
    "eastern_asia": "Asia",
    "south_eastern_asia": "Asia",
    "northern_africa": "Africa",
    "eastern_africa": "Africa",
    "middle_africa": "Africa",
    "western_africa": "Africa",
    "southern_africa": "Africa",
    "caribbean": "Latin America + Caribbean",
    "central_america": "Latin America + Caribbean",
    "south_america": "Latin America + Caribbean",
    "australia_new_zealand": "Oceania",
    "pacific_islands": "Oceania",
}

MACRO_REGION_TO_CONDITION = {
    "Northern America": "europe_northern_america",
    "Western Europe": "europe_northern_america",
    "Northern Europe": "europe_northern_america",
    "Southern Europe": "europe_northern_america",
    "Eastern Europe": "europe_northern_america",
    "Central Asia": "asia",
    "Southern Asia": "asia",
    "Western Asia": "asia",
    "Eastern Asia": "asia",
    "South-eastern Asia": "asia",
    "Northern Africa": "africa",
    "Eastern Africa": "africa",
    "Middle Africa": "africa",
    "Western Africa": "africa",
    "Southern Africa": "africa",
    "Caribbean": "latin_america_caribbean",
    "Central America": "latin_america_caribbean",
    "South America": "latin_america_caribbean",
    "Australia and New Zealand": "oceania",
    "Pacific islands": "oceania",
}

MICRO_REGION_TO_CONDITION = {
    "Northern America": "northern_america",
    "Western Europe": "western_europe",
    "Northern Europe": "northern_europe",
    "Southern Europe": "southern_europe",
    "Eastern Europe": "eastern_europe",
    "Central Asia": "central_asia",
    "Southern Asia": "southern_asia",
    "Northern Africa": "northern_africa",
    "Western Asia": "western_asia",
    "Eastern Asia": "eastern_asia",
    "South-eastern Asia": "south_eastern_asia",
    "Eastern Africa": "eastern_africa",
    "Middle Africa": "middle_africa",
    "Western Africa": "western_africa",
    "Southern Africa": "southern_africa",
    "Caribbean": "caribbean",
    "Central America": "central_america",
    "South America": "south_america",
    "Australia and New Zealand": "australia_new_zealand",
    "Pacific islands": "pacific_islands",
}

MACRO_COLORS = {
    "Europe + N. America": "#4E79A7",
    "Asia": "#59A14F",
    "Africa": "#E15759",
    "Latin America + Caribbean": "#F28E2B",
    "Oceania": "#76B7B2",
}


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "run_key" in out.columns:
        out["model_label"] = out["run_key"].map(MODEL_LABELS).fillna(out.get("model_label"))
    return out


def hide_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def draw_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    cmap: str,
    norm: Normalize,
    fmt: str,
    cbar_label: str,
) -> None:
    im = ax.imshow(data.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    cmap_obj = colormaps[cmap]
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=0)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index)
    ax.tick_params(length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iat[i, j]
            if np.isnan(val):
                continue
            r, g, b, _ = cmap_obj(norm(val))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = "white" if luminance < 0.52 else "#1a1a1a"
            ax.text(j, i, format(val, fmt), ha="center", va="center", color=text_color, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.set_ylabel(cbar_label, rotation=90, va="bottom")
    hide_spines(ax)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_ROOT / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_ROOT / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_identity_ipi_bootstrap(
    run_keys: list[str],
    mapper: dict[str, str],
    output_name: str,
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    existing = SUPP_FIG_DATA / output_name
    if existing.exists():
        return clean_frame(pd.read_csv(existing))
    rng = np.random.default_rng(seed)
    run_root = ANALYSIS_ROOT / "run_analyses"
    records = []
    for run_key in run_keys:
        rows = pd.read_csv(run_root / run_key / "rows_sentiment.csv")
        region_col = "region_x" if "region_x" in rows.columns else "region"
        rows = rows[["row_index", "condition_key", "sentiment_score_stripped", region_col]].copy()
        rows["self_condition_key"] = rows[region_col].map(mapper)
        pivot = rows.pivot_table(index="row_index", columns="condition_key", values="sentiment_score_stripped")
        meta = rows[["row_index", "self_condition_key"]].drop_duplicates(subset=["row_index"]).set_index("row_index")
        pivot = pivot.merge(meta, left_index=True, right_index=True, how="inner")
        identity_keys = [c for c in pivot.columns if c not in {"self_condition_key", "neutral"}]

        for identity_key in identity_keys:
            sub = pivot[pivot["self_condition_key"] == identity_key].copy()
            mat = sub[identity_keys].to_numpy(dtype=float)
            if mat.size == 0:
                continue
            self_idx = identity_keys.index(identity_key)
            other_mask = np.ones(len(identity_keys), dtype=bool)
            other_mask[self_idx] = False

            boot = []
            for _ in range(n_boot):
                sample_idx = rng.integers(0, mat.shape[0], size=mat.shape[0])
                boot_mat = mat[sample_idx]
                sigma = float(np.nanstd(boot_mat.reshape(-1), ddof=1))
                if sigma <= 0 or np.isnan(sigma):
                    boot.append(np.nan)
                    continue
                mu_self = float(np.nanmean(boot_mat[:, self_idx]))
                mu_other = float(np.nanmean(boot_mat[:, other_mask]))
                boot.append((mu_self - mu_other) / sigma)

            point = float(np.nanmean(mat[:, self_idx]) - np.nanmean(mat[:, other_mask]))
            sigma = float(np.nanstd(mat.reshape(-1), ddof=1))
            ipi = point / sigma if sigma > 0 else np.nan
            ci_low, ci_high = np.nanpercentile(np.asarray(boot, dtype=float), [2.5, 97.5])
            records.append(
                {
                    "run_key": run_key,
                    "model_label": MODEL_LABELS[run_key],
                    "condition_key": identity_key,
                    "ipi": ipi,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    out = pd.DataFrame(records)
    out.to_csv(OUT_ROOT / output_name, index=False, encoding="utf-8-sig")
    return out


def plot_s8_macro() -> None:
    nd = clean_frame(pd.read_csv(GROUP_ROOT / "macro5_identity_neutral_distance.csv"))
    ipi = compute_identity_ipi_bootstrap(
        ["macro5_gpt52", "macro5_claude", "macro5_gemini"],
        MACRO_REGION_TO_CONDITION,
        "supp_fig_s8_macro5_ipi_bootstrap.csv",
    )

    macro_order = (
        nd.groupby("condition_key")["mean_pair_distance"].mean().sort_values().index.tolist()
    )
    nd = nd.copy()
    nd["ci_low"] = nd["mean_pair_distance"] - 1.96 * nd["se_pair_distance"]
    nd["ci_high"] = nd["mean_pair_distance"] + 1.96 * nd["se_pair_distance"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    offsets = [-0.22, 0.0, 0.22]
    base_y = np.arange(len(macro_order))[::-1]

    for model_idx, model in enumerate(MODEL_ORDER_3):
        sub = nd[nd["model_label"] == model].set_index("condition_key").loc[macro_order].reset_index()
        y = base_y + offsets[model_idx]
        x = sub["mean_pair_distance"].to_numpy()
        xerr = np.vstack([x - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - x])
        axes[0].errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=1.5,
            capsize=2.5,
            markersize=5.2,
            label=model,
        )
    axes[0].set_yticks(base_y)
    axes[0].set_yticklabels([MACRO_LABELS[k].replace("\n", " ") for k in macro_order])
    axes[0].set_xlabel("Mean cosine distance to neutral")
    axes[0].set_title("Macro5 semantic distance to neutral")
    add_panel_label(axes[0], "a")
    hide_spines(axes[0])

    for model_idx, model in enumerate(MODEL_ORDER_3):
        sub = ipi[ipi["model_label"] == model].set_index("condition_key").loc[macro_order].reset_index()
        y = base_y + offsets[model_idx]
        x = sub["ipi"].to_numpy()
        xerr = np.vstack([x - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - x])
        axes[1].errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=1.5,
            capsize=2.5,
            markersize=5.2,
        )
    axes[1].axvline(0, color="#444444", lw=0.8)
    axes[1].set_yticks(base_y)
    axes[1].set_yticklabels([MACRO_LABELS[k].replace("\n", " ") for k in macro_order])
    axes[1].set_xlabel("Affect-based IPI")
    axes[1].set_title("Macro5 affect-based IPI")
    add_panel_label(axes[1], "b")
    hide_spines(axes[1])
    handles = [
        plt.Line2D([0], [0], marker="o", color=MODEL_COLORS[m], lw=1.5, markersize=5, label=m)
        for m in MODEL_ORDER_3
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
    save(fig, "supp_fig_s8_macro5_identity_robustness")


def plot_s9_micro() -> None:
    nd = clean_frame(pd.read_csv(GROUP_ROOT / "micro20_identity_neutral_distance.csv"))
    ipi = compute_identity_ipi_bootstrap(
        ["micro20_gpt52", "micro20_claude", "micro20_gemini"],
        MICRO_REGION_TO_CONDITION,
        "supp_fig_s9_micro20_ipi_bootstrap.csv",
    )

    micro_order = (
        nd.groupby("condition_key")["rank_closest_to_neutral"].mean().sort_values().index.tolist()
    )
    nd = nd.copy()
    nd["ci_low"] = nd["mean_pair_distance"] - 1.96 * nd["se_pair_distance"]
    nd["ci_high"] = nd["mean_pair_distance"] + 1.96 * nd["se_pair_distance"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 8.2), constrained_layout=True)
    offsets = [-0.22, 0.0, 0.22]
    base_y = np.arange(len(micro_order))[::-1]

    for model_idx, model in enumerate(MODEL_ORDER_3):
        sub = nd[nd["model_label"] == model].set_index("condition_key").loc[micro_order].reset_index()
        y = base_y + offsets[model_idx]
        x = sub["mean_pair_distance"].to_numpy()
        xerr = np.vstack([x - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - x])
        axes[0].errorbar(
            x,
            y,
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            fmt="o",
            xerr=xerr,
            elinewidth=1.2,
            capsize=2.0,
            markersize=4.0,
            label=model,
        )
    axes[0].set_yticks(base_y)
    axes[0].set_yticklabels([MICRO_LABELS[k] for k in micro_order])
    axes[0].set_xlabel("Mean cosine distance to neutral")
    axes[0].set_title("Micro20 semantic distance to neutral")
    add_panel_label(axes[0], "a")
    hide_spines(axes[0])

    for model_idx, model in enumerate(MODEL_ORDER_3):
        sub = ipi[ipi["model_label"] == model].set_index("condition_key").loc[micro_order].reset_index()
        y = base_y + offsets[model_idx]
        x = sub["ipi"].to_numpy()
        xerr = np.vstack([x - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - x])
        axes[1].errorbar(
            x,
            y,
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            fmt="o",
            xerr=xerr,
            elinewidth=1.2,
            capsize=2.0,
            markersize=4.0,
        )
    axes[1].axvline(0, color="#444444", lw=0.8)
    axes[1].set_yticks(base_y)
    axes[1].set_yticklabels([MICRO_LABELS[k] for k in micro_order])
    axes[1].set_xlabel("Affect-based IPI")
    axes[1].set_title("Micro20 affect-based IPI")
    add_panel_label(axes[1], "b")
    hide_spines(axes[1])

    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS[m], lw=6, label=m)
        for m in MODEL_ORDER_3
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)

    save(fig, "supp_fig_s9_micro20_identity_robustness")


def plot_s10_prompt_form() -> None:
    nd = pd.read_csv(GROUP_ROOT / "meso7_prompt_form_neutral_distance_comparison.csv")
    ipi = clean_frame(pd.read_csv(GROUP_ROOT / "meso7_prompt_form_ipi.csv"))

    order = nd.sort_values("rank_closest_to_neutral_original")["condition_key"].tolist()
    labels = [MESO_LABELS[key].replace("\n", " ") for key in order]

    fig = plt.figure(figsize=(11.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.9])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    x = [0, 1]
    for _, row in nd.set_index("condition_key").loc[order].reset_index().iterrows():
        y = [row["mean_pair_distance_original"], row["mean_pair_distance_weak"]]
        ax1.plot(x, y, marker="o", lw=1.4, color="#4E79A7", alpha=0.85)
        ax1.text(1.03, y[1], MESO_LABELS[row["condition_key"]].replace("\n", " "), va="center", fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Original", "Weak-context"])
    ax1.set_xlim(-0.05, 1.48)
    ax1.set_ylabel("Mean cosine distance to neutral")
    ax1.set_title("Prompt-form effect on neutral distance")
    add_panel_label(ax1, "a")
    hide_spines(ax1)

    ipi_wide = (
        ipi.pivot(index="condition_key", columns="model_label", values="ipi")
        .rename(columns={"GPT-5.2": "Original", "GPT-5.2 weak": "Weak-context"})
        .loc[order]
    )
    for condition_key in order:
        y = [ipi_wide.loc[condition_key, "Original"], ipi_wide.loc[condition_key, "Weak-context"]]
        ax2.plot(x, y, marker="o", lw=1.4, color="#E15759", alpha=0.85)
        ax2.text(1.03, y[1], MESO_LABELS[condition_key].replace("\n", " "), va="center", fontsize=7)
    ax2.axhline(0, color="#444444", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Original", "Weak-context"])
    ax2.set_xlim(-0.05, 1.48)
    ax2.set_ylabel("Affect-based IPI")
    ax2.set_title("Prompt-form effect on IPI")
    add_panel_label(ax2, "b")
    hide_spines(ax2)

    nd["delta_distance"] = nd["mean_pair_distance_weak"] - nd["mean_pair_distance_original"]
    nd_plot = nd.set_index("condition_key").loc[order].reset_index()
    ax3.barh(
        np.arange(len(nd_plot)),
        nd_plot["delta_distance"],
        color="#59A14F",
        edgecolor="none",
    )
    ax3.set_yticks(np.arange(len(nd_plot)))
    ax3.set_yticklabels(labels)
    ax3.invert_yaxis()
    ax3.set_xlabel("Weak - original distance")
    ax3.set_title("Semantic displacement under weak prompt")
    add_panel_label(ax3, "c")
    ax3.text(
        0.02,
        0.02,
        "Spearman\nneutral distance = 0.82\nIPI = 0.21",
        transform=ax3.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
    )
    hide_spines(ax3)

    save(fig, "supp_fig_s10_prompt_form_sensitivity")


def plot_s11_model_extension() -> pd.DataFrame:
    nd = clean_frame(pd.read_csv(GROUP_ROOT / "meso7_model_extension_neutral_distance.csv"))
    ipi = clean_frame(pd.read_csv(GROUP_ROOT / "meso7_model_extension_ipi.csv"))

    order = nd.groupby("condition_key")["rank_closest_to_neutral"].mean().sort_values().index.tolist()
    nd_mat = (
        nd.pivot(index="condition_key", columns="model_label", values="mean_pair_distance")
        .loc[order, MODEL_ORDER_4]
        .rename(columns=dict(zip(MODEL_ORDER_4, MODEL_ORDER_4_SHORT)))
        .rename(index=MESO_LABELS)
    )
    ipi_mat = (
        ipi.pivot(index="condition_key", columns="model_label", values="ipi")
        .loc[order, MODEL_ORDER_4]
        .rename(columns=dict(zip(MODEL_ORDER_4, MODEL_ORDER_4_SHORT)))
        .rename(index=MESO_LABELS)
    )

    metrics = []
    gpt_nd = nd[nd["model_label"] == "GPT-5.2"][["condition_key", "mean_pair_distance"]].rename(columns={"mean_pair_distance": "gpt"})
    gpt_ipi = ipi[ipi["model_label"] == "GPT-5.2"][["condition_key", "ipi"]].rename(columns={"ipi": "gpt"})
    for label in MODEL_ORDER_4[1:]:
        sub_nd = nd[nd["model_label"] == label][["condition_key", "mean_pair_distance"]].rename(columns={"mean_pair_distance": "other"})
        rho_nd = gpt_nd.merge(sub_nd, on="condition_key")["gpt"].corr(gpt_nd.merge(sub_nd, on="condition_key")["other"], method="spearman")
        sub_ipi = ipi[ipi["model_label"] == label][["condition_key", "ipi"]].rename(columns={"ipi": "other"})
        rho_ipi = gpt_ipi.merge(sub_ipi, on="condition_key")["gpt"].corr(gpt_ipi.merge(sub_ipi, on="condition_key")["other"], method="spearman")
        metrics.append({"model": label, "neutral_distance_spearman": rho_nd, "ipi_spearman": rho_ipi})
    metrics_df = pd.DataFrame(metrics)

    fig = plt.figure(figsize=(12.2, 5.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.9])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    draw_heatmap(
        fig,
        ax1,
        nd_mat,
        cmap="Blues",
        norm=Normalize(vmin=float(nd_mat.min().min()), vmax=float(nd_mat.max().max())),
        fmt=".3f",
        cbar_label="Mean cosine distance to neutral",
    )
    ax1.set_title("Neutral distance across model families")
    add_panel_label(ax1, "a")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=16, ha="right")

    draw_heatmap(
        fig,
        ax2,
        ipi_mat,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=float(ipi_mat.min().min()), vcenter=0.0, vmax=float(ipi_mat.max().max())),
        fmt=".2f",
        cbar_label="Affect-based IPI",
    )
    ax2.set_title("Affect-based IPI across model families")
    add_panel_label(ax2, "b")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=16, ha="right")

    x = np.arange(len(metrics_df))
    width = 0.34
    ax3.bar(x - width / 2, metrics_df["neutral_distance_spearman"], width=width, color="#4E79A7", label="Neutral distance")
    ax3.bar(x + width / 2, metrics_df["ipi_spearman"], width=width, color="#E15759", label="IPI")
    ax3.axhline(0, color="#444444", lw=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Qwen", "Llama", "Gemma"], rotation=14, ha="right")
    ax3.set_ylim(-0.05, 1.0)
    ax3.set_ylabel("Spearman correlation to GPT-5.2")
    ax3.set_title("Cross-model agreement relative to GPT-5.2")
    add_panel_label(ax3, "c")
    ax3.legend(frameon=False, loc="upper left")
    hide_spines(ax3)

    save(fig, "supp_fig_s11_model_extension_robustness")
    return metrics_df


def build_table_s7_metrics(model_extension_metrics: pd.DataFrame) -> None:
    macro_nd = pd.read_csv(GROUP_ROOT / "macro5_identity_neutral_distance.csv")
    macro_ipi = pd.read_csv(GROUP_ROOT / "macro5_identity_ipi.csv")
    micro_nd = pd.read_csv(GROUP_ROOT / "micro20_identity_neutral_distance.csv")
    micro_ipi = pd.read_csv(GROUP_ROOT / "micro20_identity_ipi.csv")
    prompt_comp = pd.read_csv(GROUP_ROOT / "meso7_prompt_form_neutral_distance_comparison.csv")
    prompt_ipi = clean_frame(pd.read_csv(GROUP_ROOT / "meso7_prompt_form_ipi.csv"))

    macro_nearest = (
        macro_nd.groupby("condition_key")["mean_pair_distance"].mean().sort_values().rename("value").reset_index()
    )
    macro_ipi_mean = (
        macro_ipi.groupby("condition_key")["ipi"].mean().sort_values(ascending=False).rename("value").reset_index()
    )
    micro_rank_mean = (
        micro_nd.groupby("condition_key")["rank_closest_to_neutral"].mean().sort_values().rename("value").reset_index()
    )
    micro_ipi_mean = (
        micro_ipi.groupby("condition_key")["ipi"].mean().sort_values(ascending=False).rename("value").reset_index()
    )

    prompt_original = prompt_ipi[prompt_ipi["model_label"] == "GPT-5.2"][["condition_key", "ipi"]].rename(columns={"ipi": "original"})
    prompt_weak = prompt_ipi[prompt_ipi["model_label"] == "GPT-5.2 weak"][["condition_key", "ipi"]].rename(columns={"ipi": "weak"})
    prompt_rho_ipi = prompt_original.merge(prompt_weak, on="condition_key")["original"].corr(
        prompt_original.merge(prompt_weak, on="condition_key")["weak"], method="spearman"
    )

    rows = [
        {
            "section": "macro5_identity",
            "metric": "Model-mean nearest-to-neutral macro group",
            "value": f"{MACRO_LABELS[macro_nearest.iloc[0]['condition_key']].replace(chr(10), ' ')} ({macro_nearest.iloc[0]['value']:.3f})",
        },
        {
            "section": "macro5_identity",
            "metric": "Model-mean strongest positive macro IPI",
            "value": f"{MACRO_LABELS[macro_ipi_mean.iloc[0]['condition_key']].replace(chr(10), ' ')} ({macro_ipi_mean.iloc[0]['value']:.3f})",
        },
        {
            "section": "micro20_identity",
            "metric": "Model-mean top three nearest-to-neutral micro regions",
            "value": "; ".join(
                f"{MICRO_LABELS[row.condition_key]} ({row.value:.2f})"
                for row in micro_rank_mean.head(3).itertuples(index=False)
            ),
        },
        {
            "section": "micro20_identity",
            "metric": "Model-mean strongest positive and negative micro IPI",
            "value": (
                f"{MICRO_LABELS[micro_ipi_mean.iloc[0]['condition_key']]} ({micro_ipi_mean.iloc[0]['value']:.3f}); "
                f"{MICRO_LABELS[micro_ipi_mean.iloc[-1]['condition_key']]} ({micro_ipi_mean.iloc[-1]['value']:.3f})"
            ),
        },
        {
            "section": "meso7_prompt_form",
            "metric": "Spearman correlation between original and weak prompt",
            "value": (
                f"Neutral distance {prompt_comp['mean_pair_distance_original'].corr(prompt_comp['mean_pair_distance_weak'], method='spearman'):.3f}; "
                f"IPI {prompt_rho_ipi:.3f}"
            ),
        },
        {
            "section": "meso7_model_extension",
            "metric": "Spearman correlation to GPT-5.2 by model family",
            "value": "; ".join(
                f"{row.model}: semantic {row.neutral_distance_spearman:.3f}, IPI {row.ipi_spearman:.3f}"
                for row in model_extension_metrics.itertuples(index=False)
            ),
        },
    ]

    pd.DataFrame(rows).to_csv(OUT_ROOT / "supp_table_s7_key_metrics.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    plot_s8_macro()
    plot_s9_micro()
    plot_s10_prompt_form()
    metrics_df = plot_s11_model_extension()
    build_table_s7_metrics(metrics_df)
    print(f"Saved robustness supplement figures to {OUT_ROOT}")


if __name__ == "__main__":
    main()
