from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MAIN_DATA = ROOT / "source_data" / "main_figures"
SUPP_DATA = ROOT / "source_data" / "supplement"
OUT = ROOT / "figures" / "supplement"

MODEL_ORDER = ["GPT-5.2", "Claude Sonnet 4", "Gemini 2.5 Flash"]
MODEL_COLORS = {
    "GPT-5.2": "#4d80bc",
    "Claude Sonnet 4": "#76789b",
    "Gemini 2.5 Flash": "#c9758e",
}
MODEL_MARKERS = {
    "GPT-5.2": "o",
    "Claude Sonnet 4": "^",
    "Gemini 2.5 Flash": "s",
}
MODEL_KEY_TO_LABEL = {
    "gpt52": "GPT-5.2",
    "claude_sonnet4": "Claude Sonnet 4",
    "gemini25flash": "Gemini 2.5 Flash",
}
REGION_ORDER = ["NEU", "ENA", "CSA", "NAWA", "ESEA", "SSA", "LAC", "OCE"]
REGION_ORDER_NO_NEU = ["ENA", "CSA", "NAWA", "ESEA", "SSA", "LAC", "OCE"]
DIM_ORDER = ["wealthy", "safe", "beautiful", "depressing", "lively", "boring"]
AXIS_ORDER = ["gender", "age", "country"]


def model_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D([0], [0], marker=MODEL_MARKERS[m], linestyle="none", markerfacecolor=MODEL_COLORS[m], markeredgecolor="none", label=m)
        for m in MODEL_ORDER
    ]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def annotate_panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_s1_sentiment() -> None:
    cond = pd.read_csv(
        SUPP_DATA
        / "main_neutral_meso7_sentiment_siebert_v1"
        / "condition_summary_combined.csv"
    )
    shift = pd.read_csv(
        SUPP_DATA
        / "main_neutral_meso7_sentiment_siebert_v1"
        / "neutral_shift_summary_combined.csv"
    )
    cond = cond[cond["text_variant"].eq("stripped")].copy()
    shift = shift[shift["text_variant"].eq("stripped")].copy()
    cond["model_label"] = pd.Categorical(cond["model_label"], MODEL_ORDER, ordered=True)
    cond["condition_abbr"] = pd.Categorical(cond["condition_abbr"], REGION_ORDER, ordered=True)
    shift["model_label"] = pd.Categorical(shift["model_label"], MODEL_ORDER, ordered=True)
    shift["condition_abbr"] = pd.Categorical(shift["condition_abbr"], REGION_ORDER_NO_NEU, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    x = np.arange(len(REGION_ORDER))
    offsets = np.linspace(-0.18, 0.18, len(MODEL_ORDER))
    for off, model in zip(offsets, MODEL_ORDER):
        sub = cond[cond["model_label"].eq(model)].sort_values("condition_abbr")
        y = sub["mean_score"].to_numpy()
        yerr = 1.96 * sub["se_score"].to_numpy()
        ax.errorbar(
            x + off,
            y,
            yerr=yerr,
            fmt=MODEL_MARKERS[model],
            ms=4,
            lw=1,
            capsize=2,
            color=MODEL_COLORS[model],
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER, rotation=35, ha="right")
    ax.set_ylabel("Mean sentiment")
    ax.set_title("Affective level by prompt")
    ax.grid(axis="y", color="#e6e2da", lw=0.6)
    annotate_panel(ax, "a")

    ax = axes[1]
    x = np.arange(len(REGION_ORDER_NO_NEU))
    for off, model in zip(offsets, MODEL_ORDER):
        sub = shift[shift["model_label"].eq(model)].sort_values("condition_abbr")
        ax.scatter(
            x + off,
            sub["mean_shift_vs_neutral"],
            s=20,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            label=model,
            zorder=3,
        )
        ax.vlines(
            x + off,
            0,
            sub["mean_shift_vs_neutral"],
            color=MODEL_COLORS[model],
            lw=1,
            alpha=0.75,
        )
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER_NO_NEU, rotation=35, ha="right")
    ax.set_ylabel("Mean shift vs neutral")
    ax.set_title("Prompt-induced sentiment shift")
    ax.grid(axis="y", color="#e6e2da", lw=0.6)
    annotate_panel(ax, "b")
    fig.legend(handles=model_handles(), frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.52, 1.04))
    fig.subplots_adjust(top=0.82, wspace=0.35)
    save(fig, "supp_fig_s1_sentiment_identity.pdf")


def plot_s2_open_semantics() -> None:
    neutral = pd.read_csv(MAIN_DATA / "Fig1" / "upstream_analysis" / "neutral_distance_combined.csv")
    pca = pd.read_csv(
        MAIN_DATA
        / "Fig1"
        / "c_local_pca"
        / "figure_source"
        / "fig2c_pca_local_scatter_source.csv"
    )
    ipi = pd.read_csv(MAIN_DATA / "Fig1" / "upstream_analysis" / "ipi_region_summary.csv")
    neutral["model_label"] = neutral["model_key"].map(MODEL_KEY_TO_LABEL)
    pca["model_label"] = pca["model_key"].map(MODEL_KEY_TO_LABEL)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), gridspec_kw={"width_ratios": [1.05, 1, 1.2]})

    ax = axes[0]
    rank = (
        neutral.pivot_table(
            index="model_label",
            columns="condition_abbr",
            values="rank_closest_to_neutral",
            aggfunc="first",
        )
        .reindex(MODEL_ORDER)[REGION_ORDER_NO_NEU]
        .astype(float)
    )
    im = ax.imshow(rank.to_numpy(), cmap="YlGnBu_r", vmin=1, vmax=7, aspect="auto")
    ax.set_xticks(np.arange(len(REGION_ORDER_NO_NEU)))
    ax.set_xticklabels(REGION_ORDER_NO_NEU, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    for i in range(rank.shape[0]):
        for j in range(rank.shape[1]):
            ax.text(j, i, f"{rank.iloc[i, j]:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title("Rank closest to neutral")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Rank")
    annotate_panel(ax, "a")

    ax = axes[1]
    disp_rows = []
    for model, sub in pca.groupby("model_label"):
        neu = sub[sub["condition_abbr"].eq("NEU")][["pc1", "pc2"]].iloc[0].to_numpy()
        for _, row in sub[~sub["condition_abbr"].eq("NEU")].iterrows():
            disp_rows.append(
                {
                    "model_label": model,
                    "condition_abbr": row["condition_abbr"],
                    "pca_displacement": float(np.linalg.norm(row[["pc1", "pc2"]].to_numpy() - neu)),
                }
            )
    disp = pd.DataFrame(disp_rows)
    x = np.arange(len(REGION_ORDER_NO_NEU))
    offsets = np.linspace(-0.18, 0.18, len(MODEL_ORDER))
    for off, model in zip(offsets, MODEL_ORDER):
        sub = (
            disp[disp["model_label"].eq(model)]
            .set_index("condition_abbr")
            .reindex(REGION_ORDER_NO_NEU)
            .reset_index()
        )
        ax.scatter(x + off, sub["pca_displacement"], s=20, color=MODEL_COLORS[model], marker=MODEL_MARKERS[model], label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER_NO_NEU, rotation=35, ha="right")
    ax.set_ylabel("PCA displacement from neutral")
    ax.set_title("Centroid displacement")
    ax.grid(axis="y", color="#e6e2da", lw=0.6)
    annotate_panel(ax, "b")

    ax = axes[2]
    merged = neutral.merge(
        ipi[ipi["text_variant"].eq("stripped")],
        left_on=["model_key", "condition_abbr"],
        right_on=["model_key", "region_abbr"],
        how="inner",
    )
    for model in MODEL_ORDER:
        sub = merged[merged["model_label_x"].map(MODEL_KEY_TO_LABEL).fillna(merged["model_label_x"]).eq(model)]
        if sub.empty:
            sub = merged[merged["model_label_y"].eq(model)]
        ax.scatter(
            sub["mean_pair_distance"],
            sub["ipi"],
            s=28,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            alpha=0.85,
            label=model,
        )
    for abbr in ["ENA", "OCE", "NAWA", "SSA"]:
        sub_lab = merged[merged["condition_abbr"].eq(abbr)]
        if sub_lab.empty:
            continue
        row = sub_lab.iloc[0]
        ax.text(row["mean_pair_distance"] + 0.001, row["ipi"], abbr, fontsize=6.5, alpha=0.75)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Semantic distance to neutral")
    ax.set_ylabel("Sentiment IPI")
    ax.set_title("Semantic distance vs affective preference")
    ax.grid(color="#e6e2da", lw=0.6)
    annotate_panel(ax, "c")
    fig.legend(handles=model_handles(), frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 1.04))
    fig.subplots_adjust(top=0.82, wspace=0.55)
    save(fig, "supp_fig_s2_open_semantic_diagnostics.pdf")


def plot_s3_geograph_benchmark() -> None:
    condition = pd.read_csv(
        MAIN_DATA
        / "Fig2"
        / "upstream_analysis"
        / "uk_open_1000_semantics"
        / "condition_summary_three_models.csv"
    )
    dispersion = pd.read_csv(
        MAIN_DATA
        / "Fig2"
        / "upstream_analysis"
        / "uk_open_1000_semantics"
        / "dispersion_summary_three_models.csv"
    )
    distinct = pd.read_csv(
        MAIN_DATA
        / "Fig2"
        / "c_distinct2"
        / "figure_source"
        / "uk_open_1000_distinct2_neutral_hum_bar_source.csv"
    )
    sentiment = pd.read_csv(
        MAIN_DATA
        / "Fig2"
        / "d_sentiment_forest"
        / "figure_source"
        / "uk_open_1000_sentiment_neutral_hum_forest_source.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [0.9, 1.2]})

    ax = axes[0]
    gains = []
    for model in MODEL_ORDER:
        sub = condition[condition["model_label"].eq(model)].set_index("condition_key")
        gains.append(
            {
                "model": model,
                "delta_neutral_minus_uk": sub.loc["neutral", "mean"] - sub.loc["united_kingdom", "mean"],
            }
        )
    gains = pd.DataFrame(gains)
    y = np.arange(len(MODEL_ORDER))
    ax.barh(y, gains["delta_neutral_minus_uk"], color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    ax.set_yticks(y)
    ax.set_yticklabels(MODEL_ORDER)
    ax.invert_yaxis()
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Neutral distance minus UK-prompt distance")
    ax.set_title("UK prompt gain")
    ax.grid(axis="x", color="#e6e2da", lw=0.6)
    annotate_panel(ax, "a")

    ax = axes[1]
    human_disp = dispersion[dispersion["key"].eq("human")].iloc[0]["mean_pairwise_distance"]
    human_distinct = distinct[distinct["set_key"].eq("HUM")].iloc[0]["value"]
    human_sent = sentiment[sentiment["series"].eq("HUM")].iloc[0]["mean_score"]
    ratios = []
    for model in MODEL_ORDER:
        model_short = "Claude" if model == "Claude Sonnet 4" else ("Gemini" if model == "Gemini 2.5 Flash" else model)
        disp_val = dispersion[
            dispersion["model_label"].eq(model) & dispersion["key"].eq("neutral")
        ].iloc[0]["mean_pairwise_distance"]
        distinct_key = f"{model}::NEU"
        dist_rows = distinct[distinct["set_key"].eq(distinct_key)]
        if dist_rows.empty:
            dist_rows = distinct[distinct["label"].eq(model_short)]
        sent_val = sentiment[sentiment["series"].eq(model)].iloc[0]["mean_score"]
        ratios.extend(
            [
                {"model": model, "metric": "Semantic dispersion", "ratio": disp_val / human_disp},
                {"model": model, "metric": "Distinct-2", "ratio": dist_rows.iloc[0]["value"] / human_distinct},
                {"model": model, "metric": "Sentiment", "ratio": sent_val / human_sent},
            ]
        )
    ratios = pd.DataFrame(ratios)
    metrics = ["Semantic dispersion", "Distinct-2", "Sentiment"]
    x = np.arange(len(metrics))
    offsets = np.linspace(-0.22, 0.22, len(MODEL_ORDER))
    for off, model in zip(offsets, MODEL_ORDER):
        sub = ratios[ratios["model"].eq(model)].set_index("metric").reindex(metrics)
        ax.bar(x + off, sub["ratio"], width=0.19, color=MODEL_COLORS[model], label=model)
    ax.axhline(1, color="#555555", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Dispersion", "Distinct-2", "Sentiment"], rotation=20, ha="right")
    ax.set_ylabel("LLM / human ratio")
    ax.set_title("Compression and affective elevation")
    ax.grid(axis="y", color="#e6e2da", lw=0.6)
    fig.legend(handles=model_handles(), frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.56, 1.04))
    fig.subplots_adjust(top=0.82, wspace=0.35)
    annotate_panel(ax, "b")
    save(fig, "supp_fig_s3_geograph_benchmark_diagnostics.pdf")


def plot_s4_geograph_spatial() -> None:
    admin = pd.read_csv(
        SUPP_DATA
        / "study2_uk_open_1000_sentiment_gap_neutral_admin1_maps_v1"
        / "admin1_summary.csv"
    )
    admin = admin[admin["eligible"].eq(True)].dropna(subset=["mean_gap"]).copy()

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.2))
    geos = ["England", "Scotland", "Wales"]
    positions = np.arange(len(geos))
    offsets = np.linspace(-0.22, 0.22, len(MODEL_ORDER))
    rng = np.random.default_rng(20260419)
    for off, model in zip(offsets, MODEL_ORDER):
        for i, geo in enumerate(geos):
            vals = admin[(admin["series"].eq(model)) & (admin["geonunit"].eq(geo))]["mean_gap"].to_numpy()
            if len(vals) == 0:
                continue
            jitter = rng.normal(0, 0.025, size=len(vals))
            ax.scatter(
                np.full(len(vals), positions[i] + off) + jitter,
                vals,
                s=8,
                alpha=0.35,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                edgecolors="none",
            )
            ax.scatter(
                positions[i] + off,
                np.mean(vals),
                s=42,
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(geos)
    ax.set_ylabel("Admin-region mean sentiment gap")
    ax.set_title("Spatial spread of neutral LLM-human sentiment gap")
    ax.grid(axis="y", color="#e6e2da", lw=0.6)
    annotate_panel(ax, "a")

    fig.legend(handles=model_handles(), frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.52, 1.04))
    fig.subplots_adjust(top=0.80)
    save(fig, "supp_fig_s4_geograph_spatial_gap_diagnostics.pdf")


def plot_s5_structured_agreement() -> None:
    scores = pd.read_csv(
        MAIN_DATA
        / "Fig3"
        / "upstream_analysis"
        / "structured_sdunits"
        / "scores_all_models.csv"
    )
    scores = scores[scores["condition_label"].eq("NEU")].copy()
    long = scores.melt(
        id_vars=["model_label", "stable_image_key"],
        value_vars=DIM_ORDER,
        var_name="dimension",
        value_name="score",
    )
    summary = (
        long.groupby(["model_label", "dimension"])["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    x = np.arange(len(DIM_ORDER))
    offsets = np.linspace(-0.22, 0.22, len(MODEL_ORDER))
    for off, model in zip(offsets, MODEL_ORDER):
        sub = summary[summary["model_label"].eq(model)].set_index("dimension").reindex(DIM_ORDER)
        ax.errorbar(
            x + off,
            sub["mean"],
            yerr=1.96 * sub["se"],
            fmt=MODEL_MARKERS[model],
            ms=4,
            lw=1,
            capsize=2,
            color=MODEL_COLORS[model],
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DIM_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Neutral structured score")
    ax.set_title("Neutral score levels")
    ax.grid(axis="y", color="#e6e2da", lw=0.6)
    annotate_panel(ax, "a")

    ax = axes[1]
    corr_rows = []
    for dim in DIM_ORDER:
        sub = scores[["stable_image_key", "model_label", dim]].pivot_table(
            index="stable_image_key", columns="model_label", values=dim, aggfunc="mean"
        )
        for m1, m2 in [("GPT-5.2", "Claude Sonnet 4"), ("GPT-5.2", "Gemini 2.5 Flash"), ("Claude Sonnet 4", "Gemini 2.5 Flash")]:
            corr_rows.append({"dimension": dim, "pair": f"{m1.split()[0]}-{m2.split()[0]}", "rho": sub[m1].corr(sub[m2], method="spearman")})
    corr = pd.DataFrame(corr_rows)
    pairs = ["GPT-5.2-Claude", "GPT-5.2-Gemini", "Claude-Gemini"]
    corr["pair"] = corr["pair"].replace({"GPT-5.2-Claude": "GPT-Claude", "GPT-5.2-Gemini": "GPT-Gemini"})
    pairs = ["GPT-Claude", "GPT-Gemini", "Claude-Gemini"]
    mat = corr.pivot(index="pair", columns="dimension", values="rho").reindex(pairs)[DIM_ORDER]
    im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="Greens", aspect="auto")
    ax.set_xticks(np.arange(len(DIM_ORDER)))
    ax.set_xticklabels([d.capitalize() for d in DIM_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pairs)))
    ax.set_yticklabels(pairs)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Cross-model Spearman rho")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    annotate_panel(ax, "b")
    fig.legend(handles=model_handles(), frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.52, 1.04))
    fig.subplots_adjust(top=0.82, wspace=0.38)
    save(fig, "supp_fig_s5_structured_agreement.pdf")


def plot_s6_place_pulse() -> None:
    fit = pd.read_csv(
        MAIN_DATA
        / "Fig4"
        / "upstream_analysis"
        / "place_pulse_vs_structured"
        / "fit_summary.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, value, title, cmap, limits, letter in [
        (axes[0], "spearman_rho", "Rank agreement", "YlGnBu", (0, 0.7), "a"),
        (axes[1], "slope", "Linear score slope", "Oranges", (0, 0.45), "b"),
    ]:
        mat = (
            fit.pivot(index="model_label", columns="dimension", values=value)
            .reindex(MODEL_ORDER)[DIM_ORDER]
            .astype(float)
        )
        im = ax.imshow(mat.to_numpy(), cmap=cmap, vmin=limits[0], vmax=limits[1], aspect="auto")
        ax.set_xticks(np.arange(len(DIM_ORDER)))
        ax.set_xticklabels([d.capitalize() for d in DIM_ORDER], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(MODEL_ORDER)))
        ax.set_yticklabels(MODEL_ORDER)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        annotate_panel(ax, letter)
    save(fig, "supp_fig_s6_place_pulse_grounding_summary.pdf")


def plot_s7_pairwise_replication() -> None:
    dim = pd.read_csv(
        SUPP_DATA
        / "specs_dimension_breakdown_v1"
        / "dimension_breakdown_summary.csv"
    )
    margin = pd.read_csv(
        MAIN_DATA
        / "Fig4"
        / "upstream_analysis"
        / "margin_stratification"
        / "margin_stratification_summary.csv"
    )
    qscore = pd.read_csv(
        MAIN_DATA
        / "Fig4"
        / "upstream_analysis"
        / "qscore_comparison"
        / "qscore_summary.csv"
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), gridspec_kw={"width_ratios": [1.05, 1.1, 1]})

    ax = axes[0]
    dim_summary = (
        dim.groupby("dimension")[["llm_divergence_rate", "strict_replication_rate"]]
        .mean()
        .reindex(DIM_ORDER)
    )
    y = np.arange(len(DIM_ORDER))
    ax.barh(y - 0.16, dim_summary["llm_divergence_rate"], height=0.28, color="#b7685c", label="Divergence")
    ax.barh(y + 0.16, dim_summary["strict_replication_rate"], height=0.28, color="#5f8a5f", label="Strict replication")
    ax.set_yticks(y)
    ax.set_yticklabels([d.capitalize() for d in DIM_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Mean rate")
    ax.set_title("Dimension-level rates")
    ax.grid(axis="x", color="#e6e2da", lw=0.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.52, 1.20), ncol=2)
    annotate_panel(ax, "a")

    ax = axes[1]
    margin_summary = (
        margin.groupby(["axis", "margin_bin"])["strict_replication_rate"]
        .mean()
        .reset_index()
    )
    margin_order = ["Low", "Mid", "High"]
    x = np.arange(len(margin_order))
    axis_colors = {"gender": "#2f6f9f", "age": "#9b5f7f", "country": "#5f8a5f"}
    for axis in AXIS_ORDER:
        sub = margin_summary[margin_summary["axis"].eq(axis)].set_index("margin_bin").reindex(margin_order)
        ax.plot(x, sub["strict_replication_rate"], marker="o", lw=1.5, color=axis_colors[axis], label=axis.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(margin_order)
    ax.set_ylabel("Strict replication rate")
    ax.set_title("Replication by human-margin bin")
    ax.grid(axis="y", color="#e6e2da", lw=0.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.50, 1.20), ncol=3)
    annotate_panel(ax, "b")

    ax = axes[2]
    mat = (
        qscore.pivot(index="axis", columns="model", values="spearman_rho")
        .reindex(AXIS_ORDER)[MODEL_ORDER]
        .astype(float)
    )
    im = ax.imshow(mat.to_numpy(), cmap="PuBuGn", vmin=0, vmax=0.35, aspect="auto")
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels(["GPT-5.2", "Claude", "Gemini"], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(AXIS_ORDER)))
    ax.set_yticklabels([a.capitalize() for a in AXIS_ORDER])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("qscore Spearman rho")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    annotate_panel(ax, "c")
    fig.subplots_adjust(top=0.78, wspace=0.45)
    save(fig, "supp_fig_s7_pairwise_replication_diagnostics.pdf")


def write_manifest() -> None:
    rows = [
        ("supp_fig_s1_sentiment_identity.pdf", "main_neutral_meso7_sentiment_siebert_v1/condition_summary_combined.csv; neutral_shift_summary_combined.csv"),
        ("supp_fig_s2_open_semantic_diagnostics.pdf", "Fig1 neutral_distance_combined.csv; fig2c_pca_local_scatter_source.csv; ipi_region_summary.csv"),
        ("supp_fig_s3_geograph_benchmark_diagnostics.pdf", "Fig2 condition_summary_three_models.csv; dispersion_summary_three_models.csv; distinct2 source; sentiment source"),
        ("supp_fig_s4_geograph_spatial_gap_diagnostics.pdf", "study2_uk_open_1000_sentiment_gap_neutral_admin1_maps_v1/admin1_summary.csv"),
        ("supp_fig_s5_structured_agreement.pdf", "global_structured_sixdim_3000_sdunits_v1/scores_all_models.csv"),
        ("supp_fig_s6_place_pulse_grounding_summary.pdf", "Fig4 place_pulse_vs_structured/fit_summary.csv"),
        ("supp_fig_s7_pairwise_replication_diagnostics.pdf", "specs_dimension_breakdown_v1/dimension_breakdown_summary.csv; margin_stratification_summary.csv; qscore_summary.csv"),
    ]
    pd.DataFrame(rows, columns=["figure_file", "source_data"]).to_csv(OUT / "supplement_figure_manifest.csv", index=False)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_s1_sentiment()
    plot_s2_open_semantics()
    plot_s3_geograph_benchmark()
    plot_s4_geograph_spatial()
    plot_s5_structured_agreement()
    plot_s6_place_pulse()
    plot_s7_pairwise_replication()
    write_manifest()
    print(f"Wrote curated supplementary figures to {OUT}")


if __name__ == "__main__":
    main()
