"""
=============================================================
  EDA — AI Health Risk Prediction System (Indian Population)
=============================================================
Datasets used (synthetic, Indian-population-calibrated):
  - PIMA Indians Diabetes-inspired features
  - Framingham Heart Study-inspired features
  - BMI / Obesity features with Indian cutoffs

Run (from project root):
  python scripts/eda_health.py

Outputs:
  reports/figures/eda_health_report.png
  data/processed/health_dataset_india.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Reproducibility ───────────────────────────────────────
np.random.seed(42)
N = 1000  # synthetic patients


def ensure_dirs():
    """Create output directories if they don't exist."""
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)


def generate_synthetic_dataset(n: int = 1000) -> pd.DataFrame:
    """Generate a synthetic health dataset calibrated for an Indian-like population profile."""
    age = np.random.normal(42, 13, n).clip(18, 80).astype(int)
    bmi = np.random.normal(26.5, 5, n).clip(14, 48).round(1)  # higher mean for India
    sugar = np.random.normal(105, 30, n).clip(60, 400).round(1)
    bpSys = np.random.normal(128, 18, n).clip(80, 200).round(0)
    bpDia = np.random.normal(83, 12, n).clip(50, 130).round(0)

    smoking = np.random.choice([0, 1], n, p=[0.72, 0.28])
    sedentary = np.random.choice([0, 1], n, p=[0.45, 0.55])
    stress = np.random.choice([0, 1], n, p=[0.50, 0.50])
    alcohol = np.random.choice([0, 1], n, p=[0.75, 0.25])
    exercise = np.random.choice([0, 1], n, p=[0.60, 0.40])
    fam_hist = np.random.choice([0, 1], n, p=[0.55, 0.45])
    gender = np.random.choice(["Male", "Female"], n, p=[0.52, 0.48])

    # Rule-based labels (realistic thresholds)
    diabetes_risk = (
        (sugar >= 126).astype(int) * 2
        + (bmi >= 25).astype(int)
        + (age >= 45).astype(int)
        + fam_hist
        + sedentary
        + stress
    )
    diabetes = (diabetes_risk >= 3).astype(int)

    heart_risk = (
        (bpSys >= 140).astype(int) * 2
        + smoking
        + (age >= 50).astype(int)
        + fam_hist
        + stress
        + alcohol
        + (bmi >= 25).astype(int)
    )
    heart_disease = (heart_risk >= 3).astype(int)

    # Indian BMI obesity threshold = 25 (not 30)
    obesity = (bmi >= 25).astype(int)

    df = pd.DataFrame(
        {
            "Age": age,
            "Gender": gender,
            "BMI": bmi,
            "BloodSugar": sugar,
            "SystolicBP": bpSys,
            "DiastolicBP": bpDia,
            "Smoking": smoking,
            "Sedentary": sedentary,
            "Stress": stress,
            "Alcohol": alcohol,
            "Exercise": exercise,
            "FamilyHistory": fam_hist,
            "Diabetes": diabetes,
            "HeartDisease": heart_disease,
            "Obesity": obesity,
        }
    )
    return df


def print_basic_stats(df: pd.DataFrame):
    print("=" * 60)
    print("  DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Shape          : {df.shape}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"\n  Diabetes prevalence  : {df['Diabetes'].mean() * 100:.1f}%")
    print(f"  Heart disease prev.  : {df['HeartDisease'].mean() * 100:.1f}%")
    print(f"  Obesity prevalence   : {df['Obesity'].mean() * 100:.1f}%")
    print("\n  Numerical summary:")
    print(df[["Age", "BMI", "BloodSugar", "SystolicBP", "DiastolicBP"]].describe().round(2))
    print("=" * 60)


def build_eda_figure(df: pd.DataFrame, out_path: str):
    # ── Styling ────────────────────────────────────────────
    DARK = "#0e1512"
    CARD = "#162019"
    GREEN = "#2dca6e"
    ORANGE = "#f7a84a"
    RED = "#e05c5c"
    BLUE = "#5ba4e5"
    PURPLE = "#a78bfa"
    MUTED = "#6b8870"
    TEXT = "#dde8df"

    plt.rcParams.update(
        {
            "figure.facecolor": DARK,
            "axes.facecolor": CARD,
            "axes.edgecolor": "#1e301f",
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "grid.color": "#1e301f",
            "grid.linestyle": "--",
            "font.family": "monospace",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(24, 30), facecolor=DARK)
    fig.suptitle(
        "ArogyaAI  ·  Exploratory Data Analysis\nHealth Risk Prediction — Indian Population  (n=1000)",
        fontsize=20,
        fontweight="bold",
        color=GREEN,
        y=0.98,
        linespacing=1.6,
        fontfamily="monospace",
    )

    gs = gridspec.GridSpec(
        5, 4, figure=fig, hspace=0.52, wspace=0.38, top=0.94, bottom=0.04, left=0.06, right=0.97
    )

    def style_ax(ax, title):
        ax.set_title(title, fontsize=10, color=TEXT, pad=8, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    # ── Row 0 : Prevalence + Gender split + Age distributions ──────────────
    ax0 = fig.add_subplot(gs[0, 0])
    prev = [df["Diabetes"].mean() * 100, df["HeartDisease"].mean() * 100, df["Obesity"].mean() * 100]
    bars = ax0.bar(["Diabetes", "Heart\nDisease", "Obesity"], prev, color=[RED, ORANGE, GREEN], width=0.5, zorder=3)
    for b, v in zip(bars, prev):
        ax0.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9, color=TEXT, fontweight="bold")
    ax0.set_ylim(0, max(prev) * 1.25)
    ax0.set_ylabel("Prevalence (%)", fontsize=8)
    style_ax(ax0, "① Disease Prevalence")

    ax1 = fig.add_subplot(gs[0, 1])
    gdf = df.groupby("Gender")[["Diabetes", "HeartDisease", "Obesity"]].mean() * 100
    x = np.arange(3)
    w = 0.35
    ax1.bar(x - w / 2, gdf.loc["Male"], width=w, color=BLUE, label="Male", zorder=3)
    ax1.bar(x + w / 2, gdf.loc["Female"], width=w, color=PURPLE, label="Female", zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Diabetes", "Heart\nDisease", "Obesity"], fontsize=8)
    ax1.set_ylabel("Prevalence (%)", fontsize=8)
    ax1.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax1, "② Gender vs Disease Risk")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df[df["Diabetes"] == 0]["Age"], bins=20, alpha=0.7, color=GREEN, label="No Diabetes", zorder=3)
    ax2.hist(df[df["Diabetes"] == 1]["Age"], bins=20, alpha=0.7, color=RED, label="Diabetes", zorder=3)
    ax2.set_xlabel("Age", fontsize=8)
    ax2.set_ylabel("Count", fontsize=8)
    ax2.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax2, "③ Age Distribution by Diabetes")

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.hist(df[df["HeartDisease"] == 0]["Age"], bins=20, alpha=0.7, color=GREEN, label="No Heart Dis.", zorder=3)
    ax3.hist(df[df["HeartDisease"] == 1]["Age"], bins=20, alpha=0.7, color=ORANGE, label="Heart Disease", zorder=3)
    ax3.set_xlabel("Age", fontsize=8)
    ax3.set_ylabel("Count", fontsize=8)
    ax3.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax3, "④ Age Distribution by Heart Disease")

    # ── Row 1 : BMI & Blood Sugar + BP scatter ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    for label, grp, col in [("No Diabetes", 0, GREEN), ("Diabetes", 1, RED)]:
        ax4.hist(df[df["Diabetes"] == grp]["BMI"], bins=25, alpha=0.7, color=col, label=label, zorder=3)
    # Indian BMI reference lines
    for line, lbl in [(18.5, "Underweight"), (23, "Normal"), (25, "Overweight")]:
        ax4.axvline(line, color=MUTED, lw=1, ls="--", zorder=4)
        ymax = ax4.get_ylim()[1] if ax4.get_ylim()[1] > 0 else 10
        ax4.text(line + 0.3, ymax * 0.9, lbl, fontsize=6.5, color=MUTED, rotation=90, va="top")
    ax4.set_xlabel("BMI", fontsize=8)
    ax4.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax4, "⑤ BMI Distribution (Indian cutoffs)")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(
        df["BloodSugar"], df["BMI"],
        c=df["Diabetes"].map({0: GREEN, 1: RED}),
        alpha=0.4, s=15, zorder=3
    )
    ax5.axvline(100, color=ORANGE, lw=1, ls="--", label="Pre-diabetic (100)")
    ax5.axvline(126, color=RED, lw=1, ls="--", label="Diabetic (126)")
    ax5.axhline(25, color=MUTED, lw=1, ls=":", label="BMI 25 (Indian)")
    ax5.set_xlabel("Blood Sugar (mg/dL)", fontsize=8)
    ax5.set_ylabel("BMI", fontsize=8)
    ax5.legend(fontsize=6.5, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax5, "⑥ Blood Sugar vs BMI (Red=Diabetic)")

    ax6 = fig.add_subplot(gs[1, 2])
    cats = pd.cut(df["BloodSugar"], bins=[0, 99, 125, 400], labels=["Normal", "Pre-diabetic", "Diabetic"])
    counts = cats.value_counts().sort_index()
    wedges, texts, autotexts = ax6.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=[GREEN, ORANGE, RED],
        textprops={"fontsize": 8, "color": TEXT},
        wedgeprops={"edgecolor": DARK, "linewidth": 2},
        startangle=90
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax6.set_title("⑦ Blood Sugar Categories", fontsize=10, color=TEXT, pad=8, fontweight="bold")

    ax7 = fig.add_subplot(gs[1, 3])
    ax7.scatter(
        df["SystolicBP"], df["DiastolicBP"],
        c=df["HeartDisease"].map({0: GREEN, 1: ORANGE}),
        alpha=0.4, s=15, zorder=3
    )
    ax7.axvline(130, color=ORANGE, lw=1, ls="--", label="Stage 1 Hyp (130)")
    ax7.axvline(140, color=RED, lw=1, ls="--", label="Stage 2 Hyp (140)")
    ax7.axhline(85, color=MUTED, lw=1, ls=":", label="Dia 85")
    ax7.set_xlabel("Systolic BP", fontsize=8)
    ax7.set_ylabel("Diastolic BP", fontsize=8)
    ax7.legend(fontsize=6.5, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax7, "⑧ BP Scatter (Orange=Heart Disease)")

    # ── Row 2 : Box plots + Risk by age group ──────────────────────────────
    ax8 = fig.add_subplot(gs[2, 0])
    data_box = [df[df["Diabetes"] == 0]["BloodSugar"], df[df["Diabetes"] == 1]["BloodSugar"]]
    bp = ax8.boxplot(data_box, patch_artist=True, widths=0.5, medianprops={"color": DARK, "linewidth": 2})
    for patch, col in zip(bp["boxes"], [GREEN, RED]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for part in ["whiskers", "caps", "fliers"]:
        for item in bp[part]:
            item.set_color(MUTED)
    ax8.set_xticklabels(["No Diabetes", "Diabetes"], fontsize=8)
    ax8.set_ylabel("Blood Sugar (mg/dL)", fontsize=8)
    ax8.axhline(126, color=RED, lw=1, ls="--", alpha=0.6)
    style_ax(ax8, "⑨ Blood Sugar Boxplot")

    ax9 = fig.add_subplot(gs[2, 1])
    data_box2 = [df[df["HeartDisease"] == 0]["SystolicBP"], df[df["HeartDisease"] == 1]["SystolicBP"]]
    bp2 = ax9.boxplot(data_box2, patch_artist=True, widths=0.5, medianprops={"color": DARK, "linewidth": 2})
    for patch, col in zip(bp2["boxes"], [GREEN, ORANGE]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for part in ["whiskers", "caps", "fliers"]:
        for item in bp2[part]:
            item.set_color(MUTED)
    ax9.set_xticklabels(["No Heart Dis.", "Heart Disease"], fontsize=8)
    ax9.set_ylabel("Systolic BP (mmHg)", fontsize=8)
    ax9.axhline(140, color=RED, lw=1, ls="--", alpha=0.6)
    style_ax(ax9, "⑩ Systolic BP Boxplot")

    ax10 = fig.add_subplot(gs[2, 2])
    data_box3 = [df[df["Obesity"] == 0]["BMI"], df[df["Obesity"] == 1]["BMI"]]
    bp3 = ax10.boxplot(data_box3, patch_artist=True, widths=0.5, medianprops={"color": DARK, "linewidth": 2})
    for patch, col in zip(bp3["boxes"], [GREEN, ORANGE]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for part in ["whiskers", "caps", "fliers"]:
        for item in bp3[part]:
            item.set_color(MUTED)
    ax10.set_xticklabels(["Not Obese", "Obese"], fontsize=8)
    ax10.set_ylabel("BMI", fontsize=8)
    style_ax(ax10, "⑪ BMI Boxplot by Obesity")

    ax11 = fig.add_subplot(gs[2, 3])
    age_bins = pd.cut(df["Age"], bins=[18, 30, 40, 50, 60, 80], labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    age_risk = df.groupby(age_bins, observed=True)[["Diabetes", "HeartDisease", "Obesity"]].mean() * 100
    x = np.arange(len(age_risk))
    w = 0.28
    ax11.bar(x - w, age_risk["Diabetes"], width=w, color=RED, label="Diabetes", zorder=3)
    ax11.bar(x, age_risk["HeartDisease"], width=w, color=ORANGE, label="Heart", zorder=3)
    ax11.bar(x + w, age_risk["Obesity"], width=w, color=GREEN, label="Obesity", zorder=3)
    ax11.set_xticks(x)
    ax11.set_xticklabels(age_risk.index, fontsize=7, rotation=15)
    ax11.set_ylabel("Prevalence (%)", fontsize=8)
    ax11.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax11, "⑫ Risk by Age Group")

    # ── Row 3 : Correlation heatmap + Lifestyle correlations ──────────────────────────────
    ax12 = fig.add_subplot(gs[3, 0:2])
    num_cols = [
        "Age", "BMI", "BloodSugar", "SystolicBP", "DiastolicBP",
        "Smoking", "Sedentary", "Stress", "Alcohol", "Exercise", "FamilyHistory",
        "Diabetes", "HeartDisease", "Obesity"
    ]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(10, 145, s=85, l=35, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, ax=ax12, cmap=cmap, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 6.5, "color": TEXT},
        linewidths=0.4, linecolor=DARK,
        cbar_kws={"shrink": 0.7, "aspect": 20}
    )
    ax12.set_xticklabels(ax12.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax12.set_yticklabels(ax12.get_yticklabels(), rotation=0, fontsize=7)
    ax12.set_title("⑬ Correlation Heatmap — All Features", fontsize=10, color=TEXT, pad=8, fontweight="bold")
    ax12.tick_params(colors=MUTED)

    ax13 = fig.add_subplot(gs[3, 2])
    lifestyle_cols = ["Smoking", "Sedentary", "Stress", "Alcohol", "Exercise", "FamilyHistory"]
    labels_ls = ["Smoking", "Sedentary", "Stress", "Alcohol", "Exercise", "Family Hist."]
    risk_corr_d = [df[c].corr(df["Diabetes"]) for c in lifestyle_cols]
    risk_corr_h = [df[c].corr(df["HeartDisease"]) for c in lifestyle_cols]
    x = np.arange(len(lifestyle_cols))
    w = 0.35
    ax13.barh(x - w / 2, risk_corr_d, height=w, color=RED, label="Diabetes", zorder=3)
    ax13.barh(x + w / 2, risk_corr_h, height=w, color=ORANGE, label="Heart Disease", zorder=3)
    ax13.set_yticks(x)
    ax13.set_yticklabels(labels_ls, fontsize=8)
    ax13.axvline(0, color=MUTED, lw=0.8)
    ax13.set_xlabel("Correlation Coefficient", fontsize=8)
    ax13.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    ax13.grid(axis="x", alpha=0.3)
    style_ax(ax13, "⑭ Lifestyle Correlations with Disease")

    ax14 = fig.add_subplot(gs[3, 3])
    ls_prevalence = {lbl: df[df[c] == 1]["Diabetes"].mean() * 100 for c, lbl in zip(lifestyle_cols, labels_ls)}
    y_vals = list(ls_prevalence.values())
    y_lbls = list(ls_prevalence.keys())
    avg = df["Diabetes"].mean() * 100
    colors14 = [RED if v > avg else GREEN for v in y_vals]
    bars14 = ax14.barh(y_lbls, y_vals, color=colors14, zorder=3)
    ax14.axvline(avg, color=MUTED, lw=1.2, ls="--", label=f"Avg {avg:.1f}%")
    for b, v in zip(bars14, y_vals):
        ax14.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7.5, color=TEXT)
    ax14.set_xlabel("Diabetes Prevalence (%)", fontsize=8)
    ax14.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    ax14.grid(axis="x", alpha=0.3)
    style_ax(ax14, "⑮ Diabetes % by Lifestyle Factor")

    # ── Row 4 : Violin + Outliers + Proxy Feature Importance ──────────────────────────────
    ax15 = fig.add_subplot(gs[4, 0])
    parts = ax15.violinplot(
        [df[df["Diabetes"] == 0]["BloodSugar"].values, df[df["Diabetes"] == 1]["BloodSugar"].values],
        positions=[0, 1],
        showmedians=True,
        showextrema=True,
    )
    for pc, col in zip(parts["bodies"], [GREEN, RED]):
        pc.set_facecolor(col)
        pc.set_alpha(0.65)
    for part in ["cmedians", "cbars", "cmins", "cmaxes"]:
        parts[part].set_color(TEXT)
    ax15.set_xticks([0, 1])
    ax15.set_xticklabels(["No Diabetes", "Diabetes"], fontsize=8)
    ax15.set_ylabel("Blood Sugar (mg/dL)", fontsize=8)
    ax15.axhline(126, color=RED, lw=1, ls="--", alpha=0.7, label="Diabetic threshold")
    ax15.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax15, "⑯ Blood Sugar Violin Plot")

    ax16 = fig.add_subplot(gs[4, 1])
    parts2 = ax16.violinplot(
        [df[df["HeartDisease"] == 0]["Age"].values, df[df["HeartDisease"] == 1]["Age"].values],
        positions=[0, 1],
        showmedians=True,
    )
    for pc, col in zip(parts2["bodies"], [GREEN, ORANGE]):
        pc.set_facecolor(col)
        pc.set_alpha(0.65)
    for part in ["cmedians", "cbars", "cmins", "cmaxes"]:
        parts2[part].set_color(TEXT)
    ax16.set_xticks([0, 1])
    ax16.set_xticklabels(["No Heart Dis.", "Heart Disease"], fontsize=8)
    ax16.set_ylabel("Age (years)", fontsize=8)
    style_ax(ax16, "⑰ Age Violin — Heart Disease")

    # Outlier analysis (fixed text placement)
    ax17 = fig.add_subplot(gs[4, 2])
    out_feats = ["BloodSugar", "BMI", "SystolicBP"]
    out_cols = [RED, GREEN, ORANGE]
    out_pcts = []
    for feat in out_feats:
        q1, q3 = df[feat].quantile([0.25, 0.75])
        iqr = q3 - q1
        out_pct = ((df[feat] < q1 - 1.5 * iqr) | (df[feat] > q3 + 1.5 * iqr)).mean() * 100
        out_pcts.append(out_pct)

    for i, (feat, col, pct) in enumerate(zip(out_feats, out_cols, out_pcts)):
        ax17.barh(i, pct, color=col, zorder=3)
        ax17.text(pct + 0.3, i, f"{pct:.1f}%", va="center", fontsize=8, color=TEXT)

    ax17.set_yticks([0, 1, 2])
    ax17.set_yticklabels(out_feats, fontsize=8)
    ax17.set_xlabel("Outlier % (IQR method)", fontsize=8)
    ax17.grid(axis="x", alpha=0.3)
    style_ax(ax17, "⑱ Outlier Analysis")

    ax18 = fig.add_subplot(gs[4, 3])
    feats = [
        "BloodSugar", "BMI", "Age", "SystolicBP", "FamilyHistory",
        "Sedentary", "Stress", "Smoking", "Exercise", "DiastolicBP"
    ]
    fi_vals = [abs(df[f].corr(df["Diabetes"])) for f in feats]
    fi_df = pd.DataFrame({"feat": feats, "imp": fi_vals}).sort_values("imp")
    bar_cols = [GREEN if v > 0.15 else MUTED for v in fi_df["imp"]]
    ax18.barh(fi_df["feat"], fi_df["imp"], color=bar_cols, zorder=3)
    ax18.axvline(0.15, color=ORANGE, lw=1, ls="--", label="Sig. threshold 0.15")
    ax18.set_xlabel("|Correlation| with Diabetes", fontsize=8)
    ax18.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    ax18.grid(axis="x", alpha=0.3)
    style_ax(ax18, "⑲ Proxy Feature Importance (Diabetes)")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK, edgecolor="none")
    plt.close(fig)


def main():
    # Ensure relative paths work even if script is run from scripts/
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    ensure_dirs()
    df = generate_synthetic_dataset(N)
    print_basic_stats(df)

    fig_path = "reports/figures/eda_health_report.png"
    csv_path = "data/processed/health_dataset_india.csv"

    build_eda_figure(df, fig_path)
    df.to_csv(csv_path, index=False)

    print(f"\n✅ EDA figure saved   → {fig_path}")
    print(f"✅ Dataset saved      → {csv_path}")
    print(f"\nColumns: {list(df.columns)}")


if __name__ == "__main__":
    main()