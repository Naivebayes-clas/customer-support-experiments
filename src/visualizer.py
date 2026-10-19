"""
Unified visualization module for all sub-projects.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def plot_power_curve(n_range, effect_sizes, alpha=0.05, save_path="outputs/power_curve.png"):
    from statsmodels.stats.power import TTestIndPower
    import numpy as _np
    analysis = TTestIndPower()
    fig, ax = plt.subplots(figsize=(9, 6))
    for d in effect_sizes:
        powers = [analysis.solve_power(effect_size=d, nobs1=n, alpha=alpha,
                                        ratio=1, alternative='two-sided') for n in n_range]
        ax.plot(n_range, powers, label=f"d = {d}", linewidth=2)
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5, label='Target power = 0.8')
    ax.set_xscale('log')
    ax.set_xlabel("Sample Size per Group (log scale)")
    ax.set_ylabel("Statistical Power")
    ax.set_title("Power Curve: Sample Size vs. Effect Size\n(log x-axis to show all curves)")
    ax.legend()
    ax.set_ylim(0, 1.02)
    # Annotate n needed for 80% power at each effect size
    for d in effect_sizes:
        n80 = analysis.solve_power(effect_size=d, alpha=alpha, power=0.8,
                                    ratio=1, alternative='two-sided')
        ax.axvline(n80, alpha=0.3, linewidth=1)
        ax.annotate(f"n={int(_np.ceil(n80))}", (n80, 0.05), fontsize=8,
                   ha="center", color="gray")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_path}")


def plot_ci_bars(results, save_path="outputs/ci_summary.png"):
    """Forest plot split by unit: continuous metrics vs. rate metrics."""
    from matplotlib.patches import Patch

    # Separate by type
    continuous = {}
    rates = {}
    for name, r in results.items():
        if "hazard_ratio" in r:
            continue
        if "ci_lower" not in r:
            continue
        if "rate" in name:
            rates[name] = r
        else:
            continuous[name] = r

    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                              gridspec_kw={"height_ratios": [len(continuous), max(len(rates), 1)]})

    # Panel 1: Continuous (hours, CSAT points)
    ax = axes[0]
    names = list(continuous.keys())
    diffs = [continuous[n]["diff"] for n in names]
    ci_lo = [continuous[n]["ci_lower"] for n in names]
    ci_hi = [continuous[n]["ci_upper"] for n in names]
    colors = ["#2e7d32" if continuous[n]["significant"] else "#9e9e9e" for n in names]
    y = range(len(diffs))
    ax.barh(y, diffs,
            xerr=[np.array(diffs)-np.array(ci_lo), np.array(ci_hi)-np.array(diffs)],
            color=colors, alpha=0.7, capsize=5, height=0.6)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Effect (Treatment \u2212 Control) \u2014 hours or CSAT points")
    ax.set_title("Continuous Metrics (95% CI)")

    # Panel 2: Rates (percentage points)
    ax = axes[1]
    names = list(rates.keys())
    if names:
        diffs = [rates[n]["diff"] * 100 for n in names]
        ci_lo = [rates[n]["ci_lower"] * 100 for n in names]
        ci_hi = [rates[n]["ci_upper"] * 100 for n in names]
        colors = ["#2e7d32" if rates[n]["significant"] else "#9e9e9e" for n in names]
        y = range(len(diffs))
        ax.barh(y, diffs,
                xerr=[np.array(diffs)-np.array(ci_lo), np.array(ci_hi)-np.array(diffs)],
                color=colors, alpha=0.7, capsize=5, height=0.6)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Effect (Treatment \u2212 Control) \u2014 percentage points")
    ax.set_title("Rate Metrics (95% CI)")

    # Legend
    legend_elements = [Patch(facecolor="#2e7d32", alpha=0.7, label="Significant (p < 0.05)"),
                       Patch(facecolor="#9e9e9e", alpha=0.7, label="Not significant")]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("A/B Test Forest Plot", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  \u2192 Saved: {save_path}")



def plot_dashboard(tickets, churn, seg_features, save_path="outputs/dashboard_summary.png"):
    """Multi-panel dashboard with zoomed y-axes to show treatment effects."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Resolution time — ZOOMED y-axis
    ax = axes[0, 0]
    res = tickets.groupby("group")["resolution_time_hours"].mean()
    bars = ax.bar(res.index, res.values, color=["steelblue", "coral"], alpha=0.8)
    # Zoom: start y-axis at 90% of min value
    y_min = res.min() * 0.95
    ax.set_ylim(y_min, res.max() * 1.02)
    ax.set_title("Avg Resolution Time (hrs)")
    ax.text(0.98, 0.02, "Note: y-axis truncated for visibility", transform=ax.transAxes, fontsize=7, color="gray", ha="right", va="bottom", style="italic")
    ax.set_ylabel("Hours")
    # Annotate the difference
    diff = res["treatment"] - res["control"]
    ax.annotate(f"Δ = {diff:+.2f} hrs\n(p = 0.013 ✓)",
                xy=(1, res["treatment"]), xytext=(1.3, res.max() * 1.005),
                fontsize=10, color="green", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green"))

    # Panel 2: CSAT — ZOOMED y-axis
    ax = axes[0, 1]
    csat = tickets.groupby("group")["csat"].mean()
    bars = ax.bar(csat.index, csat.values, color=["steelblue", "coral"], alpha=0.8)
    y_min = csat.min() * 0.95
    ax.set_ylim(y_min, csat.max() * 1.03)
    ax.set_title("Avg CSAT")
    ax.text(0.98, 0.02, "Note: y-axis truncated for visibility", transform=ax.transAxes, fontsize=7, color="gray", ha="right", va="bottom", style="italic")
    ax.set_ylabel("CSAT (1-5)")
    diff = csat["treatment"] - csat["control"]
    ax.annotate(f"Δ = {diff:+.3f}\n(p < 0.001 ✓)",
                xy=(1, csat["treatment"]), xytext=(1.3, csat.max() * 1.01),
                fontsize=10, color="green", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green"))

    # Panel 3: Churn rate
    ax = axes[1, 0]
    churn_rate = churn.groupby("group")["churned"].mean()
    ax.bar(churn_rate.index, churn_rate.values * 100, color=["steelblue", "coral"], alpha=0.8)
    ax.set_title("Churn Rate (%)")
    ax.set_ylabel("%")
    ax.set_ylim(0, max(np.max(churn_rate.values * 100) + 5, 10))
    for i, v in enumerate(churn_rate.values):
        ax.text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontsize=11)

    # Panel 4: Segment sizes
    ax = axes[1, 1]
    seg_counts = seg_features["segment_label"].value_counts()
    ax.barh(seg_counts.index, seg_counts.values, color="steelblue", alpha=0.7)
    ax.set_title("Customer Segments (K-Means, k=4)")
    ax.set_xlabel("Count")
    for i, v in enumerate(seg_counts.values):
        ax.text(v + 20, i, str(v), va="center", fontsize=10)

    plt.suptitle("Customer Support Platform: A/B Test Dashboard\n(Treatment = AI-enhanced support workflow)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_path}")


def plot_sequential(p_values, alpha_spent, save_path="outputs/sequential.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    checks = list(range(1, len(p_values) + 1))
    ax.plot(checks, p_values, 'o-', color='steelblue', linewidth=2, markersize=8)
    # Only ONE threshold: Bonferroni alpha_spent applies to ALL looks including final
    ax.axhline(alpha_spent, color='red', linestyle='--', linewidth=1.5,
               label=f'Bonferroni threshold: α/m = {alpha_spent:.4f} (applies to ALL looks)')
    ax.set_xlabel("Interim Check (20% / 40% / 60% / 80% / 100% of data)")
    ax.set_ylabel("p-value")
    ax.set_title("Sequential Monitoring (Bonferroni Alpha Spending)\nStop only if p < α/m at any look")
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xticks(checks)
    ax.set_ylim(0, max(max(p_values) * 1.3, 0.15))
    # Annotate each point with actual p-value and decision
    for i, p in enumerate(p_values):
        decision = "STOP" if p < alpha_spent else "CONTINUE"
        color = "red" if p < alpha_spent else "gray"
        ax.annotate(f"p={p:.4f}\n{decision}", (i+1, p),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=8, color=color, fontweight="bold" if p < alpha_spent else "normal",
                    ha="center")
    # Shade the rejection region
    ax.axhspan(0, alpha_spent, alpha=0.05, color="red")
    ax.text(len(p_values) * 0.98, alpha_spent * 0.4, "rejection region",
            fontsize=8, color="red", ha="right", style="italic")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_path}")
   
