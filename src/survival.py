"""
Project 4: Survival / Hazard Model for Churn
- Kaplan-Meier estimator
- Cox Proportional Hazards
- Log-rank test
- Hazard ratio A/B testing
- Survival curves by segment
"""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class ChurnSurvivalModel:
    """Survival analysis for customer churn."""

    def __init__(self):
        self.km_control = None
        self.km_treatment = None
        self.cox_model = None

    def fit_km(self, control_times, control_events, treatment_times, treatment_events):
        """Fit Kaplan-Meier survival curves."""
        kmf_c = KaplanMeierFitter()
        kmf_t = KaplanMeierFitter()
        self.km_control = kmf_c.fit(control_times, control_events, label="Control")
        self.km_treatment = kmf_t.fit(treatment_times, treatment_events, label="Treatment")
        return self.km_control, self.km_treatment

    def fit_cox(self, df):
        """
        Fit Cox PH model.
        df must have columns: duration, event, and covariates (including 'group' as 0/1)
        """
        self.cox_model = CoxPHFitter()
        covariates = [c for c in df.columns if c not in ['duration', 'event']]
        self.cox_model.fit(df, duration_col='duration', event_col='event')
        return self.cox_model

    def hazard_ratio(self, covariate="group"):
        """Get hazard ratio for a covariate."""
        if self.cox_model is None:
            raise ValueError("Call fit_cox() first")
        summary = self.cox_model.summary
        hr = summary.loc[covariate, "exp(coef)"]
        # Find CI columns (handles different lifelines versions)
        lower_col = [c for c in summary.columns if "lower" in c.lower() and "exp" in c.lower()][0]
        upper_col = [c for c in summary.columns if "upper" in c.lower() and "exp" in c.lower()][0]
        ci = [summary.loc[covariate, lower_col], summary.loc[covariate, upper_col]]
        p = summary.loc[covariate, "p"]
        return {"hazard_ratio": round(float(hr), 4),
                "ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
                "p_value": round(float(p), 6),
                "significant": bool(p < 0.05)}

    def log_rank_test(self, control_times, control_events,
                      treatment_times, treatment_events):
        """Log-rank test for survival difference."""
        lr = logrank_test(control_times, treatment_times,
                          control_events, treatment_events)
        return {"p_value": round(float(lr.p_value), 6),
                "significant": bool(lr.p_value < 0.05),
                "median_survival_control": float(self.km_control.median_survival_time_ if self.km_control else np.nan),
                "median_survival_treatment": float(self.km_treatment.median_survival_time_ if self.km_treatment else np.nan)}

    def survival_at_time(self, t=30):
        """Survival probability at time t for both groups."""
        if self.km_control is None:
            raise ValueError("Call fit_km() first")
        try:
            s_c = float(self.km_control.predict(t))
        except:
            s_c = 1.0
        try:
            s_t = float(self.km_treatment.predict(t))
        except:
            s_t = 1.0
        return {"control": round(s_c, 4), "treatment": round(s_t, 4),
                "churn_control": round(1 - s_c, 4), "churn_treatment": round(1 - s_t, 4)}


def plot_survival_curves(km_control, km_treatment, save_path="outputs/survival_curves.png"):
    """Plot Kaplan-Meier curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    km_control.plot_survival_function(ax=ax, color="steelblue", linewidth=2)
    km_treatment.plot_survival_function(ax=ax, color="coral", linewidth=2)
    ax.set_xlabel("Days")
    ax.set_ylabel("P(not churned)")
    ax.set_title("Kaplan-Meier Survival Curves: Churn")
    ax.legend(loc="lower left")
    # Add log-rank p-value annotation
    from lifelines.statistics import logrank_test
    # Access the data used to fit (stored on the fitter objects)
    lr = logrank_test(km_control.durations, km_treatment.durations,
                      km_control.event_observed, km_treatment.event_observed)
    ax.text(0.95, 0.05, f"Log-rank p = {lr.p_value:.4f}\n"
            f"{'Significant (p < 0.05)' if lr.p_value < 0.05 else 'Not significant'}",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_path}")


def plot_cox_forest(cox_model, save_path="outputs/cox_forest.png"):
    """Forest plot of Cox PH hazard ratios."""
    summary = cox_model.summary
    hr = summary["exp(coef)"]
    lower_col = [c for c in summary.columns if "lower" in c.lower() and "exp" in c.lower()][0]
    upper_col = [c for c in summary.columns if "upper" in c.lower() and "exp" in c.lower()][0]

    fig, ax = plt.subplots(figsize=(10, max(4, len(hr) * 0.6)))
    y_pos = range(len(hr))

    for i, var in enumerate(hr.index):
        ci_lower = summary.loc[var, lower_col]
        ci_upper = summary.loc[var, upper_col]
        ax.plot([ci_lower, ci_upper], [i, i], 'o-', color='steelblue', linewidth=2)
        ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
        ax.text(ci_upper + 0.05, i, f"HR={hr[var]:.3f}", va='center', fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(hr.index)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title("Cox PH: Hazard Ratios")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_path}")
   
