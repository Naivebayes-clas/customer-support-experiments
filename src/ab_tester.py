"""
Project 2: A/B Testing & Experimentation Framework
- Power Analysis
- A/B Test (z-test, t-test, Welch)
- Multiple Comparison Correction
- Propensity Score Matching
- Sequential Monitoring
- Survival A/B Test (log-rank + Cox PH)
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.power import TTestIndPower
import warnings
warnings.filterwarnings("ignore")


class PowerAnalyzer:
    """Compute sample size and power for A/B tests."""

    def __init__(self, alpha=0.05, power=0.8):
        self.alpha = alpha
        self.power = power

    def sample_size_two_proportions(self, p1, p2, alpha=None, power=None):
        alpha = alpha or self.alpha
        power = power or self.power
        p_pool = (p1 + p2) / 2
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        n = ((z_alpha * np.sqrt(2 * p_pool * (1 - p_pool)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2
        return int(np.ceil(n))

    def sample_size_two_means(self, effect_size, alpha=None, power=None):
        alpha = alpha or self.alpha
        power = power or self.power
        analysis = TTestIndPower()
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha,
                                  power=power, ratio=1, alternative='two-sided')
        return int(np.ceil(n))

    def post_hoc_power(self, n_per_group, effect_size, alpha=None):
        alpha = alpha or self.alpha
        analysis = TTestIndPower()
        return analysis.solve_power(effect_size=effect_size, nobs1=n_per_group,
                                     alpha=alpha, ratio=1, alternative='two-sided')

    def min_detectable_effect(self, n_per_group, alpha=None, power=None):
        alpha = alpha or self.alpha
        power = power or self.power
        analysis = TTestIndPower()
        return analysis.solve_power(nobs1=n_per_group, alpha=alpha,
                                     power=power, ratio=1, alternative='two-sided')


class ABTester:
    """Core A/B testing engine."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.results = {}

    def test_continuous(self, control, treatment, equal_variance=True, name="metric"):
        control = np.array(control, dtype=float)
        treatment = np.array(treatment, dtype=float)
        n1, n2 = len(control), len(treatment)

        t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=equal_variance)
        pooled_std = np.sqrt(((n1 - 1) * control.std(ddof=1)**2 +
                              (n2 - 1) * treatment.std(ddof=1)**2) / (n1 + n2 - 2))
        cohens_d = (treatment.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0

        diff = treatment.mean() - control.mean()
        se = np.sqrt(control.var(ddof=1)/n1 + treatment.var(ddof=1)/n2)
        ci_lower = diff - norm.ppf(1 - self.alpha/2) * se
        ci_upper = diff + norm.ppf(1 - self.alpha/2) * se

        result = {
            "metric": name, "test": "t-test" if equal_variance else "Welch t-test",
            "control_mean": round(float(control.mean()), 4),
            "treatment_mean": round(float(treatment.mean()), 4),
            "diff": round(float(diff), 4),
            "ci_lower": round(float(ci_lower), 4), "ci_upper": round(float(ci_upper), 4),
            "t_stat": round(float(t_stat), 4), "p_value": round(float(p_val), 6),
            "cohens_d": round(float(cohens_d), 4),
            "significant": bool(p_val < self.alpha),
            "n_control": n1, "n_treatment": n2
        }
        self.results[name] = result
        return result

    def test_binary(self, control_successes, control_total,
                    treatment_successes, treatment_total, name="conversion"):
        p1 = control_successes / control_total
        p2 = treatment_successes / treatment_total
        p_pool = (control_successes + treatment_successes) / (control_total + treatment_total)

        se = np.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))
        z_stat = (p2 - p1) / se if se > 0 else 0
        p_val = 2 * (1 - norm.cdf(abs(z_stat)))

        diff = p2 - p1
        se_diff = np.sqrt(p1*(1-p1)/control_total + p2*(1-p2)/treatment_total)
        ci_lower = diff - norm.ppf(1 - self.alpha/2) * se_diff
        ci_upper = diff + norm.ppf(1 - self.alpha/2) * se_diff
        lift = (p2 - p1) / p1 * 100 if p1 != 0 else 0

        result = {
            "metric": name, "test": "two-proportion z-test",
            "control_rate": round(p1, 4), "treatment_rate": round(p2, 4),
            "diff": round(diff, 4), "relative_lift_pct": round(lift, 2),
            "ci_lower": round(ci_lower, 4), "ci_upper": round(ci_upper, 4),
            "z_stat": round(float(z_stat), 4), "p_value": round(float(p_val), 6),
            "significant": bool(p_val < self.alpha),
            "n_control": control_total, "n_treatment": treatment_total
        }
        self.results[name] = result
        return result

    def test_survival(self, control_times, control_events,
                     treatment_times, treatment_events, name="churn"):
        """Log-rank test + Cox PH hazard ratio."""
        from lifelines import CoxPHFitter
        from lifelines.statistics import logrank_test

        lr = logrank_test(control_times, treatment_times,
                          control_events, treatment_events)

        df = pd.DataFrame({
            'T': [0]*len(control_times) + [1]*len(treatment_times),
            'duration': list(control_times) + list(treatment_times),
            'event': list(control_events) + list(treatment_events)
        })
        ph = CoxPHFitter().fit(df, duration_col='duration', event_col='event')
        summary = ph.summary
        hr = summary.loc['T', "exp(coef)"]
        lower_col = [c for c in summary.columns if "lower" in c.lower() and "exp" in c.lower()][0]
        upper_col = [c for c in summary.columns if "upper" in c.lower() and "exp" in c.lower()][0]
        hr_ci = [summary.loc['T', lower_col], summary.loc['T', upper_col]]

        result = {
            "metric": name, "test": "log-rank + Cox PH",
            "hazard_ratio": round(float(hr), 4),
            "hr_ci_lower": round(float(hr_ci[0]), 4),
            "hr_ci_upper": round(float(hr_ci[1]), 4),
            "logrank_p_value": round(float(lr.p_value), 6),
            "significant": bool(lr.p_value < self.alpha),
            "interpretation": f"HR={hr:.3f} -> treatment {'reduces' if hr < 1 else 'increases'} churn risk by {abs(1-hr)*100:.1f}%",
            "n_control": len(control_times), "n_treatment": len(treatment_times)
        }
        self.results[name] = result
        return result

    def multiple_comparison_correction(self, p_values, method="holm"):
        p = np.array(p_values)
        m = len(p)
        if method == "bonferroni":
            return np.minimum(p * m, 1.0)
        elif method == "holm":
            sorted_idx = np.argsort(p)
            sorted_p = p[sorted_idx]
            corrected = np.zeros(m)
            for i in range(m):
                corrected[sorted_idx[i]] = (m - i) * sorted_p[i]
            corrected = np.minimum(corrected, 1.0)
            for i in range(1, m):
                corrected[sorted_idx[i]] = max(corrected[sorted_idx[i]],
                                                 corrected[sorted_idx[i-1]])
            return corrected
        raise ValueError("method must be 'bonferroni' or 'holm'")

    def summary_table(self):
        if not self.results:
            return pd.DataFrame()
        rows = []
        for name, r in self.results.items():
            p_val = r.get("p_value", r.get("logrank_p_value", "N/A"))
            row = {"Metric": r["metric"], "Test": r["test"],
                   "p-value": p_val, "Significant": "✓" if r["significant"] else "✗"}
            if "control_mean" in r:
                row["Control"] = r["control_mean"]
                row["Treatment"] = r["treatment_mean"]
                row["Effect"] = f"d={r['cohens_d']}"
            elif "control_rate" in r:
                row["Control"] = r["control_rate"]
                row["Treatment"] = r["treatment_rate"]
                row["Effect"] = f"lift={r['relative_lift_pct']}%"
            elif "hazard_ratio" in r:
                row["Effect"] = f"HR={r['hazard_ratio']}"
            rows.append(row)
        return pd.DataFrame(rows)


class PropensityScoreMatcher:
    """Propensity score matching for observational data."""

    def __init__(self, caliper=0.1):
        self.caliper = caliper
        self.scores_ = None

    def fit(self, X, treatment):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, treatment)
        self.scores_ = model.predict_proba(X_scaled)[:, 1]
        return self.scores_

    def match(self, X, treatment):
        if self.scores_ is None:
            self.fit(X, treatment)
        t_mask = np.array(treatment) == 1
        c_mask = np.array(treatment) == 0
        t_idx = np.where(t_mask)[0]
        c_idx = np.where(c_mask)[0]
        t_scores = self.scores_[t_idx]
        c_scores = self.scores_[c_idx]

        matched_t, matched_c = [], []
        for i, ts in enumerate(t_scores):
            distances = np.abs(c_scores - ts)
            if distances.min() <= self.caliper:
                matched_t.append(t_idx[i])
                matched_c.append(c_idx[distances.argmin()])
        return np.array(matched_t), np.array(matched_c)

    def ate(self, outcome, X, treatment):
        t_idx, c_idx = self.match(X, treatment)
        if len(t_idx) == 0:
            return None, (0, 0)
        t_out = np.array(outcome)[t_idx]
        c_out = np.array(outcome)[c_idx]
        ate = t_out.mean() - c_out.mean()
        se = np.sqrt(t_out.var(ddof=1)/len(t_out) + c_out.var(ddof=1)/len(c_out))
        return ate, (1.96*se, len(t_idx))


class SequentialMonitor:
    """Sequential monitoring with alpha spending."""

    def __init__(self, alpha=0.05, n_checks=5):
        self.alpha = alpha
        self.n_checks = n_checks
        self.alpha_spent = alpha / n_checks

    def check(self, control, treatment, check_number):
        t_stat, p_val = stats.ttest_ind(treatment, control)
        significant = p_val < self.alpha_spent
        return {
            "check": check_number, "p_value": round(float(p_val), 6),
            "alpha_spent": round(self.alpha_spent, 6),
            "can_stop": significant,
            "recommendation": "STOP" if significant else "CONTINUE"
        }
