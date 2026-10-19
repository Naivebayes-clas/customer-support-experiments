"""
Customer Support Platform: Full Experimentation Pipeline
========================================================
Runs all 5 sub-projects end-to-end:
  1. Data Generation
  2. Customer Journey Segmentation + Markov (Project 3)
  3. RAG Chatbot Evaluation (Project 1)
  4. A/B Testing (Project 2)
  5. Survival / Churn Analysis (Project 4)
  6. End-to-End Pipeline (Project 5)
  7. Integrated Dashboard
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from data_generator import (generate_support_platform, get_journey_transitions,
                            get_segment_features)
from segmentation import JourneySegmenter, MarkovChainAnalyzer, plot_segment_clusters
from rag_evaluator import run_rag_evaluation, plot_rag_metrics
from ab_tester import (PowerAnalyzer, ABTester, PropensityScoreMatcher,
                       SequentialMonitor)
from survival import ChurnSurvivalModel, plot_survival_curves, plot_cox_forest
from pipeline import MetricsPipeline
from visualizer import (plot_power_curve, plot_ci_bars, plot_dashboard, plot_sequential)

os.makedirs("outputs", exist_ok=True)

# ═══════════════════════════════════════════════════════
# 1. DATA GENERATION
# ═══════════════════════════════════════════════════════
print("=" * 65)
print("  CUSTOMER SUPPORT PLATFORM: FULL EXPERIMENTATION PIPELINE")
print("=" * 65)

print("\n[1/7] Generating synthetic data...")
tickets, conversations, churn = generate_support_platform(n_users=5000, true_lift=0.08)
print(f"  → {len(tickets)} tickets | {len(conversations)} conversation turns | {len(churn)} customers")
print(f"  → Groups: {tickets['group'].value_counts().to_dict()}")
print(f"  → Escalation rate: {tickets['escalated'].mean()*100:.1f}%")
print(f"  → Reopen rate: {tickets['reopened'].mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════
# 2. SEGMENTATION + MARKOV (Project 3)
# ═══════════════════════════════════════════════════════
print("\n[2/7] Customer Journey Segmentation + Markov Chain...")

# Segmentation
seg_features = get_segment_features(tickets)
segmenter = JourneySegmenter(n_clusters=4)
seg_features = segmenter.fit(seg_features)
seg_features = segmenter.label_segments(seg_features)
profile = segmenter.get_segment_profile(seg_features)
print(f"\n  Segment Profile:")
print(profile.to_string())
plot_segment_clusters(seg_features)

# Markov Chain
print(f"\n  Markov Chain Analysis:")
trans_all = get_journey_transitions(tickets)
trans_control = get_journey_transitions(tickets[tickets["group"] == "control"])
trans_treatment = get_journey_transitions(tickets[tickets["group"] == "treatment"])

markov = MarkovChainAnalyzer()
markov.fit(trans_all)
tm = markov.transition_matrix
print(f"  Direct resolution rate: {markov.direct_resolution_rate():.3f}")
print(f"  Expected path length: {markov.expected_path_length():.2f} steps")
markov.plot_transition_matrix()
markov.plot_path_comparison(trans_control, trans_treatment)

# Markov transition as A/B metric
p_ctrl_direct = trans_control["Open"]["In-Progress"] * trans_control["In-Progress"]["Resolved"]
p_treat_direct = trans_treatment["Open"]["In-Progress"] * trans_treatment["In-Progress"]["Resolved"]
print(f"  Direct resolution: Control={p_ctrl_direct:.3f} vs Treatment={p_treat_direct:.3f}")

# ═══════════════════════════════════════════════════════
# 3. RAG CHATBOT EVALUATION (Project 1)
# ═══════════════════════════════════════════════════════
print("\n[3/7] RAG Chatbot Multi-Turn Evaluation...")
rag_df = run_rag_evaluation(n_conversations=100, seed=42)
print(f"\n  RAGAS Metrics by Config:")
rag_summary = rag_df.groupby("config").agg({
    "faithfulness": "mean", "relevance": "mean", "correctness": "mean",
    "resolved": "mean", "n_turns": "mean"
}).round(3)
print(rag_summary.to_string())
plot_rag_metrics(rag_df)

# ═══════════════════════════════════════════════════════
# 4. A/B TESTING (Project 2)
# ═══════════════════════════════════════════════════════
print("\n[4/7] A/B Testing...")

# Power Analysis
pa = PowerAnalyzer(alpha=0.05, power=0.80)
n_prop = pa.sample_size_two_proportions(0.10, 0.108)
n_mean = pa.sample_size_two_means(0.2)
mde = pa.min_detectable_effect(len(tickets[tickets["group"]=="control"]))
print(f"  Power: n={n_prop:,} for 8% lift | n={n_mean:,} for d=0.2 | MDE={mde:.4f}")
plot_power_curve(range(100, 10000, 100), [0.1, 0.2, 0.5, 0.8])

# A/B Tests
tester = ABTester(alpha=0.05)
control = tickets[tickets["group"] == "control"]
treatment = tickets[tickets["group"] == "treatment"]

# Primary: resolution time
res_res_time = tester.test_continuous(
    control["resolution_time_hours"], treatment["resolution_time_hours"],
    name="resolution_time"
)
# Secondary: CSAT
res_csat = tester.test_continuous(
    control["csat"], treatment["csat"], name="csat"
)
# Secondary: escalation rate
res_esc = tester.test_binary(
    int(control["escalated"].sum()), len(control),
    int(treatment["escalated"].sum()), len(treatment),
    name="escalation_rate"
)
# Secondary: reopen rate
res_reopen = tester.test_binary(
    int(control["reopened"].sum()), len(control),
    int(treatment["reopened"].sum()), len(treatment),
    name="reopen_rate"
)

print(f"\n  A/B Test Results:")
for name, r in tester.results.items():
    sig = "✓" if r["significant"] else "✗"
    if "cohens_d" in r:
        print(f"    {name}: {r['control_mean']:.3f} vs {r['treatment_mean']:.3f} | "
              f"d={r['cohens_d']:.4f} | p={r['p_value']:.4f} {sig}")
    elif "control_rate" in r:
        print(f"    {name}: {r['control_rate']:.4f} vs {r['treatment_rate']:.4f} | "
              f"lift={r['relative_lift_pct']:.2f}% | p={r['p_value']:.4f} {sig}")

# Multiple comparison correction
p_vals = [r["p_value"] for r in tester.results.values()]
holm = tester.multiple_comparison_correction(p_vals, "holm")
print(f"\n  Holm-corrected p-values: {[f'{p:.4f}' for p in holm]}")

# Segment-level A/B test (heterogeneous treatment effect)
print(f"\n  Segment-Level A/B Test (Heterogeneous Treatment Effect):")
seg_tickets = tickets.merge(seg_features[["customer_id", "segment_label"]], on="customer_id")
for seg in seg_tickets["segment_label"].unique():
    seg_c = seg_tickets[(seg_tickets["segment_label"] == seg) & (seg_tickets["group"] == "control")]
    seg_t = seg_tickets[(seg_tickets["segment_label"] == seg) & (seg_tickets["group"] == "treatment")]
    if len(seg_c) > 10 and len(seg_t) > 10:
        res = tester.test_continuous(
            seg_c["resolution_time_hours"], seg_t["resolution_time_hours"],
            name=f"resolution_{seg}"
        )
        print(f"    {seg}: d={res['cohens_d']:.4f} | p={res['p_value']:.4f} "
              f"{'✓' if res['significant'] else '✗'}")

# Sequential monitoring
print(f"\n  Sequential Monitoring:")
monitor = SequentialMonitor(alpha=0.05, n_checks=5)
p_series = []
for i in range(1, 6):
    n_check = int(len(control) * i / 5)
    c = control["resolution_time_hours"][:n_check]
    t = treatment["resolution_time_hours"][:n_check]
    check = monitor.check(c, t, i)
    p_series.append(check["p_value"])
    print(f"    Check {i}/5: p={check['p_value']:.4f} → {check['recommendation']}")
plot_sequential(p_series, monitor.alpha_spent)

# CI plot
plot_ci_bars(tester.results)

# ═══════════════════════════════════════════════════════
# 5. SURVIVAL / CHURN (Project 4)
# ═══════════════════════════════════════════════════════
print("\n[5/7] Survival / Churn Analysis...")

survival = ChurnSurvivalModel()
c_churn = churn[churn["group"] == "control"]
t_churn = churn[churn["group"] == "treatment"]

# Kaplan-Meier
km_c, km_t = survival.fit_km(
    c_churn["time_to_churn_days"], c_churn["churned"],
    t_churn["time_to_churn_days"], t_churn["churned"]
)
plot_survival_curves(km_c, km_t)

# Log-rank
lr = survival.log_rank_test(
    c_churn["time_to_churn_days"], c_churn["churned"],
    t_churn["time_to_churn_days"], t_churn["churned"]
)
print(f"  Log-rank: p={lr['p_value']:.4f} {'✓ SIGNIFICANT' if lr['significant'] else '✗ NOT SIGNIFICANT'}")

# Cox PH
cox_df = churn.copy()
cox_df["treatment"] = (cox_df["group"] == "treatment").astype(int)
cox_df = cox_df.rename(columns={"time_to_churn_days": "duration", "churned": "event"})
# Only adjust for PRE-treatment covariates (n_tickets is assigned before treatment)
# avg_csat and n_escalations are POST-treatment (mediators) — adjusting for them
# blocks the causal pathway and biases HR toward 1.0
cox_df = cox_df[["duration", "event", "treatment", "n_tickets"]]
cox_model = survival.fit_cox(cox_df)
hr = survival.hazard_ratio("treatment")
print(f"  Cox PH (group): HR={hr['hazard_ratio']:.4f} "
      f"(95% CI: {hr['ci'][0]:.4f}-{hr['ci'][1]:.4f}) | p={hr['p_value']:.4f}")
print(f"  → Treatment {'reduces' if hr['hazard_ratio'] < 1 else 'increases'} "
      f"churn risk by {abs(1-hr['hazard_ratio'])*100:.1f}%")

# Survival at 30 days
s30 = survival.survival_at_time(30)
print(f"  30-day churn: Control={s30['churn_control']*100:.1f}% | Treatment={s30['churn_treatment']*100:.1f}%")

plot_cox_forest(cox_model)

# Add survival test to A/B tester
res_survival = tester.test_survival(
    c_churn["time_to_churn_days"], c_churn["churned"],
    t_churn["time_to_churn_days"], t_churn["churned"],
    name="churn_hazard"
)
print(f"  A/B Survival Test: HR={res_survival['hazard_ratio']:.4f} | "
      f"p={res_survival['logrank_p_value']:.4f} {'✓' if res_survival['significant'] else '✗'}")

# ═══════════════════════════════════════════════════════
# 6. END-TO-END PIPELINE (Project 5)
# ═══════════════════════════════════════════════════════
print("\n[6/7] End-to-End Metrics Pipeline...")
pipeline = MetricsPipeline()
pipeline.load_raw_data(tickets, conversations, churn)
analytics = pipeline.run_analytics()
pipeline.export_dashboard_data(analytics)
print(f"\n  Pipeline Analytics:")
print(f"  A/B Summary:\n{analytics['ab_summary'].to_string()}")
print(f"\n  Churn Summary:\n{analytics['churn_summary'].to_string()}")
pipeline.close()

# ═══════════════════════════════════════════════════════
# 7. INTEGRATED DASHBOARD
# ═══════════════════════════════════════════════════════
print("\n[7/7] Integrated Dashboard...")
plot_dashboard(tickets, churn, seg_features)

# Final summary
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"\n  {tester.summary_table().to_string()}")
print(f"\n  Outputs saved to: outputs/")
print(f"  Total plots: 10+ | CSVs: 5+ | Parquet: 3 | DuckDB: 1")
print("\n  DONE!")
print("=" * 65)  
