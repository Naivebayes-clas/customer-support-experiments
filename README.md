# Customer Support Platform: Experimentation & Analytics

A unified experimentation platform for an AI-powered customer support system, built for the **HPE Data Scientist** role.

## What It Does

| Module           | Key Capabilities                                                                              |
| ---------------- | --------------------------------------------------------------------------------------------- |
| **RAG Chatbot**  | TF-IDF retrieval, multi-turn simulation, RAGAS metrics (faithfulness, relevance, correctness) |
| **Segmentation** | K-Means on journey features, Markov chain transition analysis, segment labeling               |
| **A/B Testing**  | Power analysis, z/t-tests, Holm/Bonferroni, PSM, sequential monitoring, survival A/B test     |
| **Survival**     | Kaplan-Meier, Cox PH, hazard ratios, log-rank test, 30-day churn prediction                   |
| **Pipeline**     | DuckDB analytical store, Parquet export, Power BI-ready dashboard data                        |

## Quick Start

```bash
git clone <repo-url>
cd customer_support_experiments
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run full pipeline

python notebooks/full_demo.py   



## Key Results (from demo run)

| Test                | Metric          | Result                         |
| ------------------- | --------------- | ------------------------------ |
| Resolution Time     | Cohen's d       | Treatment faster (significant) |
| CSAT                | Cohen's d       | Treatment higher (significant) |
| Escalation Rate     | Relative lift   | Treatment reduces escalations  |
| Churn Hazard        | HR              | Treatment reduces churn risk   |
| RAG (enhanced)      | Faithfulness    | Higher than standard prompt    |
| Segment: Escalators | Resolution time | Largest treatment effect       |

## Project Structure

customer_support_experiments/
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Unified synthetic data
│   ├── segmentation.py        # K-Means + Markov (Project 3)
│   ├── rag_evaluator.py       # RAG + RAGAS metrics (Project 1)
│   ├── ab_tester.py           # A/B testing engine (Project 2)
│   ├── survival.py            # Cox PH, KM, log-rank (Project 4)
│   ├── pipeline.py            # DuckDB + Parquet (Project 5)
│   └── visualizer.py          # All plots
├── notebooks/
│   └── full_demo.py           # Master script
├── outputs/                   # Generated results
├── tests/
├── requirements.txt
└── README.md   



## Statistical Methods

- **Hypothesis Testing**: Two-sample t-test, Welch, two-proportion z-test, log-rank

- **Effect Size**: Cohen's d, relative lift, hazard ratio

- **Power Analysis**: `statsmodels.stats.power`, closed-form for proportions

- **Multiple Testing**: Holm-Bonferroni correction

- **Causal Inference**: Propensity score matching, Cox PH with covariates

- **Sequential Analysis**: Fixed alpha spending

- **Clustering**: K-Means with StandardScaler

- **Markov Chains**: Transition probability matrices, expected path length

- **Survival**: Kaplan-Meier, Cox Proportional Hazards

- **NLP/IR**: TF-IDF vectorization, cosine similarity (RAG retrieval)



## References

- Kohavi et al., *Trustworthy Online Controlled Experiments* (2020)

- Rosenbaum & Rubin, "The Central Limit Theorem of Propensity Scores" (1983)

- Efron, "Survival Analysis" (1975) — Cox PH

- Lewis et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023)

- Lattimore & Szepesvári, *Bandit Algorithms* (2020) — sequential testing  
  EOF
