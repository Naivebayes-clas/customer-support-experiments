"""
Project 3: Customer Journey Segmentation
- K-Means clustering on customer behavior
- Markov chain transition analysis
- Segment labeling
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class JourneySegmenter:
    """Segment customers by support journey behavior."""

    def __init__(self, n_clusters=4, seed=42):
        self.n_clusters = n_clusters
        self.seed = seed
        self.kmeans = None
        self.scaler = None
        self.labels_ = None

    def fit(self, seg_features):
        """
        seg_features: DataFrame from get_segment_features()
        Clusters on: n_tickets, avg_resolution_time, n_escalations, n_reopens, avg_csat
        """
        feature_cols = ["n_tickets", "avg_resolution_time", "n_escalations",
                        "n_reopens", "avg_csat"]
        X = seg_features[feature_cols].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        self.labels_ = self.kmeans.fit_predict(X_scaled)

        seg_features = seg_features.copy()
        seg_features["segment"] = self.labels_
        seg_features["segment"] = seg_features["segment"].apply(self._label_segment)
        return seg_features

    def _label_segment(self, label, seg_df=None):
        """Assign human-readable labels based on cluster centroids."""
        # We'll assign labels after seeing the cluster characteristics
        return label  # Will be overridden by label_segments()

    def label_segments(self, seg_features):
        """Assign meaningful names to segments (guaranteed unique per cluster)."""
        seg_means = seg_features.groupby("segment").agg({
            "n_tickets": "mean",
            "avg_resolution_time": "mean",
            "n_escalations": "mean",
            "n_reopens": "mean",
            "avg_csat": "mean"
        }).reset_index()

        # Rank by dominant feature to assign unique labels
        seg_means = seg_means.sort_values("n_escalations", ascending=False)
        seg_means["rank_esc"] = range(len(seg_means))
        seg_means = seg_means.sort_values("avg_resolution_time", ascending=False)
        seg_means["rank_time"] = range(len(seg_means))

        labels = {}
        for _, row in seg_means.iterrows():
            seg = row["segment"]
            if row["rank_esc"] == 0:
                labels[seg] = "Escalators"
            elif row["rank_time"] == len(seg_means) - 1:
                labels[seg] = "Quick Resolvers"
            elif row["avg_csat"] < seg_means["avg_csat"].median():
                labels[seg] = "At-Risk"
            else:
                labels[seg] = "Standard"

        # Ensure uniqueness: if duplicates exist, append cluster number
        from collections import Counter
        counts = Counter(labels.values())
        for seg in labels:
            if counts[labels[seg]] > 1:
                labels[seg] = f"{labels[seg]} ({seg})"

        seg_features["segment_label"] = seg_features["segment"].map(labels)
        return seg_features

    def get_segment_profile(self, seg_features):
        """Return a profile table for each segment."""
        profile = seg_features.groupby("segment_label").agg(
            n_customers=("customer_id", "count"),
            avg_tickets=("n_tickets", "mean"),
            avg_resolution_hrs=("avg_resolution_time", "mean"),
            avg_escalations=("n_escalations", "mean"),
            avg_reopens=("n_reopens", "mean"),
            avg_csat=("avg_csat", "mean")
        ).round(2)
        return profile


class MarkovChainAnalyzer:
    """Analyze customer journey as a Markov chain."""

    def __init__(self):
        self.transition_matrix = None
        self.states = ["Open", "In-Progress", "Escalated", "Resolved", "Reopened", "Closed"]

    def fit(self, transition_probs):
        """
        transition_probs: {from_state: {to_state: prob}}
        Result: DataFrame with rows=from_state, cols=to_state
        So tm.loc["Open", "In-Progress"] = P(Open → In-Progress)
        """
        # Build matrix manually to avoid orient confusion
        matrix = pd.DataFrame(0.0, index=self.states, columns=self.states)
        for from_state, transitions in transition_probs.items():
            if from_state in matrix.index:
                for to_state, prob in transitions.items():
                    if to_state in matrix.columns:
                        matrix.loc[from_state, to_state] = prob
        self.transition_matrix = matrix
        return self.transition_matrix

    def expected_path_length(self, start="Open", max_steps=10):
        """Expected number of steps to reach 'Closed'."""
        if self.transition_matrix is None:
            raise ValueError("Call fit() first")
        # Simple simulation-based estimate
        total_steps = 0
        n_sims = 1000
        for _ in range(n_sims):
            state = start
            steps = 0
            while state != "Closed" and steps < max_steps:
                next_states = self.transition_matrix.loc[state].dropna()
                next_states = next_states[next_states > 0]
                if next_states.empty:
                    break
                probs = next_states / next_states.sum()
                state = np.random.choice(probs.index, p=probs.values)
                steps += 1
            total_steps += steps
        return total_steps / n_sims

    def direct_resolution_rate(self):
        """P(Open → In-Progress → Resolved) without escalation."""
        if self.transition_matrix is None:
            return None
        p_open_to_ip = self.transition_matrix.loc["Open", "In-Progress"]
        p_ip_to_res = self.transition_matrix.loc["In-Progress", "Resolved"]
        return p_open_to_ip * p_ip_to_res

    def plot_transition_matrix(self, save_path="outputs/markov_matrix.png"):
        """Heatmap of transition probabilities."""
        if self.transition_matrix is None:
            raise ValueError("Call fit() first")

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(self.transition_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                    xticklabels=self.states, yticklabels=self.states, ax=ax)
        ax.set_title("Markov Chain: Journey State Transitions")
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: {save_path}")

    def plot_path_comparison(self, control_trans, treatment_trans, save_path="outputs/markov_comparison.png"):
        """Compare transition matrices between groups."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, trans, title in zip(axes, [control_trans, treatment_trans], ["Control", "Treatment"]):
            tm = pd.DataFrame(0.0, index=self.states, columns=self.states)
            for from_s, transitions in trans.items():
                if from_s in tm.index:
                    for to_s, prob in transitions.items():
                        if to_s in tm.columns:
                            tm.loc[from_s, to_s] = prob
            sns.heatmap(tm, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
            ax.set_title(f"{title} Journey")
            ax.set_xlabel("To State")
            ax.set_ylabel("From State")
        plt.suptitle("Markov Chain Comparison: Control vs Treatment", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  → Saved: {save_path}")


def plot_segment_clusters(seg_features, save_path="outputs/segments.png"):
    """PCA scatter of customer segments (with jitter to reveal discrete structure)."""
    feature_cols = ["n_tickets", "avg_resolution_time", "n_escalations", "n_reopens", "avg_csat"]
    X = seg_features[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    # Add small jitter to break grid artifacts from discrete features
    rng = np.random.default_rng(42)
    X_pca = X_pca + rng.normal(0, 0.05, X_pca.shape)

    fig, ax = plt.subplots(figsize=(10, 7))
    # Discrete colors per segment (not a continuous gradient)
    segments = sorted(seg_features["segment"].unique())
    colors = plt.cm.tab10([i / len(segments) for i in range(len(segments))])
    for seg, color in zip(segments, colors):
        mask = seg_features["segment"] == seg
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], alpha=0.5, s=10,
                   label=f"Segment {seg} (n={mask.sum()})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    total_var = (pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]) * 100
    ax.set_title(f"Customer Segments (K-Means, PCA projection)\n"
                 f"2D view captures {total_var:.1f}% of total variance")
    ax.legend(title="Segment", loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_path}")
   
