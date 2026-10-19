"""
Project 1: RAG Chatbot with Multi-Turn Evaluation
Simulates a RAG pipeline and evaluates it using RAGAS-style metrics.

Components:
  - Mock Knowledge Base (chunked documents)
  - Retrieval (TF-IDF cosine similarity)
  - Generation (template-based, simulates LLM output)
  - Multi-turn conversation simulator
  - Evaluation: faithfulness, relevance, answer correctness, resolution rate
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


# ─── MOCK KNOWLEDGE BASE ───
KNOWLEDGE_BASE = {
    "billing": [
        "To process a refund, navigate to Account > Billing > Refunds. Refunds are processed within 5-7 business days.",
        "Payment failures are usually caused by expired cards or insufficient funds. Update your card in Account > Payment Methods.",
        "To change your subscription plan, go to Account > Billing > Change Plan. Changes take effect at the next billing cycle.",
        "Invoices can be downloaded from Account > Billing > Invoices. PDF and CSV formats are available."
    ],
    "technical": [
        "Error code E-4012 indicates a failed API authentication. Regenerate your API key from Settings > API Keys.",
        "Application crashes are often caused by memory leaks. Clear the cache using the command: system --clear-cache.",
        "Slow performance may be caused by database connection pool exhaustion. Increase pool size in config.yaml under db.pool_size.",
        "Integration errors with third-party services usually require updating OAuth tokens. Go to Settings > Integrations > Re-authenticate."
    ],
    "account": [
        "To reset your password, click 'Forgot Password' on the login page. A reset link will be sent to your registered email.",
        "To enable 2FA, go to Account > Security > Two-Factor Authentication. We support TOTP and SMS.",
        "To update your profile, navigate to Account > Profile. You can change name, email, and avatar.",
        "To export your data, go to Account > Privacy > Data Export. The export will be emailed within 24 hours."
    ],
    "shipping": [
        "To track your shipment, go to Orders > [Order ID] > Track. You'll see real-time location updates.",
        "If your delivery is delayed, check the tracking page first. If no update in 48 hours, contact support.",
        "For damaged packages, take photos within 24 hours and go to Orders > [Order ID] > Report Damage.",
        "To initiate a return, go to Orders > [Order ID] > Return. You'll receive a prepaid shipping label."
    ],
    "product": [
        "To submit a feature request, go to Help > Feedback > Feature Request. Our product team reviews all requests monthly.",
        "To report a bug, go to Help > Feedback > Bug Report. Include steps to reproduce and error screenshots.",
        "Compatibility issues: Check the system requirements page. Our platform supports Windows 10+, macOS 12+, and Linux (Ubuntu 20.04+).",
        "To upgrade to the latest version, go to Settings > About > Check for Updates. Auto-update can be enabled."
    ]
}

# Mock "gold answers" for evaluation
GOLD_ANSWERS = {
    "refund": "refund processed within 5-7 business days via Account > Billing > Refunds",
    "payment failed": "update card in Account > Payment Methods",
    "invoice": "download from Account > Billing > Invoices",
    "subscription change": "Account > Billing > Change Plan, effective next cycle",
    "error code": "regenerate API key from Settings > API Keys",
    "crash": "clear cache using system --clear-cache",
    "slow performance": "increase db.pool_size in config.yaml",
    "integration": "re-authenticate OAuth in Settings > Integrations",
    "password reset": "click Forgot Password, reset link sent to email",
    "2FA": "Account > Security > Two-Factor Authentication, TOTP or SMS",
    "profile update": "Account > Profile, change name/email/avatar",
    "data export": "Account > Privacy > Data Export, emailed within 24 hours",
    "tracking": "Orders > Order ID > Track for real-time updates",
    "delivery delay": "check tracking, if no update in 48h contact support",
    "damaged package": "take photos within 24h, Orders > Report Damage",
    "return": "Orders > Order ID > Return, prepaid label provided",
    "feature request": "Help > Feedback > Feature Request, reviewed monthly",
    "bug report": "Help > Feedback > Bug Report, include steps and screenshots",
    "compatibility": "Windows 10+, macOS 12+, Ubuntu 20.04+",
    "upgrade": "Settings > About > Check for Updates"
}


class RAGPipeline:
    """Simulated RAG pipeline with retrieval and generation."""

    def __init__(self, chunk_size="full", prompt_style="standard"):
        """
        chunk_size: 'full' (use full doc) or 'split' (split into sentences)
        prompt_style: 'standard' or 'enhanced' (affects generation quality)
        """
        self.chunk_size = chunk_size
        self.prompt_style = prompt_style
        self.documents = []
        self.doc_ids = []
        self._build_index()

    def _build_index(self):
        """Build the document index (simulates embedding store)."""
        for category, docs in KNOWLEDGE_BASE.items():
            for i, doc in enumerate(docs):
                if self.chunk_size == "split":
                    # Split into sentence-level chunks
                    sentences = re.split(r'(?<=[.!?])\s+', doc)
                    for s in sentences:
                        self.documents.append(s)
                        self.doc_ids.append(f"{category}_{i}_{len(self.documents)}")
                else:
                    self.documents.append(doc)
                    self.doc_ids.append(f"{category}_{i}")

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_matrix = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query, top_k=3):
        """Retrieve top-k most relevant chunks."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], self.documents[i], scores[i]) for i in top_indices]

    def generate(self, query, retrieved_chunks):
        """
        Simulate LLM generation based on retrieved context.
        'enhanced': full KB text + reference (high faithfulness)
        'standard': truncated + generic filler (lower faithfulness)
        """
        context = " ".join([c[1] for c in retrieved_chunks])

        if self.prompt_style == "enhanced":
            # Full context, structured, with reference
            answer = f"Based on our documentation: {context} [Ref: {retrieved_chunks[0][0]}]"
        else:
            # Truncated, generic filler, no reference
            answer = f"Thanks for reaching out! I understand you're asking about this topic. Let me help you with that. The general approach would be to check your account settings and follow the standard procedure. {context[:50]}..."

        return answer


class MultiTurnEvaluator:
    """Evaluate multi-turn RAG conversations using RAGAS-style metrics."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def simulate_conversation(self, pipeline, intent, category, n_turns=4, rng=None):
        """Simulate a multi-turn conversation."""
        rng = rng or self.rng
        turns = []
        resolved = False

        for turn in range(n_turns):
            if turn % 2 == 0:  # Customer asks
                msg = f"I have a question about {intent}"
                turns.append({"role": "customer", "message": msg, "turn": turn})
            else:  # Agent responds (RAG)
                retrieved = pipeline.retrieve(intent, top_k=3)
                answer = pipeline.generate(intent, retrieved)
                turns.append({"role": "agent", "message": answer, "turn": turn,
                              "retrieved_ids": [r[0] for r in retrieved],
                              "retrieval_scores": [r[2] for r in retrieved]})

                # Check if resolved: top retrieval is from correct category (hit@1)
                top_doc_id = retrieved[0][0]
                if category in top_doc_id:
                    resolved = True

        return {"intent": intent, "category": category, "turns": turns,
                "resolved": resolved, "n_turns": n_turns}

    def evaluate_faithfulness(self, conversation):
        """
        RAGAS Faithfulness: Is the answer supported by the retrieved context?
        (Simplified: check if key terms from retrieved docs appear in answer)
        """
        agent_turns = [t for t in conversation["turns"] if t["role"] == "agent"]
        if not agent_turns:
            return 0.0

        scores = []
        for t in agent_turns:
            retrieved_text = " ".join([
                KNOWLEDGE_BASE[conversation["category"]][0]  # simplified
            ])
            # Faithfulness = fraction of context terms present in answer
            # (measures: "is the answer grounded in the retrieved context?")
            answer_words = set(t["message"].lower().split())
            context_words = set(retrieved_text.lower().split())
            overlap = len(answer_words & context_words)
            total = max(len(context_words), 1)  # denominator = context size
            scores.append(overlap / total)

        return np.mean(scores) if scores else 0.0

    def evaluate_relevance(self, conversation):
        """
        RAGAS Answer Relevance: Is the answer relevant to the question?
        (Simplified: cosine similarity between intent and answer)
        """
        intent = conversation["intent"]
        agent_turns = [t for t in conversation["turns"] if t["role"] == "agent"]
        if not agent_turns:
            return 0.0

        # Simple: does the answer mention the intent keyword?
        scores = []
        for t in agent_turns:
            if intent.lower() in t["message"].lower():
                scores.append(1.0)
            else:
                # Partial credit for related terms
                related = any(w in t["message"].lower() for w in intent.lower().split())
                scores.append(0.5 if related else 0.0)

        return np.mean(scores) if scores else 0.0

    def evaluate_correctness(self, conversation):
        """
        RAGAS Answer Correctness: Does the answer contain the gold information?
        """
        intent = conversation["intent"]
        gold = GOLD_ANSWERS.get(intent, "")
        if not gold:
            return 0.5  # neutral if no gold answer

        agent_text = " ".join([t["message"] for t in conversation["turns"] if t["role"] == "agent"])
        gold_words = set(gold.lower().split())
        answer_words = set(agent_text.lower().split())
        recall = len(gold_words & answer_words) / max(len(gold_words), 1)
        return recall

    def evaluate_multi_turn(self, conversation):
        """
        Multi-turn specific metrics:
        - Resolution rate: was the issue resolved?
        - Turns to resolve: how many turns did it take?
        - Escalation: did it need more than N turns?
        """
        return {
            "resolved": conversation["resolved"],
            "n_turns": conversation["n_turns"],
            "turns_to_resolve": conversation["n_turns"] if conversation["resolved"] else None,
            "escalated": conversation["n_turns"] > 5
        }

    def full_evaluation(self, conversation):
        """Run all metrics on a conversation."""
        return {
            "intent": conversation["intent"],
            "category": conversation["category"],
            "faithfulness": round(self.evaluate_faithfulness(conversation), 3),
            "relevance": round(self.evaluate_relevance(conversation), 3),
            "correctness": round(self.evaluate_correctness(conversation), 3),
            "resolved": conversation["resolved"],
            "n_turns": conversation["n_turns"],
            "escalated": conversation["n_turns"] > 5
        }


def run_rag_evaluation(n_conversations=200, seed=42, save_path="outputs/rag_evaluation.csv"):
    """
    Run RAG evaluation across multiple conversation configurations.
    Tests: standard vs enhanced prompt, full vs split chunks.
    """
    rng = np.random.default_rng(seed)
    evaluator = MultiTurnEvaluator(seed=seed)
    categories = list(KNOWLEDGE_BASE.keys())
    intents = list(GOLD_ANSWERS.keys())

    results = []
    configs = [
        ("standard_full", RAGPipeline(chunk_size="full", prompt_style="standard")),
        ("standard_split", RAGPipeline(chunk_size="split", prompt_style="standard")),
        ("enhanced_full", RAGPipeline(chunk_size="full", prompt_style="enhanced")),
        ("enhanced_split", RAGPipeline(chunk_size="split", prompt_style="enhanced")),
    ]

    for config_name, pipeline in configs:
        for _ in range(n_conversations):
            intent = rng.choice(intents)
            intent_to_cat = {"refund":"billing","payment failed":"billing","invoice":"billing","subscription change":"billing","error code":"technical","crash":"technical","slow performance":"technical","integration":"technical","password reset":"account","2FA":"account","profile update":"account","data export":"account","tracking":"shipping","delivery delay":"shipping","damaged package":"shipping","return":"shipping","feature request":"product","bug report":"product","compatibility":"product","upgrade":"product"}
            category = intent_to_cat.get(intent, "billing")
            n_turns = rng.poisson(4, 1)[0] + 2

            conv = evaluator.simulate_conversation(
                pipeline, intent, category, n_turns=n_turns, rng=rng
            )
            eval_result = evaluator.full_evaluation(conv)
            eval_result["config"] = config_name
            results.append(eval_result)

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"  → Saved: {save_path} ({len(df)} evaluations)")
    return df


def plot_rag_metrics(df, save_path="outputs/rag_metrics.png"):
    """Plot RAGAS metrics by configuration."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["faithfulness", "relevance", "correctness"]
    for ax, metric in zip(axes, metrics):
        grouped = df.groupby("config")[metric].mean()
        bars = ax.bar(grouped.index, grouped.values, color="steelblue", alpha=0.8)
        ax.set_title(f"Mean {metric.title()} by Config")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, grouped.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha='center', fontsize=9)

    plt.suptitle("RAG Pipeline Evaluation (RAGAS-style Metrics)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {save_path}")
   
