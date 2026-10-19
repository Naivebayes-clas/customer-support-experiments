"""
Unified synthetic data generator for the Customer Support Platform.
Generates:
  - Support tickets with journey states
  - Multi-turn conversations (for RAG evaluation)
  - Churn outcomes (time-to-event)
  - A/B test assignments
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_support_platform(n_users=5000, seed=42, true_lift=0.08):
    """
    Generate a complete synthetic dataset for the customer support platform.
    
    Parameters:
        n_users: total number of customers
        seed: random seed
        true_lift: true relative lift in resolution rate for treatment
    
    Returns:
        tickets: DataFrame of support tickets
        conversations: DataFrame of conversation turns
        churn: DataFrame of time-to-churn
    """
    rng = np.random.default_rng(seed)
    n_per_group = n_users // 2

    # ─── CUSTOMERS ───
    customer_ids = [f"c_{i:05d}" for i in range(n_users)]
    ages = rng.normal(38, 12, n_users)
    plan_tiers = rng.choice(["basic", "pro", "enterprise"], n_users, p=[0.5, 0.35, 0.15])
    signup_dates = [datetime(2024, 1, 1) - timedelta(days=int(d)) 
                    for d in rng.exponential(200, n_users)]

    # ─── A/B ASSIGNMENT ───
    groups = np.array(["control"] * n_per_group + ["treatment"] * (n_users - n_per_group))
    rng.shuffle(groups)

    # ─── SUPPORT TICKETS ───
    # Journey states: Open → In-Progress → Resolved → Closed
    # Some tickets: Open → In-Progress → Escalated → Resolved → Closed
    # Some tickets: Open → In-Progress → Resolved → Reopened → Resolved → Closed
    ticket_ids = []
    customer_ticket_map = []
    states_sequence = []
    resolution_times = []  # hours from open to resolved
    escalation_flags = []
    reopen_flags = []
    csat_scores = []
    ticket_categories = []
    ticket_groups = []

    categories = ["billing", "technical", "account", "shipping", "product"]

    for i in range(n_users):
        n_tickets = rng.poisson(3, 1)[0] + 1  # 1-5 tickets per customer
        for t in range(n_tickets):
            tid = f"t_{i:05d}_{t}"
            ticket_ids.append(tid)
            customer_ticket_map.append(customer_ids[i])
            ticket_groups.append(groups[i])
            cat = rng.choice(categories)
            ticket_categories.append(cat)

            # Journey simulation
            state_seq = ["Open"]
            t_open = 0
            t_in_progress = t_open + rng.exponential(2)
            state_seq.append("In-Progress")

            # Escalation probability (higher for "technical" and "enterprise")
            esc_prob = 0.15
            if cat == "technical":
                esc_prob += 0.15
            if plan_tiers[i] == "enterprise":
                esc_prob += 0.1

            if rng.random() < esc_prob:
                t_escalated = t_in_progress + rng.exponential(4)
                state_seq.append("Escalated")
                escalation_flags.append(1)
                t_resolved = t_escalated + rng.exponential(6)
            else:
                escalation_flags.append(0)
                # Treatment effect: faster resolution
                base_res_time = rng.exponential(8)
                if groups[i] == "treatment":
                    t_resolved = t_in_progress + base_res_time * (1 - true_lift)
                else:
                    t_resolved = t_in_progress + base_res_time

            state_seq.append("Resolved")
            t_closed = t_resolved + rng.exponential(1)
            state_seq.append("Closed")

            # Reopen probability
            reopen_prob = 0.1
            if cat == "technical":
                reopen_prob += 0.1
            if rng.random() < reopen_prob:
                state_seq.append("Reopened")
                state_seq.append("Resolved")
                reopen_flags.append(1)
                t_resolved += rng.exponential(3)
            else:
                reopen_flags.append(0)

            states_sequence.append("→".join(state_seq))
            resolution_times.append(t_resolved)

            # CSAT (1-5), correlated with resolution time and escalation
            csat_base = 4.2 - 0.04 * (t_resolved - 5) - 0.3 * escalation_flags[-1]
            if groups[i] == "treatment" and escalation_flags[-1]:
                csat_base += 0.3  # Treatment helps escalators more
            csat = np.clip(rng.normal(csat_base, 0.5), 1, 5)
            csat_scores.append(round(csat, 1))

    tickets = pd.DataFrame({
        "ticket_id": ticket_ids,
        "customer_id": customer_ticket_map,
        "group": ticket_groups,
        "category": ticket_categories,
        "state_sequence": states_sequence,
        "resolution_time_hours": np.round(resolution_times, 2),
        "escalated": escalation_flags,
        "reopened": reopen_flags,
        "csat": csat_scores
    })

    # ─── CONVERSATIONS (for RAG evaluation) ───
    # Simulate multi-turn conversations
    conv_ids = []
    conv_ticket_ids = []
    turn_numbers = []
    speaker = []
    messages = []
    intents = []
    resolved_flags = []

    # Mock knowledge base topics
    kb_topics = {
        "billing": ["refund", "payment failed", "invoice", "subscription change"],
        "technical": ["error code", "crash", "slow performance", "integration"],
        "account": ["password reset", "2FA", "profile update", "data export"],
        "shipping": ["tracking", "delivery delay", "damaged package", "return"],
        "product": ["feature request", "bug report", "compatibility", "upgrade"]
    }

    for _, row in tickets.iterrows():
        cat = row["category"]
        n_turns = rng.poisson(4, 1)[0] + 2  # 2-8 turns
        intent = rng.choice(kb_topics[cat])
        resolved = 1 if row["resolution_time_hours"] < 12 else 0

        for turn in range(n_turns):
            cid = f"conv_{row['ticket_id']}"
            conv_ids.append(cid)
            conv_ticket_ids.append(row["ticket_id"])
            turn_numbers.append(turn)
            speaker.append("customer" if turn % 2 == 0 else "agent")

            if turn % 2 == 0:  # customer message
                msg = f"Customer asks about {intent}"
            else:  # agent response
                if row["group"] == "treatment":
                    # Treatment: more specific, references KB
                    msg = f"Agent provides KB-guided answer for {intent} (ref: KB-{cat.upper()}-{rng.integers(100,999)})"
                else:
                    msg = f"Agent responds to {intent} query"
            messages.append(msg)
            intents.append(intent)
            resolved_flags.append(resolved)

    conversations = pd.DataFrame({
        "conversation_id": conv_ids,
        "ticket_id": conv_ticket_ids,
        "turn": turn_numbers,
        "speaker": speaker,
        "message": messages,
        "intent": intents,
        "resolved": resolved_flags
    })

    # ─── CHURN DATA (time-to-event) ───
    churn_times = []
    churn_events = []
    churn_groups = []
    churn_csat = []
    churn_escalated = []
    churn_tickets = []

    for i in range(n_users):
        cust_tickets = tickets[tickets["customer_id"] == customer_ids[i]]
        avg_csat = cust_tickets["csat"].mean()
        n_esc = cust_tickets["escalated"].sum()
        n_reopen = cust_tickets["reopened"].sum()
        group = groups[i]

        # Baseline hazard: worse CSAT → more likely to churn
        base_rate = 0.000289 + 0.00005 * (5 - avg_csat) + 0.00003 * n_esc + 0.00002 * n_reopen

        # Treatment effect on hazard (reduces churn)
        if group == "treatment":
            base_rate *= 0.80  # Treatment: 20% hazard reduction (HR=0.80)

        # Simulate time-to-churn (censored at 365 days)
        # Weibull with shape=0.7 (decreasing hazard → front-loaded churn)
        t_churn = rng.weibull(0.7, size=1)[0] / base_rate
        censored_at = 365
        t_obs = min(t_churn, censored_at)
        event = 1 if t_churn < censored_at else 0

        churn_times.append(round(t_obs, 1))
        churn_events.append(event)
        churn_groups.append(group)
        churn_csat.append(round(avg_csat, 2))
        churn_escalated.append(n_esc)
        churn_tickets.append(len(cust_tickets))

    churn = pd.DataFrame({
        "customer_id": customer_ids,
        "group": churn_groups,
        "time_to_churn_days": churn_times,
        "churned": churn_events,
        "avg_csat": churn_csat,
        "n_escalations": churn_escalated,
        "n_tickets": churn_tickets
    })

    return tickets, conversations, churn


def get_journey_transitions(tickets):
    """
    Extract state transition counts for Markov chain analysis.
    Returns: transition matrix (dict of dicts)
    """
    states = ["Open", "In-Progress", "Escalated", "Resolved", "Reopened", "Closed"]
    trans_count = {s: {s2: 0 for s2 in states} for s in states}

    for seq in tickets["state_sequence"]:
        states_seq = seq.split("→")
        for i in range(len(states_seq) - 1):
            s_from = states_seq[i]
            s_to = states_seq[i + 1]
            if s_from in trans_count and s_to in trans_count[s_from]:
                trans_count[s_from][s_to] += 1

    # Convert to probabilities
    trans_prob = {}
    for s in states:
        total = sum(trans_count[s].values())
        trans_prob[s] = {s2: (count / total if total > 0 else 0) for s2, count in trans_count[s].items()}

    return trans_prob


def get_segment_features(tickets):
    """
    Extract features per customer for segmentation.
    """
    seg_data = tickets.groupby("customer_id").agg(
        n_tickets=("ticket_id", "count"),
        avg_resolution_time=("resolution_time_hours", "mean"),
        n_escalations=("escalated", "sum"),
        n_reopens=("reopened", "sum"),
        avg_csat=("csat", "mean"),
        group=("group", "first")
    ).reset_index()
    return seg_data


if __name__ == "__main__":
    tickets, convs, churn = generate_support_platform(n_users=5000)
    print(f"Tickets: {len(tickets)}")
    print(f"Conversations: {len(convs)} turns")
    print(f"Churn records: {len(churn)}")
    print(f"\nJourney transitions (sample):")
    tp = get_journey_transitions(tickets)
    for s in ["Open", "In-Progress", "Resolved"]:
        print(f"  {s} → {dict(list(tp[s].items())[:3])}")
   
