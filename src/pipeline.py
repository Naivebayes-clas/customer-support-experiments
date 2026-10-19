"""
Project 5: End-to-End Metrics Pipeline
Simulates Snowflake/Databricks → Power BI pipeline using DuckDB + Parquet.

Architecture:
  Raw Data (CSV/Parquet)
    → DuckDB (analytical store, simulates Snowflake)
    → Parquet files (simulates Delta Lake)
    → Dashboard data (Power BI-ready)
"""
import duckdb
import pandas as pd
import numpy as np
import os


class MetricsPipeline:
    """End-to-end data pipeline using DuckDB as the analytical store."""

    def __init__(self, db_path="outputs/metrics.duckdb"):
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        os.makedirs("outputs", exist_ok=True)

    def load_raw_data(self, tickets, conversations, churn):
        """Load raw data into DuckDB (simulates ingestion into Snowflake)."""
        self.con.execute("DROP TABLE IF EXISTS tickets")
        self.con.execute("DROP TABLE IF EXISTS conversations")
        self.con.execute("DROP TABLE IF EXISTS churn")

        self.con.register("df_tickets", tickets)
        self.con.register("df_conv", conversations)
        self.con.register("df_churn", churn)

        self.con.execute("CREATE TABLE tickets AS SELECT * FROM df_tickets")
        self.con.execute("CREATE TABLE conversations AS SELECT * FROM df_conv")
        self.con.execute("CREATE TABLE churn AS SELECT * FROM df_churn")

        self.con.unregister("df_tickets")
        self.con.unregister("df_conv")
        self.con.unregister("df_churn")

        # Also save as Parquet (simulates Delta Lake)
        tickets.to_parquet("outputs/tickets.parquet", index=False)
        conversations.to_parquet("outputs/conversations.parquet", index=False)
        churn.to_parquet("outputs/churn.parquet", index=False)

        print(f"  → Loaded: {len(tickets)} tickets, {len(conversations)} conversation turns, {len(churn)} churn records")
        print(f"  → Stored in DuckDB: {self.db_path}")
        print(f"  → Parquet: outputs/*.parquet")

    def run_analytics(self):
        """Run analytical queries (simulates dbt / Databricks notebooks)."""
        # A/B test summary by segment
        ab_summary = self.con.execute("""
            SELECT 
                "group",
                COUNT(*) as n_tickets,
                AVG(resolution_time_hours) as avg_resolution_hrs,
                AVG(csat) as avg_csat,
                SUM(escalated) as n_escalations,
                AVG(escalated) as escalation_rate,
                SUM(reopened) as n_reopens
            FROM tickets
            GROUP BY "group"
        """).df()

        # Churn by group
        churn_summary = self.con.execute("""
            SELECT 
                "group",
                COUNT(*) as n_customers,
                AVG(churned) as churn_rate,
                AVG(time_to_churn_days) as avg_time_to_churn
            FROM churn
            GROUP BY "group"
        """).df()

        # Resolution by category
        category_perf = self.con.execute("""
            SELECT 
                category,
                "group",
                COUNT(*) as n_tickets,
                AVG(resolution_time_hours) as avg_resolution_hrs,
                AVG(csat) as avg_csat
            FROM tickets
            GROUP BY category, "group"
            ORDER BY category, "group"
        """).df()

        # Conversation metrics
        conv_metrics = self.con.execute("""
            SELECT 
                ticket_id,
                COUNT(*) as n_turns,
                MAX(turn) as max_turn
            FROM conversations
            GROUP BY ticket_id
        """).df()

        return {
            "ab_summary": ab_summary,
            "churn_summary": churn_summary,
            "category_perf": category_perf,
            "conv_metrics": conv_metrics
        }

    def export_dashboard_data(self, analytics, save_dir="outputs/dashboard"):
        """Export Power BI-ready datasets."""
        os.makedirs(save_dir, exist_ok=True)

        for name, df in analytics.items():
            path = f"{save_dir}/{name}.csv"
            df.to_csv(path, index=False)
            print(f"  → Dashboard data: {path}")

    def run_query(self, sql):
        """Run arbitrary SQL query (for ad-hoc analysis)."""
        return self.con.execute(sql).df()

    def close(self):
        self.con.close()
  
