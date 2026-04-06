"""
Project 3: LLM-Powered Data Analysis Copilot
Natural language → SQL/Python with self-correction loop
"""

import os
import re
import json
import logging
import sqlite3
import pandas as pd
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Database Setup ────────────────────────────────────────────
def create_sample_db(db_path: str = "sample.db"):
    """Creates a sample SQLite DB with sales data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
    DROP TABLE IF EXISTS sales;
    DROP TABLE IF EXISTS customers;
    DROP TABLE IF EXISTS products;

    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT, region TEXT, segment TEXT
    );

    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT, category TEXT, price REAL
    );

    CREATE TABLE sales (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER, product_id INTEGER,
        quantity INTEGER, revenue REAL,
        sale_date TEXT,
        FOREIGN KEY(customer_id) REFERENCES customers(id),
        FOREIGN KEY(product_id) REFERENCES products(id)
    );

    INSERT INTO customers VALUES
        (1,'Alice','North','Enterprise'),(2,'Bob','South','SMB'),
        (3,'Charlie','East','Enterprise'),(4,'Diana','West','SMB'),
        (5,'Eve','North','Mid-Market');

    INSERT INTO products VALUES
        (1,'Widget A','Hardware',299.99),(2,'Widget B','Hardware',199.99),
        (3,'Software X','Software',999.99),(4,'Software Y','Software',499.99),
        (5,'Service Z','Service',1499.99);

    INSERT INTO sales VALUES
        (1,1,3,2,1999.98,'2024-01-15'),(2,2,1,5,1499.95,'2024-01-20'),
        (3,3,5,1,1499.99,'2024-02-01'),(4,1,4,3,1499.97,'2024-02-10'),
        (5,4,2,10,1999.90,'2024-02-15'),(6,5,3,1,999.99,'2024-03-01'),
        (7,2,5,2,2999.98,'2024-03-10'),(8,3,1,4,1199.96,'2024-03-15'),
        (9,1,2,6,1199.94,'2024-04-01'),(10,4,4,2,999.98,'2024-04-05');
    """)

    conn.commit()
    conn.close()
    logger.info(f"Sample DB created at {db_path}")
    return db_path


def get_schema(db_path: str) -> str:
    """Returns schema as a string for LLM context."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schema = []
    for (table,) in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        col_defs = ", ".join([f"{c[1]} {c[2]}" for c in cols])
        schema.append(f"Table: {table} ({col_defs})")
    conn.close()
    return "\n".join(schema)


# ── LLM Agent ─────────────────────────────────────────────────
class DataCopilot:
    """LLM agent that converts natural language to SQL and executes it."""

    SQL_SYSTEM = """You are a data analyst expert. Convert the user's question into a SQL query.
Return ONLY the SQL query, nothing else. No markdown, no explanation.
The database schema is:
{schema}"""

    EXPLAIN_SYSTEM = """You are a helpful data analyst. 
Given a SQL query result, provide a clear, concise business insight in 2-3 sentences.
Be specific with numbers. Make it executive-friendly."""

    def __init__(self, db_path: str = "sample.db", api_key: Optional[str] = None):
        self.db_path = db_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.schema = get_schema(db_path)
        self.history: List[Dict] = []
        self._setup_llm()

    def _setup_llm(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.llm_available = True
        except Exception:
            self.llm_available = False
            logger.warning("OpenAI not available. Using mock responses.")

    def _call_llm(self, system: str, user: str) -> str:
        if not self.llm_available:
            return self._mock_sql(user)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def _mock_sql(self, question: str) -> str:
        """Returns a mock SQL for demo purposes."""
        q = question.lower()
        if "top" in q and "customer" in q:
            return "SELECT c.name, SUM(s.revenue) as total FROM customers c JOIN sales s ON c.id=s.customer_id GROUP BY c.name ORDER BY total DESC LIMIT 5"
        if "revenue" in q and "month" in q:
            return "SELECT strftime('%Y-%m', sale_date) as month, SUM(revenue) as total FROM sales GROUP BY month ORDER BY month"
        if "product" in q:
            return "SELECT p.name, SUM(s.revenue) as total FROM products p JOIN sales s ON p.id=s.product_id GROUP BY p.name ORDER BY total DESC"
        return "SELECT * FROM sales LIMIT 10"

    def _execute_sql(self, sql: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df

    def _fix_sql(self, sql: str, error: str, question: str) -> str:
        """Self-correction loop — asks LLM to fix broken SQL."""
        fix_prompt = f"""The following SQL query failed with error: {error}

Original question: {question}
Broken SQL: {sql}
Schema: {self.schema}

Return the corrected SQL only."""
        return self._call_llm("You are a SQL expert. Fix the SQL query.", fix_prompt)

    def query(self, question: str, max_retries: int = 2) -> Dict[str, Any]:
        """Main entry point: NL question → SQL → result → explanation."""
        system = self.SQL_SYSTEM.format(schema=self.schema)
        sql = self._call_llm(system, question)
        sql = re.sub(r"```sql|```", "", sql).strip()

        df = None
        error = None
        for attempt in range(max_retries + 1):
            try:
                df = self._execute_sql(sql)
                break
            except Exception as e:
                error = str(e)
                if attempt < max_retries:
                    logger.warning(f"SQL failed (attempt {attempt+1}): {error}. Retrying...")
                    sql = self._fix_sql(sql, error, question)
                    sql = re.sub(r"```sql|```", "", sql).strip()
                else:
                    return {"question": question, "sql": sql, "error": error, "df": None, "explanation": None}

        explanation = self._call_llm(
            self.EXPLAIN_SYSTEM,
            f"Question: {question}\nSQL: {sql}\nResult:\n{df.to_string(index=False)}"
        ) if df is not None else ""

        result = {
            "question": question,
            "sql": sql,
            "df": df,
            "explanation": explanation,
            "rows": len(df) if df is not None else 0,
            "error": None
        }
        self.history.append({"question": question, "sql": sql, "rows": result["rows"]})
        return result


if __name__ == "__main__":
    db = create_sample_db()
    copilot = DataCopilot(db_path=db)

    questions = [
        "Who are the top 3 customers by revenue?",
        "Show monthly revenue trend",
        "Which product category generates the most revenue?"
    ]

    for q in questions:
        print(f"\n🔍 Q: {q}")
        result = copilot.query(q)
        if result["error"]:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"📊 SQL: {result['sql']}")
            print(result["df"].to_string(index=False))
            print(f"💡 Insight: {result['explanation']}")
