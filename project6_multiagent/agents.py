"""
Project 6: Multi-Agent LLM Workflow Orchestration System
Specialized agents: Researcher → Analyst → Writer → Reviewer
Uses LangGraph-style state machine with full audit logging
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── State Schema ──────────────────────────────────────────────
@dataclass
class AgentMessage:
    agent: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tokens_used: int = 0


@dataclass
class WorkflowState:
    task_id: str
    topic: str
    messages: List[AgentMessage] = field(default_factory=list)
    research_notes: str = ""
    analysis: str = ""
    draft_report: str = ""
    final_report: str = ""
    review_feedback: str = ""
    status: str = "pending"
    current_agent: str = ""
    completed: bool = False


# ── LLM Caller ────────────────────────────────────────────────
class LLMCaller:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._setup()

    def _setup(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.available = True
        except Exception:
            self.available = False
            logger.warning("OpenAI unavailable. Using mock agent responses.")

    def call(self, system: str, user: str, temperature: float = 0.3) -> str:
        if not self.available:
            return self._mock_response(user)
        try:
            res = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return self._mock_response(user)

    def _mock_response(self, prompt: str) -> str:
        return (f"[Mock Response] Based on the input: '{prompt[:60]}...'\n"
                "This is a placeholder response. Connect an OpenAI API key for real outputs.\n"
                "The agent would normally provide detailed, structured analysis here.")


# ── Agent Definitions ─────────────────────────────────────────
class ResearcherAgent:
    """Gathers key facts and background on the topic."""
    NAME = "Researcher"
    SYSTEM = """You are a research analyst. Given a topic, provide:
1. Key background facts (3-5 bullet points)
2. Current trends (2-3 points)
3. Key data points or statistics if relevant
4. Main stakeholders or entities involved
Be concise and factual. Format clearly with headers."""

    def __init__(self, llm: LLMCaller):
        self.llm = llm

    def run(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[{self.NAME}] Starting research on: {state.topic}")
        state.current_agent = self.NAME
        state.status = "researching"

        prompt = f"Research topic: {state.topic}"
        result = self.llm.call(self.SYSTEM, prompt)
        state.research_notes = result
        state.messages.append(AgentMessage(agent=self.NAME, content=result))
        logger.info(f"[{self.NAME}] Research complete ({len(result)} chars)")
        return state


class AnalystAgent:
    """Performs deeper analysis on the research notes."""
    NAME = "Analyst"
    SYSTEM = """You are a senior data analyst. Given research notes on a topic:
1. Identify the 3 most important insights
2. Note any risks or challenges
3. Highlight opportunities
4. Draw a data-driven conclusion
Build directly on the research notes provided. Be analytical and structured."""

    def __init__(self, llm: LLMCaller):
        self.llm = llm

    def run(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[{self.NAME}] Analyzing research notes...")
        state.current_agent = self.NAME
        state.status = "analyzing"

        prompt = f"Topic: {state.topic}\n\nResearch Notes:\n{state.research_notes}"
        result = self.llm.call(self.SYSTEM, prompt)
        state.analysis = result
        state.messages.append(AgentMessage(agent=self.NAME, content=result))
        logger.info(f"[{self.NAME}] Analysis complete ({len(result)} chars)")
        return state


class WriterAgent:
    """Drafts a professional report from research + analysis."""
    NAME = "Writer"
    SYSTEM = """You are a professional business writer. Using the research and analysis provided,
write a clear, well-structured report with:
- Executive Summary (2-3 sentences)
- Key Findings (3-4 bullet points)
- Analysis & Implications (2 paragraphs)
- Recommendations (2-3 actionable points)
- Conclusion (1 paragraph)
Write for a business audience. Be clear and concise."""

    def __init__(self, llm: LLMCaller):
        self.llm = llm

    def run(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[{self.NAME}] Drafting report...")
        state.current_agent = self.NAME
        state.status = "writing"

        prompt = (f"Topic: {state.topic}\n\n"
                  f"Research Notes:\n{state.research_notes}\n\n"
                  f"Analysis:\n{state.analysis}")
        result = self.llm.call(self.SYSTEM, prompt, temperature=0.5)
        state.draft_report = result
        state.messages.append(AgentMessage(agent=self.NAME, content=result))
        logger.info(f"[{self.NAME}] Draft complete ({len(result)} chars)")
        return state


class ReviewerAgent:
    """Reviews the draft and either approves or requests revisions."""
    NAME = "Reviewer"
    SYSTEM = """You are a senior editor and quality reviewer. Review the draft report and:
1. Rate overall quality (1-10)
2. List 2-3 specific strengths
3. List any gaps or improvements needed
4. Provide the FINAL polished version of the report (improved if needed)
Format your response as:
QUALITY_SCORE: X/10
STRENGTHS: ...
IMPROVEMENTS: ...
FINAL_REPORT:
[the full final report]"""

    def __init__(self, llm: LLMCaller):
        self.llm = llm

    def run(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"[{self.NAME}] Reviewing draft...")
        state.current_agent = self.NAME
        state.status = "reviewing"

        prompt = f"Topic: {state.topic}\n\nDraft Report:\n{state.draft_report}"
        result = self.llm.call(self.SYSTEM, prompt)
        state.review_feedback = result

        # Extract final report from review
        if "FINAL_REPORT:" in result:
            state.final_report = result.split("FINAL_REPORT:")[-1].strip()
        else:
            state.final_report = state.draft_report

        state.messages.append(AgentMessage(agent=self.NAME, content=result))
        state.completed = True
        state.status = "completed"
        logger.info(f"[{self.NAME}] Review complete. Workflow done.")
        return state


# ── Orchestrator ──────────────────────────────────────────────
class MultiAgentOrchestrator:
    """Runs agents in sequence with full state tracking and audit log."""

    def __init__(self, api_key: Optional[str] = None):
        self.llm = LLMCaller(api_key=api_key)
        self.agents = [
            ResearcherAgent(self.llm),
            AnalystAgent(self.llm),
            WriterAgent(self.llm),
            ReviewerAgent(self.llm)
        ]
        self.audit_log: List[Dict] = []

    def run(self, topic: str) -> WorkflowState:
        task_id = str(uuid.uuid4())[:8]
        state = WorkflowState(task_id=task_id, topic=topic)
        logger.info(f"=== Starting workflow [{task_id}] for: {topic} ===")

        for agent in self.agents:
            try:
                state = agent.run(state)
                self.audit_log.append({
                    "task_id": task_id,
                    "agent": agent.NAME,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": state.status,
                    "output_chars": len(state.messages[-1].content) if state.messages else 0
                })
            except Exception as e:
                logger.error(f"Agent {agent.NAME} failed: {e}")
                state.status = f"failed_at_{agent.NAME}"
                break

        return state

    def save_audit_log(self, path: str = "audit_log.json"):
        with open(path, "w") as f:
            json.dump(self.audit_log, f, indent=2)
        logger.info(f"Audit log saved to {path}")


if __name__ == "__main__":
    orchestrator = MultiAgentOrchestrator()
    topic = "The impact of generative AI on the software development industry"

    print(f"\n🚀 Running multi-agent workflow for: {topic}\n")
    state = orchestrator.run(topic)

    print("=" * 60)
    print(f"Task ID: {state.task_id}")
    print(f"Status: {state.status}")
    print(f"Agents run: {len(state.messages)}")
    print("\n📄 FINAL REPORT:")
    print(state.final_report)

    orchestrator.save_audit_log()
    print("\n✅ Audit log saved to audit_log.json")
