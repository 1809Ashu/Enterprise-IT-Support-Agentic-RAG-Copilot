import json
import logging
import re
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.rag.state import AgentState
from app.rag.vectorstores import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_CHARS_PER_DOC = 3000
DDG_MAX_RESULTS = 5

_llm = None
_web_search = None


# ---------- Model / tool singletons ----------

def llm():
    global _llm
    if _llm is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        kwargs = dict(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        base_url = getattr(settings, "openai_base_url", None)
        if base_url:
            kwargs["base_url"] = base_url
        _llm = ChatOpenAI(**kwargs)
    return _llm


def web_search_tool():
    """Tavily client. Raises if the key is missing; callers handle that."""
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search


# ---------- Helpers ----------

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


def _parse_json(text: str, default: dict) -> dict:
    """Extract the first JSON object from an LLM reply. Never raises."""
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    logger.warning("Could not parse JSON from LLM output: %r", text[:200])
    return default


def _format_docs(docs) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_name") or "unknown"
        parts.append(f"[Source: {src}]\n{d.page_content[:MAX_CHARS_PER_DOC]}")
    return "\n\n".join(parts)


def _short_name(src: str) -> str:
    return re.split(r"[\\/]", src)[-1] or src


def _normalize_search_result(raw) -> dict:
    """Tavily can return a dict, a list, a JSON string, or a (content, artifact) tuple."""
    if isinstance(raw, tuple):
        raw = raw[1] if len(raw) > 1 and raw[1] else raw[0]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {"answer": "", "results": [{"title": "", "url": "", "content": raw}]}
    if isinstance(raw, list):
        return {"answer": "", "results": raw}
    if isinstance(raw, dict):
        return raw
    return {"answer": "", "results": []}


# ---------- Search providers ----------

def _tavily_search(query: str) -> tuple[str, list[dict], str]:
    """Returns (answer, results, error_message). Never raises."""
    try:
        raw = web_search_tool().invoke({"query": query})
        data = _normalize_search_result(raw)
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in data.get("results", []) or []
        ]
        return data.get("answer", "") or "", results, ""
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return "", [], f"{type(exc).__name__}: {str(exc)[:150]}"


def _ddg_search(query: str, max_results: int = DDG_MAX_RESULTS) -> tuple[list[dict], str]:
    """DuckDuckGo search with no API key. Returns (results, error_message). Never raises."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            hits = ddgs.text(query, max_results=max_results) or []

        results = [
            {
                "title": h.get("title", ""),
                "url": h.get("href", ""),
                "content": h.get("body", ""),
            }
            for h in hits
        ]
        return results, ""
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return [], f"{type(exc).__name__}: {str(exc)[:150]}"


# ---------- Nodes ----------

def route_question(state: AgentState):
    prompt = f"""You route messages for an enterprise IT support assistant.
Use "kb" for questions about company IT policies, VPN, password reset, MFA, laptop setup,
software access, security, email, devices, troubleshooting, technology support, or
company documents such as resumes, certifications, and policies.
Use "direct" ONLY for greetings, thanks, or casual chat that needs no company knowledge.
When unsure, use "kb".
Question: {state['question']}
Return only JSON: {{"route": "kb"}} or {{"route": "direct"}}"""
    data = _parse_json(llm().invoke(prompt).content, {"route": "kb"})
    route = "direct" if str(data.get("route", "kb")).lower() == "direct" else "kb"
    return {
        "source_used": route,
        "trace": add_trace(state, f"Router → {route.upper()}"),
    }


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"


def retrieve_kb(state: AgentState):
    try:
        docs = get_retriever().invoke(state["current_query"])
    except Exception as exc:
        logger.exception("KB retrieval failed: %s", exc)
        docs = []
    return {
        "kb_docs": docs,
        "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks"),
    }


def grade_kb(state: AgentState):
    docs = state.get("kb_docs") or []
    if not docs:
        return {
            "kb_grade": "weak",
            "trace": add_trace(state, "KB evidence grade → WEAK (no chunks retrieved)"),
        }

    prompt = f"""You check whether company KB excerpts are relevant to an IT support question.
Return "good" if the excerpts contain information relevant to the question and can help
answer it, even if only partially. Return "weak" only if the excerpts are unrelated.
Question: {state['question']}

Private KB excerpts:
{_format_docs(docs)}

Return only JSON: {{"grade": "good"}} or {{"grade": "weak"}}"""
    data = _parse_json(llm().invoke(prompt).content, {"grade": "good"})
    grade = "good" if str(data.get("grade", "good")).lower() == "good" else "weak"
    return {
        "kb_grade": grade,
        "trace": add_trace(state, f"KB evidence grade → {grade.upper()}"),
    }


def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"


def search_web(state: AgentState):
    query = state["current_query"]

    # 1. Tavily first (if configured)
    answer, results, tavily_err = "", [], ""
    if settings.tavily_api_key:
        answer, results, tavily_err = _tavily_search(query)
        provider = "Tavily"
    else:
        tavily_err = "TAVILY_API_KEY not set"

    # 2. DuckDuckGo fallback when Tavily failed or returned nothing
    ddg_err = ""
    if not results:
        results, ddg_err = _ddg_search(query)
        provider = "DuckDuckGo"

    lines, citations = [], []
    if answer:
        lines.append("Search answer: " + answer)
    for item in results:
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
        citations.append({"title": title or url, "url": url, "type": "web"})

    if results:
        trace_msg = f"Web fallback → {provider} ({len(results)} results)"
    elif ddg_err:
        trace_msg = f"Web fallback → no results (Tavily: {tavily_err or 'no results'}; DuckDuckGo ERROR: {ddg_err})"
    else:
        trace_msg = f"Web fallback → no results (Tavily: {tavily_err or 'no results'}; DuckDuckGo: no results)"

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, trace_msg),
    }


def grade_web(state: AgentState):
    if not state.get("web_results"):
        return {
            "web_grade": "weak",
            "trace": add_trace(state, "Web evidence grade → WEAK (no results)"),
        }

    prompt = f"""Question: {state['question']}

Web evidence:
{state['web_results']}

Return "good" if the evidence is relevant to the question and can help answer it.
Return "weak" only if it is unrelated.
Return only JSON: {{"grade": "good"}} or {{"grade": "weak"}}"""
    data = _parse_json(llm().invoke(prompt).content, {"grade": "weak"})
    grade = "good" if str(data.get("grade", "weak")).lower() == "good" else "weak"
    return {
        "web_grade": grade,
        "trace": add_trace(state, f"Web evidence grade → {grade.upper()}"),
    }


def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"


def rewrite_query(state: AgentState):
    rewritten = llm().invoke(f"""Rewrite this IT support question for better private knowledge retrieval
and web search. Preserve intent, add useful technical keywords, do not answer.
Return only the rewritten query.
Question: {state['question']}""").content.strip()
    return {
        "current_query": rewritten or state["current_query"],
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }


def generate_from_kb(state: AgentState):
    answer = llm().invoke(f"""You are an enterprise IT support copilot. Answer ONLY from the private
company KB excerpts below. Be concise, actionable, and safe. If steps are present, list them clearly.
Do not invent policy details. If the excerpts only partly answer the question, say what is missing
and suggest contacting the IT help desk. Mention that the answer is based on the company's private
knowledge base.

Question: {state['question']}

Private KB:
{_format_docs(state['kb_docs'])}""").content

    citations, seen = [], set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source") or d.metadata.get("file_name") or "Private KB"
        if src not in seen:
            seen.add(src)
            citations.append({"title": _short_name(src), "url": "", "type": "private_kb"})

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(state, "Answer generation → PRIVATE KB"),
    }


def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""You are an enterprise IT support copilot. The private company KB was
insufficient. Answer ONLY from the web evidence below. Clearly state that this is external web
information and may need IT validation before changing company-managed systems.

Question: {state['question']}

Web evidence:
{state['web_results']}""").content
    return {
        "answer": answer,
        "source_used": "web_search",
        "trace": add_trace(state, "Answer generation → WEB SEARCH"),
    }


def direct_answer(state: AgentState):
    answer = llm().invoke(f"Respond briefly and naturally to: {state['question']}").content
    return {
        "answer": answer,
        "source_used": "direct",
        "trace": add_trace(state, "Direct response → no retrieval"),
    }


def insufficient(state: AgentState):
    return {
        "answer": (
            "I couldn't find enough reliable evidence in the company knowledge base or external "
            "search to answer confidently. Please contact the IT help desk or provide more details."
        ),
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }


# ---------- Graph ----------

def build_graph():
    graph = StateGraph(AgentState)
    nodes = {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb",
        "direct_answer": "direct_answer",
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb",
        "search_web": "search_web",
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web",
        "rewrite_query": "rewrite_query",
        "insufficient": "insufficient",
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()


agent_graph = build_graph()


def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)