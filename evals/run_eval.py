from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import OfflineToolAgent
from app.knowledge import KnowledgeBase
from app.retrieval import KeywordRetriever, SemanticRetriever


def load_cases() -> list[dict]:
    path = Path(__file__).with_name("cases.json")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_retriever(name: str, retriever, cases: list[dict]) -> dict:
    answerable = [case for case in cases if case["expected_documents"]]
    reciprocal_ranks: list[float] = []
    hit_count = 0
    latencies: list[float] = []
    details: list[dict] = []
    for case in answerable:
        started = time.perf_counter()
        results = retriever.search(case["query"], limit=3)
        latencies.append((time.perf_counter() - started) * 1000)
        result_ids = [item.id for item in results]
        ranks = [result_ids.index(doc_id) + 1 for doc_id in case["expected_documents"] if doc_id in result_ids]
        hit = bool(ranks)
        hit_count += int(hit)
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0.0)
        details.append(
            {
                "id": case["id"],
                "expected": case["expected_documents"],
                "actual": result_ids,
                "hit_at_3": hit,
            }
        )
    return {
        "method": name,
        "cases": len(answerable),
        "hit_at_3": round(hit_count / len(answerable), 3),
        "mrr": round(statistics.mean(reciprocal_ranks), 3),
        "median_latency_ms": round(statistics.median(latencies), 3),
        "details": details,
    }


def evaluate_agent(agent: OfflineToolAgent, cases: list[dict]) -> dict:
    document_recalls: list[float] = []
    boundary_matches = 0
    citation_matches = 0
    latencies: list[float] = []
    details: list[dict] = []
    for case in cases:
        response = agent.run(case["query"])
        latencies.append(response.latency_ms)
        expected = set(case["expected_documents"])
        actual = set(response.used_documents)
        recall = len(expected.intersection(actual)) / len(expected) if expected else 1.0
        document_recalls.append(recall)
        boundary_ok = response.boundary is case["boundary"]
        boundary_matches += int(boundary_ok)
        citation_ok = all(f"[{document_id}]" in response.answer for document_id in expected)
        citation_matches += int(citation_ok)
        details.append(
            {
                "id": case["id"],
                "expected": sorted(expected),
                "actual": response.used_documents,
                "document_recall": round(recall, 3),
                "boundary_ok": boundary_ok,
                "citation_ok": citation_ok,
            }
        )
    return {
        "method": "offline-tool-agent",
        "cases": len(cases),
        "document_recall": round(statistics.mean(document_recalls), 3),
        "boundary_accuracy": round(boundary_matches / len(cases), 3),
        "citation_rate": round(citation_matches / len(cases), 3),
        "median_latency_ms": round(statistics.median(latencies), 3),
        "external_model_calls": 0,
        "details": details,
    }


def run() -> dict:
    cases = load_cases()
    knowledge_base = KnowledgeBase()
    keyword = KeywordRetriever(knowledge_base)
    semantic = SemanticRetriever(knowledge_base, keyword)
    agent = OfflineToolAgent(knowledge_base, semantic)
    return {
        "dataset": "synthetic-runbooks-v1",
        "case_count": len(cases),
        "keyword": evaluate_retriever("keyword-bm25", keyword, cases),
        "semantic": evaluate_retriever("semantic-concept-vector", semantic, cases),
        "agent": evaluate_agent(agent, cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic On-call Agent Lab evaluation")
    parser.add_argument("--json", action="store_true", help="Print the full JSON payload")
    args = parser.parse_args()
    report = run()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(f"数据集: {report['dataset']} · {report['case_count']} 个问题")
    print("方法                         Hit@3   MRR   文档召回   边界准确率   引用率   中位延迟(ms)")
    keyword = report["keyword"]
    semantic = report["semantic"]
    agent = report["agent"]
    print(f"关键词 BM25                  {keyword['hit_at_3']:<7} {keyword['mrr']:<5} -          -            -        {keyword['median_latency_ms']}")
    print(f"概念向量语义检索             {semantic['hit_at_3']:<7} {semantic['mrr']:<5} -          -            -        {semantic['median_latency_ms']}")
    print(f"离线工具 Agent               -       -     {agent['document_recall']:<10} {agent['boundary_accuracy']:<12} {agent['citation_rate']:<8} {agent['median_latency_ms']}")


if __name__ == "__main__":
    main()
