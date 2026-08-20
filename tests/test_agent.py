from app.agent import OfflineToolAgent
from app.knowledge import KnowledgeBase
from app.retrieval import KeywordRetriever, SemanticRetriever


def build_agent():
    kb = KnowledgeBase()
    keyword = KeywordRetriever(kb)
    semantic = SemanticRetriever(kb, keyword)
    return OfflineToolAgent(kb, semantic)


def test_agent_calls_read_only_tool_and_cites_document():
    response = build_agent().run("刚发布的知识文章怎么都搜不到")
    assert response.used_documents == ["rb-search"]
    assert "[rb-search]" in response.answer
    assert [event.type for event in response.trace] == ["tool_call", "tool_result"]
    assert response.trace[0].tool == "read_runbook"


def test_agent_reads_two_documents_for_cross_service_query():
    response = build_agent().run("新上传文件既无法预览，也无法搜索")
    assert set(response.used_documents) == {"rb-preview", "rb-search"}
    assert response.usage["tool_calls"] == 2


def test_agent_stops_at_knowledge_boundary_without_tool_call():
    response = build_agent().run("怎么申请差旅报销")
    assert response.boundary is True
    assert response.trace == []
    assert response.used_documents == []
    assert "无法支持" in response.answer
