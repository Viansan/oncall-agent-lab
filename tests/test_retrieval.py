from app.knowledge import KnowledgeBase
from app.retrieval import KeywordRetriever, SemanticRetriever


def build_retrievers():
    kb = KnowledgeBase()
    keyword = KeywordRetriever(kb)
    return kb, keyword, SemanticRetriever(kb, keyword)


def test_keyword_finds_exact_gateway_language():
    _, keyword, _ = build_retrievers()
    results = keyword.search("网关错误率突然升高")
    assert results[0].id == "rb-gateway"


def test_semantic_maps_paraphrase_to_stream_runbook():
    _, _, semantic = build_retrievers()
    query = "异步处理一直不往前走，待处理数字持续增加"
    result = semantic.search(query)[0]
    assert result.id == "rb-stream"
    assert any("任务积压" in reason for reason in result.reasons)


def test_semantic_cross_service_query_returns_both_runbooks():
    _, _, semantic = build_retrievers()
    ids = {item.id for item in semantic.search("新上传文件既无法预览，也无法搜索")}
    assert {"rb-preview", "rb-search"}.issubset(ids)


def test_knowledge_tool_rejects_arbitrary_paths():
    kb, _, _ = build_retrievers()
    try:
        kb.read_runbook("../../.env")
    except KeyError:
        pass
    else:
        raise AssertionError("arbitrary paths must not be readable")
