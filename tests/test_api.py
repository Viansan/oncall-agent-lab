from fastapi.testclient import TestClient

from app.web import app


client = TestClient(app)


def test_homepage_renders_without_template_placeholder():
    response = client.get("/")
    assert response.status_code == 200
    assert "100% 合成数据" in response.text
    assert "__LIVE_CONFIGURED__" not in response.text


def test_health_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["runbooks"] == 5


def test_compare_endpoint_exposes_both_methods():
    response = client.get("/api/compare", params={"q": "刚发布的知识文章怎么都搜不到"})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"query", "keyword", "semantic"}
    assert payload["semantic"][0]["id"] == "rb-search"


def test_offline_agent_endpoint():
    response = client.post("/v3/chat", json={"message": "文件预览失败", "mode": "offline"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "offline"
    assert "rb-preview" in payload["used_documents"]


def test_live_agent_returns_clear_configuration_error():
    response = client.post("/v3/chat", json={"message": "文件预览失败", "mode": "live"})
    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]
