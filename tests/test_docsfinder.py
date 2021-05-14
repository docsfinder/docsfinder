from fastapi.testclient import TestClient

from docsfinder.main import app


def test_redirections():
    with TestClient(app) as client:
        response_home = client.get("/")
        response_api = client.get("/api")
        response_api_v1 = client.get("/api/v1")
        assert response_home.status_code == 200
        assert response_api.status_code == 200
        assert response_api_v1.status_code == 200
        assert response_home.url == response_api.url
        assert response_api.url == response_api_v1.url


def test_query():
    with TestClient(app) as client:
        url = "/api/v1/query"
        query = "query=Random query for test"
        response = client.get(f"{url}?{query}")
        assert response.status_code == 200


def test_query_with_feedback():
    with TestClient(app) as client:
        url = "/api/v1/query"
        query = "query=Random query for test&good_feedback=0&bad_feedback=1"
        response = client.get(f"{url}?{query}")
        assert response.status_code == 200
