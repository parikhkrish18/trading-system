from pathlib import Path


INDEX_HTML = Path("monitoring/dashboard/static/index.html")


def test_operator_dashboard_exposes_separate_client_portal_login():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert 'id="client-portal-link"' in html
    assert 'href="/portal"' in html
    assert 'target="_blank"' in html
    assert "Client passwords are for the separate" in html
    assert "not the operator login page" in html
    assert ">Operator log out<" in html
