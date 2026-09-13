"""GitHub OAuth authorization and authenticated issue creation."""
from __future__ import annotations

from urllib.parse import urlencode

import httpx


class GitHubOAuth:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret

    def authorization_url(self, redirect_uri: str, state: str) -> str:
        return "https://github.com/login/oauth/authorize?" + urlencode({
            "client_id": self.client_id, "redirect_uri": redirect_uri, "state": state, "scope": "repo",
        })

    def exchange(self, code: str, redirect_uri: str) -> str:
        response = httpx.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            json={"client_id": self.client_id, "client_secret": self.client_secret, "code": code, "redirect_uri": redirect_uri},
            timeout=30,
        )
        response.raise_for_status()
        return str(response.json()["access_token"])

    def create_issue(self, token: str, repository: str, title: str, body: str) -> str:
        response = httpx.post(
            f"https://api.github.com/repos/{repository}/issues",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            json={"title": title, "body": body}, timeout=30,
        )
        response.raise_for_status()
        return str(response.json()["html_url"])

