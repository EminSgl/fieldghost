"""
Minimal CVAT REST client with the two gotchas every fresh script trips on:

1. CVAT deployments usually sit behind Cloudflare. A plain urllib/requests
   call with no User-Agent (or a non-browser one) gets a 403 with Cloudflare
   error code 1010 before it ever reaches CVAT. Every request here sends a
   real browser User-Agent for exactly this reason.
2. Auth tokens go stale across sessions (a token that worked yesterday can
   403 today even though nothing about the request changed). `put_annotations`
   raises `TokenExpired` on a 403 so the caller can prompt for fresh
   credentials and retry, instead of failing with a confusing raw HTTPError.

Usage:
    from cvat_client import login, get_annotations, put_annotations, TokenExpired

    token = login(host, username, password)
    data = get_annotations(host, token, job_id)
    put_annotations(host, token, job_id, data)
"""
import json
import urllib.request
import urllib.error

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)


class TokenExpired(Exception):
    """Raised when CVAT returns 403 on an authenticated request."""


def _request(url, method="GET", token=None, payload=None, timeout=600):
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Token {token}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise TokenExpired(f"{method} {url} -> 403 (token expired or invalid)") from e
        raise


def login(host, username, password):
    """POST /api/auth/login. Returns the auth token string."""
    status, body = _request(
        f"{host}/api/auth/login",
        method="POST",
        payload={"username": username, "password": password},
    )
    return body["key"]


def get_annotations(host, token, job_id):
    _, body = _request(f"{host}/api/jobs/{job_id}/annotations", token=token)
    return body


def put_annotations(host, token, job_id, payload):
    """PUT (full replace) the annotations for one job. Returns the HTTP status."""
    status, _ = _request(
        f"{host}/api/jobs/{job_id}/annotations",
        method="PUT",
        token=token,
        payload=payload,
    )
    return status
