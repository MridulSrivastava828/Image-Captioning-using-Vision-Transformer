import http.client
import os
import subprocess
import sys
import time
import urllib.request
from urllib.error import HTTPError

STREAMLIT_HOST = os.environ.get("STREAMLIT_HOST", "127.0.0.1")
STREAMLIT_PORT = os.environ.get("STREAMLIT_PORT", "8501")
STREAMLIT_URL = f"http://{STREAMLIT_HOST}:{STREAMLIT_PORT}"


def start_streamlit() -> None:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.address",
        STREAMLIT_HOST,
        "--server.port",
        STREAMLIT_PORT,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.Popen(
        cmd,
        cwd=os.path.dirname(__file__) or ".",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def wait_for_streamlit(timeout_seconds: int = 45) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            request = urllib.request.Request(f"{STREAMLIT_URL}/_stcore/health")
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


start_streamlit()
wait_for_streamlit()


def app(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    query_string = environ.get("QUERY_STRING", "")
    target_url = f"{STREAMLIT_URL}{path}"
    if query_string:
        target_url = f"{target_url}?{query_string}"

    content_length = environ.get("CONTENT_LENGTH", "0")
    data = environ["wsgi.input"].read(int(content_length)) if content_length else b""

    headers = {}
    for key, value in environ.items():
        if key.startswith("HTTP_"):
            header_name = key[5:].replace("_", "-").title()
            headers[header_name] = value

    headers["Host"] = f"{STREAMLIT_HOST}:{STREAMLIT_PORT}"
    if data:
        headers.setdefault("Content-Type", environ.get("CONTENT_TYPE", "application/octet-stream"))

    request = urllib.request.Request(
        target_url,
        data=data or None,
        headers=headers,
        method=environ.get("REQUEST_METHOD", "GET"),
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = f"{response.status} {http.client.responses.get(response.status, '')}".strip()
            response_headers = [
                (key, value)
                for key, value in response.headers.items()
                if key.lower() not in {"transfer-encoding", "content-encoding", "content-length"}
            ]
            start_response(status, response_headers)
            return [response.read()]
    except HTTPError as exc:
        status = f"{exc.code} {http.client.responses.get(exc.code, '')}".strip()
        response_headers = [
            (key, value)
            for key, value in exc.headers.items()
            if key.lower() not in {"transfer-encoding", "content-encoding", "content-length"}
        ]
        start_response(status, response_headers)
        return [exc.read()]
    except Exception as exc:
        start_response("502 Bad Gateway", [("Content-Type", "text/plain")])
        return [str(exc).encode("utf-8")]
