"""
Deployment-level smoke tests for FRE GCP v1.0.
================================================
Designed to run against a LIVE Cloud Run deployment (no proxy, no local venv).
Also works locally if CLOUD_RUN_URL is set to http://localhost:8080.

Usage
-----
# Against Cloud Run:
    $env:CLOUD_RUN_URL="https://fre-cognitive-search-xxxx-uc.a.run.app"
    python test_deployment.py

# Against local ADK (no proxy needed — ADK itself handles GCP calls):
    $env:CLOUD_RUN_URL="http://localhost:8080"
    python test_deployment.py

Environment variables
---------------------
CLOUD_RUN_URL          Required. Base URL of the deployed service.
TEST_APP_NAME          ADK app name (default: cognitive_search_agent)
TEST_TIMEOUT_S         Per-request timeout in seconds (default: 90)
TEST_SESSION_ID        Optional fixed session ID (auto-generated if omitted)
"""

import json
import os
import subprocess
import sys
import time
import uuid
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_URL = os.environ.get("CLOUD_RUN_URL", "").rstrip("/")
APP_NAME = os.environ.get("TEST_APP_NAME", "cognitive_search_agent")
TIMEOUT  = int(os.environ.get("TEST_TIMEOUT_S", "90"))
SESSION  = os.environ.get("TEST_SESSION_ID", f"smoke-{uuid.uuid4().hex[:8]}")
USER_ID  = "smoke-test-user"

if not BASE_URL:
    sys.exit("ERROR: Set CLOUD_RUN_URL before running. e.g. https://fre-xxxx-uc.a.run.app")

# Get identity token for authenticated Cloud Run (no-op for localhost)
def _get_auth_headers() -> dict:
    if BASE_URL.startswith("http://localhost") or BASE_URL.startswith("http://127"):
        return {}
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True, text=True
    )
    token = result.stdout.strip()
    if not token:
        sys.exit("ERROR: could not get identity token — run 'gcloud auth login' first")
    return {"Authorization": f"Bearer {token}"}

AUTH_HEADERS = _get_auth_headers()

PASS = "\033[32m PASS\033[0m"
FAIL = "\033[31m FAIL\033[0m"
WARN = "\033[33m WARN\033[0m"
results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    results.append((name, ok, detail))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def post_run(message: str, session_id: str = SESSION) -> dict:
    """POST to /run — waits for full JSON reply (non-streaming)."""
    url = f"{BASE_URL}/run"
    payload = {
        "app_name":  APP_NAME,
        "user_id":   USER_ID,
        "session_id": session_id,
        "new_message": {
            "role": "user",
            "parts": [{"text": message}],
        },
        "streaming": False,
    }
    resp = requests.post(url, json=payload, headers=AUTH_HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def extract_text(run_response: dict) -> str:
    """Pull the final text reply out of a /run response."""
    events = run_response if isinstance(run_response, list) else [run_response]
    texts = []
    for event in events:
        content = event.get("content", {})
        for part in content.get("parts", []):
            if isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])
    return " ".join(texts)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Health / list-apps
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 1: /list-apps (service is up) ===")
t0 = time.perf_counter()
try:
    r = requests.get(f"{BASE_URL}/list-apps", headers=AUTH_HEADERS, timeout=15)
    elapsed = time.perf_counter() - t0
    ok = r.status_code == 200 and APP_NAME in r.text
    record("list-apps returns 200", r.status_code == 200, f"HTTP {r.status_code}")
    record(f"app '{APP_NAME}' is registered", APP_NAME in r.text)
    record(f"response time < 5s", elapsed < 5, f"{elapsed:.1f}s")
except Exception as exc:
    record("list-apps reachable", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Status query (router agent, no search backend needed)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 2: status query (router + tool call) ===")
t0 = time.perf_counter()
try:
    resp = post_run("what is the system status?", session_id=f"{SESSION}-t2")
    elapsed = time.perf_counter() - t0
    text = extract_text(resp)
    record("status query completes", True, f"{elapsed:.1f}s")
    record("response not empty", len(text) > 20, f"{len(text)} chars")
    record(f"response time < {TIMEOUT}s", elapsed < TIMEOUT, f"{elapsed:.1f}s")
    if elapsed < TIMEOUT:
        print(f"    Reply preview: {text[:200].strip()!r}")
except requests.exceptions.Timeout:
    elapsed = time.perf_counter() - t0
    record("status query completes", False, f"TIMEOUT after {elapsed:.0f}s")
except Exception as exc:
    record("status query completes", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Document Q&A (full pipeline: ES + Vertex + VAIS + synthesis)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 3: document Q&A (full RAG pipeline) ===")
t0 = time.perf_counter()
try:
    resp = post_run(
        "what tests were done with low oil levels?",
        session_id=f"{SESSION}-t3",
    )
    elapsed = time.perf_counter() - t0
    text = extract_text(resp)
    record("Q&A pipeline completes", True, f"{elapsed:.1f}s")
    record("response not empty", len(text) > 50, f"{len(text)} chars")
    record(f"response time < {TIMEOUT}s", elapsed < TIMEOUT, f"{elapsed:.1f}s")
    record("response contains a filename citation",
           ".pdf" in text or "Source:" in text, "checked for .pdf or Source:")
    # Check for Rule 7 — image pages listed if images found
    has_image_line = "📷" in text or "Image" in text or "page" in text.lower()
    record("Rule 7 image line present (if images in results)", has_image_line,
           "📷 line or image reference found" if has_image_line else "no image references — may be OK if no images in results")
    print(f"    Reply preview: {text[:300].strip()!r}")
except requests.exceptions.Timeout:
    elapsed = time.perf_counter() - t0
    record("Q&A pipeline completes", False, f"TIMEOUT after {elapsed:.0f}s — proxy bottleneck present" if "localhost" not in BASE_URL else f"TIMEOUT after {elapsed:.0f}s")
except Exception as exc:
    record("Q&A pipeline completes", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Preview follow-up (batch preview tool)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 4: preview follow-up (preview_documents_batch) ===")
t0 = time.perf_counter()
try:
    # Reuse session from TEST 3 so synthesis_agent has the 📷 line in history
    resp = post_run(
        "can you preview the relevant pages",
        session_id=f"{SESSION}-t3",  # same session — history has the 📷 line
    )
    elapsed = time.perf_counter() - t0
    text = extract_text(resp)
    record("preview follow-up completes", True, f"{elapsed:.1f}s")
    record(f"response time < {TIMEOUT}s", elapsed < TIMEOUT, f"{elapsed:.1f}s")
    # A successful batch preview returns image markdown starting with "!["
    has_image_md = "![" in text
    record("image markdown rendered", has_image_md,
           "found '![' in reply" if has_image_md else "no image markdown — Step 0 may not have triggered")
    print(f"    Reply preview: {text[:300].strip()!r}")
except requests.exceptions.Timeout:
    elapsed = time.perf_counter() - t0
    record("preview follow-up completes", False, f"TIMEOUT after {elapsed:.0f}s")
except Exception as exc:
    record("preview follow-up completes", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Latency regression check (proxy vs direct)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== TEST 5: latency regression (proxy vs direct) ===")
proxy_in_use = "localhost" not in BASE_URL and not os.environ.get("NO_PROXY_TEST")
t0 = time.perf_counter()
try:
    resp = post_run(
        "how many documents are indexed?",
        session_id=f"{SESSION}-t5",
    )
    elapsed = time.perf_counter() - t0
    # On Cloud Run (no proxy): expect <20s
    # Local with Bosch proxy: warn if >35s (approaching kill threshold)
    threshold_warn  = 35
    threshold_fail  = int(TIMEOUT)
    record("latency check completes", True, f"{elapsed:.1f}s")
    if elapsed > threshold_fail:
        record(f"latency < {threshold_fail}s (hard limit)", False, f"{elapsed:.1f}s — DISCONNECTED")
    elif elapsed > threshold_warn and proxy_in_use:
        print(f"  [{WARN}] latency {elapsed:.1f}s > {threshold_warn}s — approaching proxy kill threshold")
        results.append(("latency within proxy safe zone", False, f"{elapsed:.1f}s"))
    else:
        record(f"latency < {threshold_warn}s", True, f"{elapsed:.1f}s")
except requests.exceptions.Timeout:
    elapsed = time.perf_counter() - t0
    record(f"latency < {threshold_fail}s (hard limit)", False, f"TIMEOUT {elapsed:.0f}s")
except Exception as exc:
    record("latency check", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("  \033[32mAll tests passed.\033[0m")
else:
    print("  \033[31mSome tests failed — see details above.\033[0m")
    for name, ok, detail in results:
        if not ok:
            print(f"    FAILED: {name}" + (f" ({detail})" if detail else ""))
print("=" * 60)
sys.exit(0 if passed == total else 1)
