# ================================
#   Tools in Data Science — Project 2
#   Fully patched, stable main.py
#   Version: 2025-02 (Final)
# ================================

import os
import sys
import json
import time
import uuid
import hmac
import httpx
import subprocess
from typing import Any, Optional
from urllib.parse import urlparse, urljoin

from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, ValidationError
from playwright.sync_api import sync_playwright
from multiprocessing import Process
from dotenv import load_dotenv


# ============================================================
# Load configuration
# ============================================================

load_dotenv()
EMAIL = os.getenv("EMAIL", "")
SECRET_KEY = os.getenv("SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

JOB_TIMEOUT_SEC = 180          # Per-question time budget
RETRY_MARGIN_SEC = 50          # Only retry if > 50s left
SAFETY_MARGIN_SEC = 5          # Stop retrying if <5s left
MAX_EXEC_TIMEOUT_SEC = 30      # Timeout for "uv run LLM/llm_code.py"

os.makedirs("Scrapped", exist_ok=True)
os.makedirs("LLM", exist_ok=True)


# ============================================================
# FastAPI setup
# ============================================================

app = FastAPI(title="TDS Project-2 Solver", description="Robust FastAPI solver for TDS P2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: FastAPIRequest, exc: RequestValidationError):
    errors = [{"loc": err["loc"], "msg": err["msg"], "type": err["type"]} for err in exc.errors()]
    return JSONResponse(status_code=400, content={"detail": errors})


# ============================================================
# Models
# ============================================================

class RequestModel(BaseModel):
    email: str
    secret: str
    url: str


class FinalResponse(RequestModel):
    answer: Any


class ServerResponse(BaseModel):
    correct: bool
    url: Optional[str] = None
    reason: Optional[str] = None


class LLMResponse(BaseModel):
    type: str                           # "answer" or "code"
    answer: Optional[Any] = None
    code: Optional[str] = None
    submission_url: Optional[str] = None


# ============================================================
# Utility functions
# ============================================================

def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


def resolve_relative(base_url: str, candidate: str) -> str:
    """
    Convert relative URLs like '/submit' into absolute URLs.
    """
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate
    return urljoin(base_url, candidate)


def validate_target_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="URL must be http/https")
    return parsed


# ============================================================
# Scraper
# ============================================================

def Scraper(url: str) -> str:
    print(f"[Scraper] Visiting: {url}")

    validate_target_url(url)
    screenshot_path = f"Scrapped/screenshot_{uuid.uuid4().hex}.png"

    html, text, submit_url = "", "", None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()

        try:
            page.goto(url, timeout=30000)
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(1500)  # allow client JS to run
            # Bonus: wait for <pre> or '#result'
            try:
                page.wait_for_selector("pre, #result", timeout=4000)
            except:
                pass
        except Exception as e:
            print(f"[Scraper] goto warning: {e}")

        html = page.content()
        try:
            text = page.inner_text("body")
        except Exception:
            text = ""

        try:
            page.screenshot(path=screenshot_path)
        except:
            pass

        # === Extract submit URL ===
        # Priority 1: data-submit-url
        node = page.query_selector("[data-submit-url]")
        if node:
            try:
                submit_url = node.get_attribute("data-submit-url")
                submit_url = resolve_relative(url, submit_url)
            except:
                pass

        # Priority 2: any <form action="...">
        if not submit_url:
            form = page.query_selector("form")
            if form:
                try:
                    action = form.get_attribute("action")
                    submit_url = resolve_relative(url, action)
                except:
                    pass

        # Priority 3: any <a href="...submit...">
        if not submit_url:
            for a in page.query_selector_all("a"):
                try:
                    href = a.get_attribute("href")
                    if href and "submit" in href.lower():
                        submit_url = resolve_relative(url, href)
                        break
                except:
                    continue

        # Priority 4: look for printed text "POST this JSON to <url>"
        if not submit_url and "POST this JSON to" in text:
            import re
            m = re.search(r"https?://[^\s\"'<>]+", text)
            if m:
                submit_url = m.group(0)

        browser.close()

    return json.dumps({
        "html": html,
        "text": text,
        "screenshot": screenshot_path,
        "submit_url": submit_url
    })


# ============================================================
# LLM handling
# ============================================================

def load_system_prompt() -> str:
    try:
        with open("system-instruction1.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "You are a data-extraction and code-generation agent. Follow JSON schema strictly."


def LLMCode(scrapped_data: str, page_url: str,
            prev_response: Optional[LLMResponse] = None,
            server_response: Optional[ServerResponse] = None,
            stdout_stderr: Optional[str] = None) -> LLMResponse:
    """
    Calls Gemini and parses LLM output safely.
    """

    # Late import to avoid overhead unless needed
    from google.genai import Client
    from google.genai.types import Content, Part, GenerateContentConfig

    scr = json.loads(scrapped_data)
    page_text = scr.get("text", "")
    html_text = scr.get("html", "")
    screenshot_path = scr.get("screenshot")

    # Build user prompt
    user_prompt = f"""
You are an expert solver. 
Origin URL: {page_url}

PAGE_TEXT:
{page_text[:10000]}

HTML_CONTENT:
{html_text[:10000]}

Your job:
- Identify what the page is asking.
- If answer is trivial → return type="answer".
- If complex → return type="code" with a Python script that prints JSON {{"answer": ...}}.
- Never invent the submission URL. If missing, say null.
"""

    if any([prev_response, server_response, stdout_stderr]):
        diagnostics = {
            "prev_response": prev_response.model_dump() if prev_response else None,
            "server_response": server_response.model_dump() if server_response else None,
            "stdout_stderr": stdout_stderr
        }
        user_prompt += "\n\nDIAGNOSTICS:\n" + json.dumps(diagnostics, default=str)[:8000]

    system_prompt = load_system_prompt()

    parts = [Part.from_text(text=user_prompt)]
    if screenshot_path and os.path.exists(screenshot_path):
        with open(screenshot_path, "rb") as f:
            img = f.read()
            parts.append(Part.from_bytes(data=img, mime_type="image/png"))

    client = Client(api_key=GEMINI_API_KEY)

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[Content(role="user", parts=parts)],
            config=GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

    # Extract text robustly
    text_out = None
    try:
        if hasattr(resp, "text") and resp.text:
            text_out = resp.text
        else:
            cand = resp.candidates[0]
            part = cand.content.parts[0]
            text_out = getattr(part, "text", None)
    except:
        pass

    if not text_out:
        text_out = str(resp)

    # Parse as JSON
    try:
        parsed = json.loads(text_out)
    except:
        raise RuntimeError(f"LLM returned non-JSON: {text_out[:500]}")

    # Validate LLMResponse
    return LLMResponse.model_validate(parsed)


# ============================================================
# Code execution (uv run)
# ============================================================

def RunLLMCode(code: str, timeout: int) -> tuple[str, str]:
    path = "LLM/llm_code.py"
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        p = subprocess.run(
            ["uv", "run", path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return "", "timeout"
    except Exception as e:
        return "", f"exec_error: {e}"


# ============================================================
# Submission (with safe fallback)
# ============================================================

def SendResponse(page_url: str, answer: Any, submission_url: Optional[str]) -> ServerResponse:
    if not submission_url:
        return ServerResponse(correct=False, url=None, reason="submission_url missing")

    final = FinalResponse(email=EMAIL, secret=SECRET_KEY, url=page_url, answer=answer)
    body = final.model_dump()

    if len(json.dumps(body).encode()) > 1_000_000:
        return ServerResponse(correct=False, reason="Payload exceeds 1MB", url=None)

    try:
        r = httpx.post(submission_url, json=body, timeout=30)
    except Exception as e:
        return ServerResponse(correct=False, url=None, reason=f"Network error: {e}")

    # Try JSON parse
    try:
        parsed = r.json()
    except:
        return ServerResponse(correct=False, url=None, reason="Non-JSON response")

    # If standard fields missing, fallback
    if "correct" not in parsed:
        reason = parsed.get("reason") or parsed.get("error") or str(parsed)
        return ServerResponse(correct=False, url=parsed.get("url"), reason=reason)

    try:
        return ServerResponse.model_validate(parsed)
    except ValidationError:
        return ServerResponse(
            correct=parsed.get("correct", False),
            url=parsed.get("url"),
            reason=parsed.get("reason", "Malformed server response")
        )


# ============================================================
# Retry Logic
# ============================================================

def RecheckAnswer(start_time: float, original_llm: LLMResponse, page_url: str,
                  scrapped_data: str, server_response: Optional[ServerResponse],
                  stdout_stderr: Optional[str]) -> Optional[str]:

    def time_left():
        return JOB_TIMEOUT_SEC - (time.time() - start_time)

    # If <50s left and we DID receive server_response with correct=False → skip retries
    if time_left() <= RETRY_MARGIN_SEC and server_response and not server_response.correct:
        return server_response.url

    # Allow one final retry if no server_response ever arrived
    final_retry_allowed = (server_response is None and time_left() <= RETRY_MARGIN_SEC)

    last_next_url = server_response.url if server_response else None

    while True:

        if time_left() <= SAFETY_MARGIN_SEC:
            return last_next_url

        # Build LLM call with diagnostics
        try:
            llm = LLMCode(
                scrapped_data,
                page_url,
                prev_response=original_llm,
                server_response=server_response,
                stdout_stderr=stdout_stderr
            )
        except Exception as e:
            if final_retry_allowed:
                final_retry_allowed = False
                continue
            return last_next_url

        # Handle direct answer
        if llm.type == "answer":
            resp = SendResponse(page_url, llm.answer, llm.submission_url)
            if resp.correct:
                return resp.url
            server_response = resp
            last_next_url = resp.url
            continue

        # Handle code generation
        stdout, stderr = RunLLMCode(
            llm.code,
            timeout=int(min(MAX_EXEC_TIMEOUT_SEC, time_left()))
        )

        if stderr:
            stdout_stderr = stderr
            if final_retry_allowed:
                final_retry_allowed = False
                continue
            continue

        if stdout:
            try:
                parsed = json.loads(stdout.strip())
            except:
                stdout_stderr = stdout
                continue

            if "answer" not in parsed:
                stdout_stderr = stdout
                continue

            resp = SendResponse(page_url, parsed["answer"], llm.submission_url)
            if resp.correct:
                return resp.url

            server_response = resp
            last_next_url = resp.url
            continue

        # If we reach here, nothing worked
        if final_retry_allowed:
            final_retry_allowed = False
            continue

        return last_next_url


# ============================================================
# Main question-chain worker (spawned via Process)
# ============================================================

def HandleRequest(start_url: str, start_time: float):

    current_url = start_url
    question_start = start_time   # reset ONLY when correct answer received

    while current_url:

        remaining = JOB_TIMEOUT_SEC - (time.time() - question_start)
        if remaining <= SAFETY_MARGIN_SEC:
            print("[HandleRequest] Out of time for this question.")
            return

        print(f"[HandleRequest] time_left={remaining:.1f} for {current_url}")

        # Scrape fresh version of the page
        try:
            scrap = Scraper(current_url)
        except Exception as e:
            print(f"[HandleRequest] Scraper error: {e}")
            return

        # Initial LLM attempt
        try:
            llm = LLMCode(scrap, current_url)
        except Exception as e:
            print(f"[HandleRequest] LLM error: {e}")
            return

        server_resp = None
        stdout_err = None

        if llm.type == "answer":
            page_submit = json.loads(scrap).get("submit_url") or llm.submission_url
            server_resp = SendResponse(current_url, llm.answer, page_submit)

        else:
            stdout, stderr = RunLLMCode(llm.code, timeout=int(min(MAX_EXEC_TIMEOUT_SEC, remaining)))
            stdout_err = stderr or stdout

            if stdout:
                try:
                    parsed = json.loads(stdout.strip())
                    if "answer" in parsed:
                        page_submit = json.loads(scrap).get("submit_url") or llm.submission_url
                        server_resp = SendResponse(current_url, parsed["answer"], page_submit)
                except:
                    pass

        # If server did not accept or response was malformed → retry logic
        if not server_resp or not server_resp.correct:
            next_url = RecheckAnswer(
                question_start,
                llm,
                current_url,
                scrap,
                server_resp,
                stdout_err
            )

            if next_url and next_url != current_url:
                # Move to next question
                current_url = next_url
                question_start = time.time()   # reset timer
                continue

            # If we reached here and have server_resp with url → move on anyway
            if server_resp and server_resp.url:
                current_url = server_resp.url
                question_start = time.time()
                continue

            print("[HandleRequest] Exhausted retries. Moving on.")
            return

        # Answer was correct → move on
        print("[HandleRequest] Correct answer → next question")
        if server_resp.url:
            current_url = server_resp.url
            question_start = time.time()
            continue

        print("[HandleRequest] Quiz complete.")
        return


# ============================================================
# FastAPI endpoint
# ============================================================

@app.post("/p2")
def P2(payload: RequestModel):
    if not constant_time_compare(payload.secret, SECRET_KEY):
        raise HTTPException(status_code=403, detail="Invalid secret")

    validate_target_url(payload.url)

    start = time.time()
    p = Process(target=HandleRequest, args=(payload.url, start), daemon=True)
    p.start()

    return JSONResponse({"message": "Task started"})


# ============================================================
# Launch server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting server on port 8000…")
    uvicorn.run(app, host="0.0.0.0", port=8000)
