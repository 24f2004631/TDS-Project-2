# Updated main.py - patched with fixes (Playwright sync, retries, relative URLs, feedback)
# Generated: 2025-11-29

import os
import sys
import json
import time
import uuid
import hmac
import re
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

# -------------------------
# Logging
# -------------------------
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

log = logging.getLogger("p2")

# -------------------------
# Configuration
# -------------------------
load_dotenv()
EMAIL = os.getenv("EMAIL", "")
SECRET_KEY = os.getenv("SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

JOB_TIMEOUT_SEC = 180         # per-question budget
RETRY_MARGIN_SEC = 50         # only retry if time_left > this (unless final-allowed)
SAFETY_MARGIN_SEC = 5
MAX_EXEC_TIMEOUT_SEC = 30     # uv run timeout for generated code

os.makedirs("Scrapped", exist_ok=True)
os.makedirs("LLM", exist_ok=True)

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="TDS P2 Solver (patched)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: FastAPIRequest, exc: RequestValidationError):
    errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=400, content={"detail": errors})

# -------------------------
# Models
# -------------------------
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
    type: str                       # "answer" or "code"
    answer: Optional[Any] = None
    code: Optional[str] = None
    submission_url: Optional[str] = None

# -------------------------
# Utilities
# -------------------------
def constant_time_compare(a: str, b: str) -> bool:
    if a is None or b is None:
        return False
    try:
        return hmac.compare_digest(a, b)
    except Exception:
        return False

def resolve_relative(base: str, link: Optional[str]) -> Optional[str]:
    if not link:
        return None
    link = link.strip()
    if link.startswith("http://") or link.startswith("https://"):
        return link
    # handle javascript: and mailto:
    if link.startswith("javascript:") or link.startswith("mailto:") or link.startswith("data:"):
        return None
    try:
        return urljoin(base, link)
    except Exception:
        return None

def validate_target_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="URL must be http/https")
    # basic SSRF-ish check: disallow localhost and private IPs
    host = parsed.hostname or ""
    private_prefixes = ("localhost", "127.", "10.", "192.168.", "169.254.", "172.")
    if any(host.startswith(p) for p in private_prefixes):
        raise HTTPException(status_code=400, detail="Disallowed hostname")
    return parsed

# -------------------------
# Scraper (sync Playwright) - improved to avoid networkidle hang and extract audio/cutoff
# -------------------------
def Scraper(url: str) -> str:
    log.info(f"[Scraper] Visiting: {url}")
    validate_target_url(url)
    screenshot_path = f"Scrapped/screenshot_{uuid.uuid4().hex}.png"
    html = ""
    text = ""
    text_full = ""
    submit_url = None
    cutoff_value = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()
        try:
            # don't wait for networkidle as some pages stream forever
            page.goto(url, timeout=30000)
            page.wait_for_load_state("domcontentloaded", timeout=10000)
            page.wait_for_timeout(1200)  # let client scripts run
            # try waiting for likely selectors
            try:
                page.wait_for_selector("pre, #result, body", timeout=4000)
            except:
                pass
        except Exception as e:
            log.info(f"[Scraper] goto/load warning: {e}")

        # collect content
        try:
            html = page.content()
        except:
            html = ""
        try:
            # inner_text on body gets visible text; use locator for more robust behavior
            text = page.locator("body").inner_text(timeout=2000)
        except:
            text = ""
        try:
            # full text capture for patterns like "Cutoff: 21815"
            text_full = page.locator("body").inner_text(timeout=2000)
        except:
            text_full = text

        # screenshot
        try:
            page.screenshot(path=screenshot_path)
        except Exception as e:
            log.info(f"[Scraper] screenshot warning: {e}")

        # heuristics for submit url
        try:
            node = page.query_selector("[data-submit-url]")
            if node:
                raw = node.get_attribute("data-submit-url")
                submit_url = resolve_relative(url, raw)
        except:
            pass

        if not submit_url:
            try:
                form = page.query_selector("form")
                if form:
                    action = form.get_attribute("action")
                    submit_url = resolve_relative(url, action)
            except:
                pass

        if not submit_url:
            try:
                for a in page.query_selector_all("a"):
                    try:
                        href = a.get_attribute("href")
                        if href and "submit" in href.lower():
                            submit_url = resolve_relative(url, href)
                            break
                    except:
                        continue
            except:
                pass

        # final fallback: find first http in body text
        if not submit_url:
            m = re.search(r"https?://[^\s\"'<>]+", text_full)
            if m:
                submit_url = m.group(0)

        # extract cutoff number if present
        m = re.search(r"Cutoff\s*[:\-]?\s*(\d+)", text_full, flags=re.I)
        if m:
            cutoff_value = m.group(1)

        try:
            browser.close()
        except:
            pass

    payload = {
        "html": html,
        "text": text,
        "text_full": text_full,
        "screenshot": screenshot_path,
        "submit_url": submit_url,
        "cutoff": cutoff_value
    }
    return json.dumps(payload)

# -------------------------
# LLM call (Gemini) - robust parsing
# -------------------------
def load_system_prompt() -> str:
    if os.path.exists("system-instruction1.txt"):
        with open("system-instruction1.txt", "r", encoding="utf-8") as f:
            return f.read()
    return "You are an expert data-extraction and self-correction agent. Output JSON strictly."

def LLMCode(
    scrapped_data: str,
    page_url: str,
    prev_response: Optional[LLMResponse] = None,
    server_response: Optional[ServerResponse] = None,
    stdout_stderr: Optional[str] = None
) -> LLMResponse:

    # lazy import
    try:
        from google.genai import Client
        from google.genai.types import Content, Part, GenerateContentConfig
    except Exception as e:
        raise RuntimeError(f"google.genai not available: {e}")

    scr = json.loads(scrapped_data)
    page_text = scr.get("text", "")
    html_text = scr.get("html", "")
    screenshot = scr.get("screenshot")

    user_prompt = f"""
You are an expert agent. Origin URL: {page_url}

PAGE_TEXT:
{page_text[:9000]}

HTML:
{html_text[:9000]}

If a submission endpoint is visible, DO NOT invent others.
Return either:
- {{ "type":"answer","answer": <value>, "submission_url": "<url_or_null>" }}
or
- {{ "type":"code","code":"<python3 code>", "submission_url":"<url_or_null>" }}

If this is a retry, use prev_response/server_response/stdout_stderr to fix errors.
"""

    if any([prev_response, server_response, stdout_stderr]):
        diag = {
            "prev_response": prev_response.model_dump() if prev_response else None,
            "server_response": server_response.model_dump() if server_response else None,
            "stdout_stderr": stdout_stderr
        }
        user_prompt += "\n\nDIAGNOSTIC:\n" + json.dumps(diag, default=str)[:8000]

    system_prompt = load_system_prompt()

    parts = [Part.from_text(text=user_prompt)]
    try:
        if screenshot and os.path.exists(screenshot):
            with open(screenshot, "rb") as f:
                parts.append(Part.from_bytes(data=f.read(), mime_type="image/png"))
    except:
        pass

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

    # extract text robustly
    text_out = None
    if hasattr(resp, "text") and resp.text:
        text_out = resp.text
    else:
        try:
            cand = getattr(resp, "candidates", [None])[0]
            text_out = getattr(cand.content.parts[0], "text", None)
        except:
            text_out = None

    if not text_out:
        text_out = str(resp)

    try:
        parsed = json.loads(text_out)
    except Exception as e:
        raise RuntimeError(f"LLM returned non-JSON: {text_out[:800]}") from e

    try:
        llm_resp = LLMResponse.model_validate(parsed)
    except Exception as e:
        raise RuntimeError(
            f"LLM JSON schema mismatch: {e}. Parsed keys: {list(parsed.keys())}"
        ) from e

    return llm_resp

# -------------------------
# Run code produced by LLM
# -------------------------
def RunLLMCode(code: str, timeout: int) -> tuple[str, str]:
    path = "LLM/llm_code.py"
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        proc = subprocess.run(
            ["uv", "run", path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return "", "timeout"
    except Exception as e:
        return "", f"exec_error:{e}"

# -------------------------
# Send response - robust fallback on malformed server responses
# -------------------------
def SendResponse(page_url: str, answer: Any, submission_url: Optional[str]) -> ServerResponse:
    if not submission_url:
        return ServerResponse(correct=False, url=None, reason="submission_url missing")

    final = FinalResponse(email=EMAIL, secret=SECRET_KEY, url=page_url, answer=answer)
    body = final.model_dump()

    # payload size check
    try:
        body_bytes = json.dumps(body, default=str).encode()
        if len(body_bytes) > 1_000_000:
            return ServerResponse(correct=False, url=None, reason="Payload > 1MB")
    except Exception:
        pass

    try:
        r = httpx.post(submission_url, json=body, timeout=30)
    except Exception as e:
        return ServerResponse(correct=False, url=None, reason=f"Network error: {e}")

    try:
        parsed = r.json()
    except Exception:
        return ServerResponse(correct=False, url=None, reason="Non-JSON server response")

    # Normalize unexpected shapes
    if "correct" not in parsed:
        reason = parsed.get("reason") or parsed.get("error") or str(parsed)
        return ServerResponse(correct=False, url=parsed.get("url"), reason=reason)

    try:
        return ServerResponse.model_validate(parsed)
    except ValidationError:
        # fallback
        return ServerResponse(
            correct=parsed.get("correct", False),
            url=parsed.get("url"),
            reason=parsed.get("reason")
        )

# -------------------------
# Retry logic
# -------------------------
def RecheckAnswer(
    start_time: float,
    original_llm: LLMResponse,
    page_url: str,
    scrapped_data: str,
    server_response: Optional[ServerResponse],
    stdout_stderr: Optional[str]
) -> Optional[str]:

    def time_left():
        return JOB_TIMEOUT_SEC - (time.time() - start_time)

    # If low time AND server said incorrect → skip retries
    if time_left() <= RETRY_MARGIN_SEC and server_response and not server_response.correct:
        log.info("[RecheckAnswer] Low time and server_response.correct==False -> skipping retries")
        return server_response.url

    final_retry_allowed = (server_response is None and time_left() <= RETRY_MARGIN_SEC)
    last_next = server_response.url if server_response else None
    did_final_retry = False

    while True:
        if time_left() <= SAFETY_MARGIN_SEC:
            return last_next

        try:
            llm = LLMCode(
                scrapped_data, page_url,
                prev_response=original_llm,
                server_response=server_response,
                stdout_stderr=stdout_stderr
            )
        except Exception as e:
            log.info(f"[RecheckAnswer] LLM generation failed: {e}")
            if final_retry_allowed and not did_final_retry:
                did_final_retry = True
                continue
            return last_next

        # ========== DIRECT ANSWER ==========
        if llm.type == "answer":
            resp = SendResponse(page_url, llm.answer, llm.submission_url)
            log.info(f"[RecheckAnswer] Submitted direct answer -> server returned: {resp}")

            if resp.correct:
                return resp.url

            server_response = resp
            last_next = resp.url
            continue

        # ========== CODE PATH ==========
        remaining_time = time_left()
        exec_timeout = int(min(MAX_EXEC_TIMEOUT_SEC, max(1, remaining_time - SAFETY_MARGIN_SEC)))

        stdout, stderr = RunLLMCode(llm.code, timeout=exec_timeout)

        if stderr:
            stdout_stderr = stderr
            log.info(f"[RecheckAnswer] Code stderr: {stderr[:300]}")

            if final_retry_allowed and not did_final_retry:
                did_final_retry = True
                continue

            continue

        if stdout:
            try:
                parsed = json.loads(stdout.strip())
            except Exception:
                stdout_stderr = stdout
                log.info("[RecheckAnswer] Code produced non-json output")

                if final_retry_allowed and not did_final_retry:
                    did_final_retry = True
                    continue
                continue

            if "answer" not in parsed:
                stdout_stderr = stdout

                if final_retry_allowed and not did_final_retry:
                    did_final_retry = True
                    continue

                continue

            resp = SendResponse(page_url, parsed["answer"], llm.submission_url)
            log.info(f"[RecheckAnswer] Submitted code answer -> server returned: {resp}")

            if resp.correct:
                return resp.url

            server_response = resp
            last_next = resp.url
            continue

        # ========== FALLBACK FINAL RETRY ==========
        if final_retry_allowed and not did_final_retry:
            did_final_retry = True
            continue

        return last_next

# -------------------------
# Main worker
# -------------------------
def HandleRequest(start_url: str, start_time: float):
    current_url = start_url
    question_start = start_time

    while current_url:
        remaining = JOB_TIMEOUT_SEC - (time.time() - question_start)
        if remaining <= SAFETY_MARGIN_SEC:
            log.info("[HandleRequest] Out of time for this question - moving on.")
            return

        log.info(f"[HandleRequest] time_left={remaining:.1f} for {current_url}")

        # ------------ SCRAPE ------------
        try:
            scrap = Scraper(current_url)
        except Exception as e:
            log.info(f"[HandleRequest] Scraper failed: {e}")
            return

        # ------------ FIRST LLM CALL ------------
        try:
            llm = LLMCode(scrap, current_url)
        except Exception as e:
            log.info(f"[HandleRequest] LLM generation failed: {e}")
            return

        server_resp = None
        code_stdout = None
        code_stderr = None

        # ------------ DIRECT ANSWER CASE ------------
        if llm.type == "answer":
            page_submit = json.loads(scrap).get("submit_url") or llm.submission_url
            server_resp = SendResponse(current_url, llm.answer, page_submit)

        else:
            # ------------ CODE CASE ------------
            remaining = JOB_TIMEOUT_SEC - (time.time() - question_start)
            exec_timeout = int(min(MAX_EXEC_TIMEOUT_SEC, max(1, remaining - SAFETY_MARGIN_SEC)))

            stdout, stderr = RunLLMCode(llm.code, timeout=exec_timeout)
            code_stdout, code_stderr = stdout, stderr

            if stderr:
                log.info(f"[HandleRequest] LLM code stderr: {stderr[:300]}")

            if stdout:
                try:
                    parsed = json.loads(stdout.strip())
                    if isinstance(parsed, dict) and "answer" in parsed:
                        page_submit = json.loads(scrap).get("submit_url") or llm.submission_url
                        server_resp = SendResponse(current_url, parsed["answer"], page_submit)
                except Exception:
                    pass

        # ------------ FEEDBACK LOGGING ------------
        if server_resp:
            if server_resp.correct:
                log.info("[Feedback] ✔ Answer CORRECT.")
            else:
                log.info(f"[Feedback] ✘ Answer INCORRECT. Reason: {server_resp.reason}")
        else:
            log.info("[Feedback] ⚠ No server response (submission may have failed).")

        # ------------ INCORRECT or FAILED SUBMISSION ------------
        if not server_resp or not server_resp.correct:
            next_url = RecheckAnswer(
                question_start,
                llm,
                current_url,
                scrap,
                server_resp,
                (code_stderr or code_stdout)
            )

            if next_url and next_url != current_url:
                log.info("[HandleRequest] Moving to next question (server provided new URL).")
                current_url = next_url
                question_start = time.time()
                continue

            if server_resp and server_resp.url:
                log.info("[HandleRequest] Moving to server-provided next URL despite incorrect/no correct.")
                current_url = server_resp.url
                question_start = time.time()
                continue

            log.info("[HandleRequest] Exhausted retries. Moving on.")
            return

        # ------------ CORRECT PATH ------------
        log.info("[HandleRequest] Correct answer → next question")

        if server_resp.url:
            current_url = server_resp.url
            question_start = time.time()
            continue

        log.info("[HandleRequest] Quiz finished (no next URL).")
        return

# -------------------------
# Endpoint
# -------------------------
@app.post("/p2")
def P2(payload: RequestModel):
    # Validate secret
    if not constant_time_compare(payload.secret, SECRET_KEY):
        raise HTTPException(status_code=403, detail="Invalid secret")

    # Validate target URL
    validate_target_url(payload.url)

    # Start worker
    start = time.time()
    p = Process(target=HandleRequest, args=(payload.url, start), daemon=True)
    p.start()

    return JSONResponse({"message": "Task started"})

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import uvicorn
    log.info("Starting server on :8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
