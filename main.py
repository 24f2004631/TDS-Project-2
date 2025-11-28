import os
import sys
import time
import json
import httpx
import subprocess
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from dotenv import load_dotenv
from pydantic import BaseModel
from playwright.sync_api import sync_playwright

from google.genai import Client
from google.genai.types import Content, Part, GenerateContentConfig


# Secrets------->

load_dotenv()

EMAIL = os.getenv("EMAIL", "")
SECRET_KEY = os.getenv("SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# Time Variables ---------->

JOB_TIMEOUT_SEC = 180
RETRY_MARGIN_SEC = 50
SAFETY_MARGIN_SEC = 5


# Api Intialization ----->

app = FastAPI(
    title="Tools in Data Science Project-2",
    description="This api is creted to accomplish all the task in Project-2 of TDS." 
)

# Adding Cross Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Custom Exception Handler for Validation Error
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: FastAPIRequest, exc: RequestValidationError):
    """
    Custom exception handler to return 400 instead of 422 on validation errors.
    """
    errors = []
    for error in exc.errors():
        errors.append({"loc": error["loc"], "msg": error["msg"], "type": error["type"]})
    return JSONResponse(status_code=400, content={"detail": errors})


#region Pydantic Class

class Request(BaseModel):
    email: str
    secret: str
    url: str

class FinalResponse(Request):
    answer: Any

class Response(BaseModel):
    message: str

class ServerResponse(BaseModel):
    correct: bool
    url: Optional[str]
    reason: Optional[str]

class LLMResponse(BaseModel):
    type: str
    answer: Optional[Any]
    code: Optional[str]
    submission_url: str

#endregion


#region Helper Function

def CheckSecret(secret: str):
    if secret != SECRET_KEY:
        raise HTTPException(status_code=403, detail=f"Invalid secret key.")
    return

def Scraper(url: str):
    print("Starting data scrapping...")
    os.makedirs("Scrapped", exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

        html = page.content()
        text = page.inner_text("body")

        page.screenshot(path="Scrapped/screenshot.png")

        browser.close()
    return json.dumps({"html": html, "text": text})


def LLMCode(scrapped_data: str, url: str, prev_response: Optional[LLMResponse] = None, server_response: Optional[ServerResponse] = None, stdout_stderr: Optional[str] = None) -> LLMResponse:

    scrapped_json = json.loads(scrapped_data)

    user_prompt = f"""
    You will receive the following inputs:

    - **page_text** — Extracted visible text from the page or resource  
    {scrapped_json.get("text", "")}
    - **html_content** — Raw HTML if available  
    {scrapped_json.get("html", "")}
    - **screenshot** — Screenshot image of the resource (if page-based)  
    Provided as an attachment
    - **origin_url** — The URL where the question exists (may contain text, audio, images, API endpoints, etc.)
    {url}

    ### Your Tasks:

    1. **Locate the question** contained anywhere in the resource:  
       - in visible text  
       - HTML  
       - inside an image  
       - inside an audio transcription  
       - in embedded endpoints or dynamic content inside origin_url.

    2. **Determine if the question can be solved directly.**  
       - If yes → **Return ONLY the direct answer**.  
       - If no → **Generate a Python script to solve it**, making sure:
         - The script is fully runnable.
         - The script prints the final answer.
         - The metadata includes **only the dependencies actually used in the code**.

    3. **Do not hallucinate missing data.**  
       Rely only on the given page_text, html_content, screenshot, and origin_url.

    4. **Return either:**
       - A final answer  
       - OR a fully correct Python script that will compute it.

        """

    if any [prev_response, server_response, stdout_stderr]:
        prev_response_json = prev_response.model_dump_json() if prev_response else {}
        server_response_json = server_response.model_dump_json() if server_response else {}

        user_prompt = f"""
        Your previous response was incorrect or produced an execution error.  
        You are now provided with additional diagnostic information:

        - **page_text** 
        {scrapped_json.get("text", "")} 
        - **html_content**
        {scrapped_json.get("text", "")} 
        - **screenshot** - Provided as attachment.
        - **origin_url**  
        {url}
        - **LLM_response** — Your previous output *(if provided)*  
        {prev_response_json}
        - **server_response** — The server’s correctness feedback *(if provided)*  
        {server_response_json}
        - **stdout / stderr** — Output or traceback from executing your generated code *(if provided)*
        {stdout_stderr}


        If **any** of LLM_response, server_response, or stdout/stderr is missing, **use whatever fields are available**.

        ### Your Tasks:

        1. **Re-analyze the resource** to correctly identify the question present in the origin_url.  
           The question may be inside:
           - page text  
           - HTML  
           - an image  
           - audio  
           - a dynamically loaded endpoint  
           - any content referenced by origin_url  

        2. **Determine what went wrong** with your previous attempt using:
           - the previous LLM_response (if provided)  
           - server_response feedback (if provided)  
           - stdout/stderr traceback (if provided)

        3. **Fix the mistake** by producing:
           - the correct final answer  
           **OR**  
           - a corrected Python script that:
             - fixes the previous failure  
             - uses only necessary dependencies  
             - and prints the final answer successfully

        4. **Ensure the output is fully correct and runnable.**

        """

    with open("system-instruction1.txt", "r") as file:
        system_prompt = file.read()
    
    with open("Scrapped/screenshot.png", "rb") as img:
        img_bytes = img.read()

    client = Client(api_key=GEMINI_API_KEY)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                Content(
                    role="user",
                    parts=[
                        Part.from_text(text=user_prompt),
                        Part.from_bytes(data=img_bytes, mime_type="image/png")
                    ]
                )
            ],
            config=GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=LLMResponse.model_json_schema()
            )
        )
        generated_content = response.text
    except Exception as e:
        raise(e)
    
    try:
        llm_response = LLMResponse.model_validate_json(generated_content)
    except Exception as e:
        raise(e)
    
    return llm_response

def RunLLMCode(llm_response_code: str) -> tuple[str, str]:
    print("Running LLM Code...")
    
    os.makedirs("LLM", exist_ok=True)
    with open("LLM/llm_code.py", "w") as file:
        file.write(llm_response_code)
    
    result = subprocess.run(
        ["uv", "run", "LLM/llm_code.py"],
        capture_output=True,
        text=True,
        timeout=20
    )

    if result.returncode != 0:
        # The script failed.
        print(f"LLM code execution failed with exit code {result.returncode}.")

    return result.stdout, result.stderr

def SendResponse(payload_url: str, answer_value: Any, submission_url: str) -> ServerResponse:

    final_response = FinalResponse(
        email=EMAIL,
        secret=SECRET_KEY,
        url=payload_url,
        answer=answer_value
    )

    response = httpx.post(submission_url, json=final_response.model_dump())

    print(response.status_code)
    print(response.text)

    response_json = response.json()
    return ServerResponse(**response_json)

def RecheckAnswer(st_time: float,generated_data: LLMResponse, payload_url: str, scrapped_data: str, server_response: Optional[ServerResponse] = None, stdout_stderr: Optional[str] = None) -> str:

    time_left = lambda: JOB_TIMEOUT_SEC - (time.time() - st_time)

    server_next_url = server_response.url if server_response else None

    next_question = None

    while time_left() > RETRY_MARGIN_SEC:
        llm_response = None
        if server_response and stdout_stderr:
            llm_response = LLMCode(scrapped_data, payload_url, generated_data, server_response, stdout_stderr)
        elif server_response:
            llm_response = LLMCode(scrapped_data, payload_url, generated_data, server_response)
        elif stdout_stderr:
            llm_response = LLMCode(scrapped_data, payload_url, generated_data, stdout_stderr)
        else:
            llm_response = LLMCode(scrapped_data, payload_url, generated_data)
        
        if llm_response.type == "answer":
            retry_resopnse = SendResponse(payload_url, llm_response.answer, llm_response.submission_url)
            if retry_resopnse.correct == True:
                next_question = retry_resopnse.url
                break
            else:
                server_response = retry_resopnse
                generated_data = llm_response
                server_next_url = retry_resopnse.url
                continue
        else:
            script_output_str, script_error_str = RunLLMCode(llm_response.code)
            if script_error_str:
                stdout_stderr = script_error_str
                generated_data = llm_response
                continue
            if script_output_str:
                script_output_json = json.loads(script_output_str)
                if "error" in script_output_json:
                    stdout_stderr = script_output_json["error"]
                    generated_data = llm_response
                    continue
                else:
                    answer_value = script_output_json["answer"]
                    retry_resopnse = SendResponse(payload_url, answer_value, llm_response.submission_url)
                    if retry_resopnse.correct == True:
                        next_question = retry_resopnse.url
                        break
                    else:
                        server_response = retry_resopnse
                        generated_data = llm_response
                        server_next_url = retry_resopnse.url
                        continue
    
    if not next_question:
        next_question = server_next_url
    
    return next_question


def HandleRequest(payload_url: str, st_time: float):
    print("Recieved data in HandleRequest function...")
    time_left = lambda: JOB_TIMEOUT_SEC - (time.time() - st_time)


    question_url = payload_url

    while question_url and time_left() > SAFETY_MARGIN_SEC:
        scrapped_data = Scraper(question_url)
        try:
            generated_data = LLMCode(scrapped_data, question_url)
            print(generated_data)
        except Exception as e:
            print(f"Failed to get response from LLM: {e}")
            # If LLM fails, we cannot proceed with this question_url.
            break

        server_response = None
        code_output_str = None
        if generated_data.type == "answer":
            server_response = SendResponse(question_url, generated_data.answer, generated_data.submission_url)
        else:
            script_output_str, script_error_str = RunLLMCode(generated_data.code)
            
            if script_error_str:
                print(f"Error from LLM script execution (stderr):\n{script_error_str}")
                code_output_str = f"Stderr: {script_error_str}"

            if script_output_str:
                # Parse the JSON string from the script's output
                script_output_json = json.loads(script_output_str)
                code_output = script_output_str
                if "error" in script_output_json:
                    print(f"Error from executed LLM code: {script_output_json['error']}")
                elif "answer" in script_output_json:
                    answer_value = script_output_json["answer"]
                    print(f"Answer from executed LLM code: {answer_value}")
                    server_response = SendResponse(question_url, answer_value, generated_data.submission_url) 

        if server_response:
            if server_response.correct == False and code_output_str:
                question_url = RecheckAnswer(st_time, generated_data, payload_url, scrapped_data, server_response, code_output_str)
            elif server_response.correct == False:
                question_url = RecheckAnswer(st_time, generated_data, payload_url, scrapped_data, server_response)
            else:
                question_url = server_response.url      
        else:
            if code_output_str:
                question_url = RecheckAnswer(st_time, generated_data, payload_url, scrapped_data, stdout_stderr=code_output_str)
            else:
                question_url = RecheckAnswer(st_time, generated_data, payload_url, scrapped_data)
    
        st_time = time.time()

    print("Request Completed.")
    return

    
#endregion

@app.post("/p2")
def Project2(payload: Request, background_tasks: BackgroundTasks):
    st_time = time.time()
    
    try:
        CheckSecret(payload.secret)
    except HTTPException as e:
        raise(e)
    
    print(payload)
    
    background_tasks.add_task(HandleRequest, payload.url, st_time)

    response = Response(message="Task successfully accepted.")
    return JSONResponse(status_code=200, content=response.model_dump_json())

if __name__ == "__main__":
    import uvicorn
    print(f"Running Uvicorn. Shared Secret: {SECRET_KEY}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    