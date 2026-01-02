import os
import uuid
import json
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

# ENV VARIABLES


SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

TRANSLATOR_KEY = os.getenv("TRANSLATOR_KEY")
TRANSLATOR_ENDPOINT = os.getenv("TRANSLATOR_ENDPOINT")
TRANSLATOR_REGION = os.getenv("TRANSLATOR_REGION")

CONTENT_SAFETY_KEY = os.getenv("CONTENT_SAFETY_KEY")
CONTENT_SAFETY_ENDPOINT = os.getenv("CONTENT_SAFETY_ENDPOINT")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not SPEECH_KEY or not SPEECH_REGION:
    raise RuntimeError("Missing SPEECH_KEY or SPEECH_REGION in .env")

if not TRANSLATOR_KEY or not TRANSLATOR_ENDPOINT or not TRANSLATOR_REGION:
    raise RuntimeError("Missing TRANSLATOR_KEY / TRANSLATOR_ENDPOINT / TRANSLATOR_REGION in .env")


if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_KEY / AZURE_OPENAI_DEPLOYMENT in .env")

# Content Safety 
CONTENT_SAFETY_ENABLED = bool(CONTENT_SAFETY_KEY and CONTENT_SAFETY_ENDPOINT)

# FASTAPI APP


app = FastAPI(title="AI Customer Support - Multimodal Intake")

ALLOWED_LANGS = {"te-IN", "hi-IN", "en-US"}

LANG_MAP = {
    "te-IN": "te",
    "hi-IN": "hi",
    "en-US": "en"
}

# HELPERS


def translate_to_english(text: str, language: str) -> str:
    if language == "en-US":
        return text

    if language not in LANG_MAP:
        raise RuntimeError(f"LANG_MAP missing entry for language: {language}")

    params = {
        "api-version": "3.0",
        "from": LANG_MAP[language],
        "to": "en"
    }

    headers = {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
        "Content-Type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4())
    }

    body = [{"text": text}]

    resp = requests.post(
        TRANSLATOR_ENDPOINT.rstrip("/") + "/translate",
        params=params,
        headers=headers,
        json=body,
        timeout=10
    )
    resp.raise_for_status()

    data = resp.json()
    return data[0]["translations"][0]["text"]


def check_content_safety(text: str) -> dict:
    # If not configured, treat as safe
    if not CONTENT_SAFETY_ENABLED:
        return {
            "unsafe": False,
            "label": "safe",
            "severity": 0,
            "details": None
        }
    url = f"{CONTENT_SAFETY_ENDPOINT.rstrip('/')}/contentsafety/text:analyze?api-version=2023-10-01"

    headers = {
        "Ocp-Apim-Subscription-Key": CONTENT_SAFETY_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "text": text,
        "categories": ["Hate", "SelfHarm", "Sexual", "Violence"],
        "outputType": "FourSeverityLevels"
    }

    resp = requests.post(url, headers=headers, json=body, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    max_severity = max(
        (c.get("severity", 0) for c in data.get("categoriesAnalysis", [])),
        default=0
    )

    if max_severity >= 3:
        label = "unsafe"
    elif max_severity == 2:
        label = "angry_or_abusive"
    else:
        label = "safe"

    return {
        "unsafe": label == "unsafe",
        "label": label,
        "severity": max_severity,
        "details": data
    }




# LANGCHAIN SETUP (GLOBAL)


llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-05-01-preview",
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    temperature=0.2,
)

intent_prompt = ChatPromptTemplate.from_template("""
You are a Responsible AI Customer Support Assistant.

Your responsibilities:
1. Analyze the user message.
2. Identify ONE intent.
3. Assign a confidence score.
4. Decide whether you can safely provide a solution.

Important rules:
- If the message mentions more than one issue OR the intent is unclear,
  you MUST lower confidence below 0.6.
- Only provide a solution if the issue is common, clear, and low-risk.
- If you are unsure, complex, or lack sufficient information,
  DO NOT guess and set can_solve = false.

User message:
{input}

Supported intents:
- network_issue
- billing_issue
- service_complaint
- fraud_report
- general_query

Return ONLY valid JSON in this exact format:
{{
  "intent": "<intent>",
  "confidence": <number between 0 and 1>,
  "department": "<department>",
  "can_solve": true | false,
  "solution": "<step-by-step solution OR escalation message>",
  "summary": "<one sentence summary>"
}}
""")

intent_chain = intent_prompt | llm



def detect_intent_langchain(english_text: str) -> dict:
    response = intent_chain.invoke({"input": english_text})
    content = response.content

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(content)
    except Exception:
        return {
            "intent": "general_query",
            "confidence": 0.0,
            "department": "Manual Review",
            "summary": "Invalid LLM response",
            "raw": content
        }

def translate_from_english(text: str, target_language: str) -> str:
    if target_language == "en-US":
        return text

    LANG_MAP = {
        "te-IN": "te",
        "hi-IN": "hi"
    }

    params = {
        "api-version": "3.0",
        "from": "en",
        "to": LANG_MAP[target_language]
    }

    headers = {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
        "Content-Type": "application/json"
    }

    body = [{"text": text}]

    response = requests.post(
        TRANSLATOR_ENDPOINT.rstrip("/") + "/translate",
        params=params,
        headers=headers,
        json=body,
        timeout=10
    )

    response.raise_for_status()
    return response.json()[0]["translations"][0]["text"]

# HEALTH CHECK


@app.get("/")
def health():
    return {"status": "ok", "message": "Server running"}

# MAIN ENDPOINT

@app.post("/process-query")
async def process_query(
    language: str = Form(...),       
    text: str = Form(None),           
    audio: UploadFile = File(None)   
):


    # STEP 0: INPUT ISOLATION & AUTO-DETECTION


    if language not in ALLOWED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {language}"
        )

    has_text = text is not None and text.strip() != ""
    has_audio = audio is not None

    if has_text and has_audio:
        raise HTTPException(
            status_code=400,
            detail="Please provide either text OR audio, not both."
        )

    if not has_text and not has_audio:
        raise HTTPException(
            status_code=400,
            detail="Please provide text or audio input."
        )

    input_type = "voice" if has_audio else "text"

 
    # STEP 1: GET ORIGINAL TEXT

    if input_type == "text":
        original_text = text.strip()

    else:
        temp_name = f"temp_{uuid.uuid4().hex}.wav"

        try:
            content = await audio.read()
            with open(temp_name, "wb") as f:
                f.write(content)

            speech_config = speechsdk.SpeechConfig(
                subscription=SPEECH_KEY,
                region=SPEECH_REGION
            )
            speech_config.speech_recognition_language = language

            audio_config = speechsdk.AudioConfig(filename=temp_name)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            result = recognizer.recognize_once()

            if result.reason != speechsdk.ResultReason.RecognizedSpeech:
                return {
                    "selected_language": language,
                    "original_text": "",
                    "english_text": "",
                    "status": "manual_review",
                    "reason": "speech_not_recognized",
                    "details": str(result.reason),
                }

            original_text = result.text.strip()

        finally:
            try:
                recognizer = None
                audio_config = None
                import time
                time.sleep(0.2)
                if os.path.exists(temp_name):
                    os.remove(temp_name)
            except Exception:
                pass


    # STEP 2: TRANSLATE TO ENGLISH


    english_text = translate_to_english(original_text, language)


    # STEP 3: CONTENT SAFETY


    safety = check_content_safety(english_text)

    if safety["unsafe"]:
        return {
            "input_type": input_type,
            "selected_language": language,
            "original_text": original_text,
            "english_text": english_text,
            "status": "manual_review",
            "safety": safety
        }

    # STEP 4: LLM INTENT + SOLUTION

    llm_result = detect_intent_langchain(english_text)

    # STEP 5: DETERMINISTIC AMBIGUITY RULES



    if len(english_text.split()) < 3:
        llm_result.update({
            "intent": "general_query",
            "confidence": 0.0,
            "department": "Manual Review",
            "can_solve": False,
            "solution": "Ahh! We are sorry. Your query has been forwarded to the respective department.",
            "summary": "Input too short to determine intent reliably."
        })

    billing_keywords = ["bill", "payment", "charged", "invoice"]
    network_keywords = ["internet", "wifi", "network", "signal", "speed"]

    text_lower = english_text.lower()

    has_billing = any(w in text_lower for w in billing_keywords)
    has_network = any(w in text_lower for w in network_keywords)

    if has_billing and has_network:
        llm_result.update({
            "intent": "general_query",
            "confidence": 0.0,
            "department": "Manual Review",
            "can_solve": False,
            "solution": "Ahh! We are sorry. Your query has been forwarded to the respective department.",
            "summary": "Multiple issue types detected (billing and network)."
        })

    if llm_result.get("confidence", 0) < 0.6:
        llm_result["department"] = "Manual Review"
        llm_result["can_solve"] = False
        llm_result["solution"] = (
            "Ahh! We are sorry. Your query has been forwarded to the respective department."
        )


    # STEP 6: AUTO-SOLUTION vs ESCALATION


    AUTO_SOLVE_CONFIDENCE = 0.75

    confidence = llm_result.get("confidence", 0)
    can_solve = llm_result.get("can_solve", False)

    if confidence >= AUTO_SOLVE_CONFIDENCE and can_solve:
        resolution_type = "auto_solution"
    else:
        resolution_type = "escalated"
        llm_result["can_solve"] = False
        llm_result["solution"] = (
            "Ahh! We are sorry. Your query has been forwarded to the respective department."
        )
   
# STEP 7: TRANSLATE FINAL USER RESPONSE

 
    final_user_message = llm_result.get("solution", "")

# Translate ONLY if user language is not English
    if language != "en-US" and final_user_message:
        final_user_message = translate_from_english(
        final_user_message,
        target_language=language
    )

    llm_result["solution"] = final_user_message


    # SUCCESS
 
    return {
        "input_type": input_type,
        "selected_language": language,
        "original_text": original_text,
        "english_text": english_text,
        "status": "safe",
        "resolution_type": resolution_type,
        "safety": safety,
        "llm_result": llm_result
    }
