🛠️ Responsible-AI Customer Support Assistant

Azure AI · LangChain · FastAPI · Streamlit

📌 Overview

This project is an end-to-end AI-powered customer support intake system that accepts text or voice queries, analyzes them responsibly, and either automatically provides solutions or routes them to the correct department with human escalation when required.

The system is designed following Responsible AI principles, focusing on safety, confidence-based decisions, and human-in-the-loop escalation.

🎯 Key Features

🎙️ Voice & Text Input (Multilingual)

🌍 Automatic Language Translation

🧠 Intent Detection using Azure OpenAI (GPT-4o) + LangChain

⚠️ Content Safety & Harassment Detection

📊 Confidence Scoring & Decision Logic

🤖 Auto-Resolution for Simple Issues

🧑‍💼 Manual Review for Complex / Unsafe Queries

🖥️ Admin Panel UI (Streamlit)

🏗️ Architecture (High Level)

User submits text / audio

Azure Speech → Speech-to-Text (if voice)

Azure Translator → Convert to English

Azure Content Safety → Detect unsafe / abusive input

LangChain + GPT-4o → Intent, confidence, solution

Deterministic rules → Auto-solve or escalate

Response returned in user’s preferred language

🔐 Responsible AI Safeguards

Profanity / abuse detection using Azure Content Safety

Confidence thresholds to avoid hallucinations

Automatic manual review routing for:

Unsafe content

Low confidence predictions

Ambiguous or multi-issue queries

No hard-coded decisions without AI + rule validation

🧠 Intent Categories

network_issue

billing_issue

service_complaint

fraud_report

general_query

🛠️ Tech Stack

Backend

FastAPI

Azure Speech Services

Azure Translator

Azure Content Safety

Azure OpenAI (GPT-4o)

LangChain

Frontend

Streamlit (Admin Panel)



Environment-based configuration

▶️ How to Run Locally
1️⃣ Clone the repo
git clone https://github.com/your-username/ai-customer-support
cd ai-customer-support

2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Configure environment variables

Create a .env file:

SPEECH_KEY=your_key
SPEECH_REGION=your_region

TRANSLATOR_KEY=your_key
TRANSLATOR_ENDPOINT=your_endpoint
TRANSLATOR_REGION=your_region

CONTENT_SAFETY_KEY=your_key
CONTENT_SAFETY_ENDPOINT=your_endpoint

AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_KEY=your_key
AZURE_OPENAI_DEPLOYMENT=your_deployment

5️⃣ Start backend
uvicorn app.main:app --reload

6️⃣ Start admin panel
streamlit run admin.py

📸 Admin Panel Capabilities

Language dropdown (no manual language codes)

Text input OR voice input (mutually exclusive)

Displays:

Original text

Translated text

Safety label (safe / abusive / unsafe)

AI decision & confidence

Auto-solution or escalation status

🧪 Example Use Cases

Customer reports slow internet → auto troubleshooting steps

Ambiguous billing + network issue → escalated to human support

Abusive language → flagged and routed for manual review

Simple help request → AI-generated guidance

🚀 Why This Project Matters

This project demonstrates:

Real-world AI system design

Responsible deployment of LLMs in production

Human-AI collaboration patterns

Practical use of Azure AI services

Clear decision boundaries to avoid over-automation

👨‍💻 Author

Microsoft Certified: Azure AI Engineer Associate (AI-102)
Designed and implemented as a hands-on production-style AI system.

📎 License

This project is for educational and demonstration purposes.

