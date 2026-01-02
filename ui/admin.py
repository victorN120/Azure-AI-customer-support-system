import io
import requests
import streamlit as st


# CONFIG

API_URL = "http://127.0.0.1:8000/process-query"

LANGUAGE_MAP = {
    "English": "en-US",
    "Telugu": "te-IN",
    "Hindi": "hi-IN"
}

st.set_page_config(
    page_title="AI Customer Support – Admin Panel",
    layout="centered"
)


# UI HEADER


st.title("🛠️ AI Customer Support – Admin Panel")
st.caption("Describe the issue or upload an audio file. The system decides routing.")
# LANGUAGE SELECTION


language_label = st.selectbox(
    "🌐 Select Language",
    list(LANGUAGE_MAP.keys())
)

language_code = LANGUAGE_MAP[language_label]

# TEXT INPUT


st.markdown("### ✍️ Text Input")
text_input = st.text_area(
    "Describe the customer issue",
    placeholder="Type the problem in your own words…",
    height=120
)

# AUDIO UPLOAD


st.markdown("### 📂 Upload Audio File (WAV)")
uploaded_audio = st.file_uploader(
    "Upload WAV audio file",
    type=["wav"]
)

# SUBMIT BUTTON

submit = st.button("🚀 Process Query")

if submit:
    data = {
        "language": language_code
    }
    files = {}

    # Enforce isolation: ONLY ONE INPUT
    has_text = text_input.strip() != ""
    has_audio = uploaded_audio is not None

    if has_text and has_audio:
        st.error("Please provide either text OR audio, not both.")
        st.stop()

    if not has_text and not has_audio:
        st.error("Please enter text or upload a WAV audio file.")
        st.stop()

    if has_text:
        data["text"] = text_input.strip()

    if has_audio:
        files["audio"] = uploaded_audio

    with st.spinner("Processing…"):
        response = requests.post(
            API_URL,
            data=data,
            files=files,
            timeout=60
        )

    if response.status_code != 200:
        st.error(f"Backend error: {response.text}")
        st.stop()

    result = response.json()

    # RESULTS

    st.subheader("🛡️ Safety & Tone")

    safety = result.get("safety", {})

    label = safety.get("label", "unknown")

    if label == "safe":
      st.success("✅ Safe content")
    elif label == "angry_or_abusive":
      st.warning("⚠️ Angry / Abusive tone detected")
    elif label == "unsafe":
      st.error("🚨 Unsafe content – manual review required")
    else:
      st.info("ℹ️ Safety status unknown")


    st.success("✅ Processed successfully")

    st.subheader("📄 Original Input")
    st.write(result.get("original_text", ""))

    st.subheader("🌍 English Text")
    st.write(result.get("english_text", ""))

    st.subheader("🧠 AI Decision")
    st.json(result.get("llm_result", {}))
    st.subheader("🧩 Resolution")

    if result.get("resolution_type") == "auto_solution":
     st.success("✅ Issue resolved automatically")
     st.write(result["llm_result"].get("solution", ""))
    else:
     st.warning("📨 Escalated to human support")
     st.write(result["llm_result"].get("solution", ""))


    if result.get("status") == "manual_review":
        st.warning("⚠️ Routed to Manual Review")
