import streamlit as st
import hashlib
import base64
import os
import io
import json
import requests
import wave # Used for TTS audio
from PIL import Image # pyright: ignore[reportMissingImports]asf

# --- CONFIGURATION ---
# API Key (Streamlit secrets mein daalna behtar hai)
API_KEY = "AIzaSyBSUko24RIqj6t71fidq_RQTIpkq5cn6ug"
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"API Key set karne mein error: {e}")

# Models
TEXT_MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
IMAGE_GEN_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={API_KEY}"
TTS_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={API_KEY}"

# Owner auth details (JS logic se milte julate)
OWNER_SALT_B64 = 'DJKZO8BoVpZk9/45Bc4JuQ=='
OWNER_HASH_STR = '1Tuc7+3xhADrgv4rLvot5zIOh+ajz8ekuyG2nEYwL7o='

# --- AUTHENTICATION ---

def hash_password(p, s_b64):
    """Python equivalent of the JS crypto.subtle.digest logic"""
    salt = base64.b64decode(s_b64)
    pw_bytes = p.encode('utf-8')
    combo = salt + pw_bytes
    hash_bytes = hashlib.sha256(combo).digest()
    return base64.b64encode(hash_bytes).decode('utf-8')

def check_login(username, password):
    """Locked to owner credentials"""
    if username.lower() == 'owner':
        try:
            hashed = hash_password(password, OWNER_SALT_B64)
            if hashed == OWNER_HASH_STR:
                return True
        except Exception as e:
            st.error(f"Hashing error: {e}")
            return False
    return False

# --- API HELPERS ---

def fetch_gemini(prompt_text, file_data=None, file_mime_type=None, system_instruction="You are a helpful AI assistant."):
    """Gemini text, vision, aur file analysis ko handle karta hai"""
    model = genai.GenerativeModel(TEXT_MODEL_NAME, system_instruction=system_instruction)
    
    parts = []
    if file_data:
        if "image" in file_mime_type:
            parts.append(Image.open(io.BytesIO(file_data)))
        elif "text" in file_mime_type or "json" in file_mime_type or "javascript" in file_mime_type or "python" in file_mime_type:
            text_content = file_data.decode('utf-8')
            parts.append(f"[START OF ATTACHED FILE]\n\n{text_content}\n\n[END OF FILE]\n\n")
        else:
            st.warning(f"Unsupported file type for analysis: {file_mime_type}")
            
    parts.append(prompt_text)
    
    # URL Summarization check (JS logic se)
    if is_url(prompt_text) and not file_data:
        parts = [f"Please summarize this webpage: {prompt_text}"]
        st.info("URL ko summarize karne ki koshish...", icon="🌐")

    response = model.generate_content(parts, generation_config={"tools": [{"google_search": {}}]})
    
    # Sources (Grounding)
    sources = []
    try:
        if response.candidates[0].grounding_metadata:
            sources = [
                {"uri": attr.web.uri, "title": attr.web.title}
                for attr in response.candidates[0].grounding_metadata.grounding_attributions
                if attr.web
            ]
    except Exception:
        pass # No grounding metadata
        
    return response.text, sources

def fetch_gemini_json(prompt_text, schema):
    """Trip Planner jaise JSON output ke liye"""
    model = genai.GenerativeModel(TEXT_MODEL_NAME)
    response = model.generate_content(
        prompt_text,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema
        }
    )
    return json.loads(response.text)

def fetch_image_gen(prompt):
    """Imagen se image generate karta hai"""
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1}
    })
    response = requests.post(IMAGE_GEN_ENDPOINT, headers=headers, data=payload)
    if response.status_code != 200:
        raise Exception(f"Image Gen Error: {response.text}")
    data = response.json()
    if not data.get("predictions") or not data["predictions"][0].get("bytesBase64Encoded"):
        raise Exception("Image Gen Error: No image data received.")
    
    img_bytes = base64.b64decode(data["predictions"][0]["bytesBase64Encoded"])
    return Image.open(io.BytesIO(img_bytes))

def fetch_tts(text_to_speak):
    """Gemini se audio generate karta hai (PCM)"""
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({
        "contents": [{"parts": [{"text": f"Say: {text_to_speak}"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
            }
        },
        "model": "gemini-2.5-flash-preview-tts"
    })
    response = requests.post(TTS_ENDPOINT, headers=headers, data=payload)
    if response.status_code != 200:
        raise Exception(f"TTS Error: {response.text}")
    data = response.json()
    part = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0]
    
    if not part.get("inlineData"):
        raise Exception("TTS Error: No audio data received.")
        
    audio_data_b64 = part["inlineData"]["data"]
    mime_type = part["inlineData"]["mimeType"]
    sample_rate = int(mime_type.split("rate=")[-1])
    
    # PCM data (s16le)
    pcm_data = base64.b64decode(audio_data_b64)
    return pcm_data, sample_rate

def pcm_to_wav_bytes(pcm_data, sample_rate):
    """PCM ko in-memory WAV file mein convert karta hai"""
    wav_file = io.BytesIO()
    with wave.open(wav_file, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    wav_file.seek(0)
    return wav_file.read()

def is_url(text):
    """Simple URL check"""
    return text.startswith("http://") or text.startswith("https://")

# --- UI (STREAMLIT) ---

st.set_page_config(page_title="Mera LLM Cockpit", layout="wide", page_icon="🤖")

# Session State Initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_trip_planner" not in st.session_state:
    st.session_state.show_trip_planner = False
if "show_image_gen" not in st.session_state:
    st.session_state.show_image_gen = False

# --- 1. LOGIN SCREEN ---
if not st.session_state.logged_in:
    st.title("🔐 Owner Login")
    with st.form("login_form"):
        username = st.text_input("Username (Locked)", value="owner", disabled=True)
        password = st.text_input("Password (Locked)", value="mi@12", type="password", disabled=True)
        submitted = st.form_submit_button("Unlock System", use_container_width=True)
        
        if submitted:
            with st.spinner("Authenticating..."):
                if check_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.chat_history = [{"role": "bot", "text": "Hello there! How can I help you today?", "sources": []}]
                    st.rerun()
                else:
                    st.error("Access Denied. Integrity check failed.")

# --- 2. MAIN APP ---
else:
    # --- Sidebar (Features) ---
    with st.sidebar:
        st.title("Mera LLM Interface")
        st.markdown(f"Logged in as **owner**")
        if st.button("Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.chat_history = []
            st.rerun()
            
        st.divider()
        st.subheader("✨ Features")
        
        if st.button("Summarize Chat", use_container_width=True):
            if len(st.session_state.chat_history) < 2:
                st.warning("Summarize karne ke liye history nahi hai.")
            else:
                with st.spinner("Summarizing..."):
                    hist = "\n".join([f"{m['role']}: {m['text']}" for m in st.session_state.chat_history])
                    summary, _ = fetch_gemini(f"Summarize this chat history in one concise paragraph:\n{hist}", system_instruction="You are a helpful summarizer.")
                    st.session_state.chat_history.append({"role": "system", "text": f"**Conversation Summary:**\n{summary}", "sources": []})

        if st.button("Plan a Trip", use_container_width=True):
            st.session_state.show_trip_planner = True
        
        if st.button("Generate Image", use_container_width=True):
            st.session_state.show_image_gen = True

    # --- Feature Modals (Streamlit-style) ---
    
    # Trip Planner Modal
    if st.session_state.show_trip_planner:
        with st.form("trip_form"):
            st.subheader("🌍 AI Trip Planner")
            destination = st.text_input("Destination (e.g., Tokyo)")
            days = st.number_input("Duration (days)", min_value=1, max_value=14, value=3)
            submit_trip = st.form_submit_button("Generate Plan")
            
            if submit_trip:
                if not destination:
                    st.error("Please enter a destination.")
                else:
                    with st.spinner("Generating amazing plan..."):
                        try:
                            schema = {
                                "type": "OBJECT",
                                "properties": {
                                    "planTitle": {"type": "STRING"},
                                    "summary": {"type": "STRING"},
                                    "dailyPlan": {
                                        "type": "ARRAY",
                                        "items": {
                                            "type": "OBJECT",
                                            "properties": {
                                                "day": {"type": "NUMBER"},
                                                "theme": {"type": "STRING"},
                                                "activities": {
                                                    "type": "ARRAY",
                                                    "items": {
                                                        "type": "OBJECT",
                                                        "properties": {
                                                            "time": {"type": "STRING"},
                                                            "description": {"type": "STRING"}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "required": ["planTitle", "summary", "dailyPlan"]
                            }
                            plan = fetch_gemini_json(f"Plan a {days}-day trip to {destination}", schema)
                            
                            st.subheader(plan['planTitle'])
                            st.markdown(f"*{plan['summary']}*")
                            for day in plan['dailyPlan']:
                                with st.expander(f"Day {day['day']}: {day['theme']}"):
                                    for activity in day['activities']:
                                        st.markdown(f"**{activity['time']}:** {activity['description']}")
                        except Exception as e:
                            st.error(f"Trip plan failed: {e}")
            
            if st.button("Close Planner"):
                st.session_state.show_trip_planner = False
                st.rerun()

    # Image Gen Modal
    if st.session_state.show_image_gen:
        with st.form("image_gen_form"):
            st.subheader("🎨 AI Image Generator")
            prompt = st.text_area("Image Prompt")
            submit_image = st.form_submit_button("Generate Image")
            
            if submit_image:
                if not prompt:
                    st.error("Please enter a prompt.")
                else:
                    with st.spinner("Creating masterpiece..."):
                        try:
                            image = fetch_image_gen(prompt)
                            st.image(image, caption=prompt, use_column_width=True)
                        except Exception as e:
                            st.error(f"Image generation failed: {e}")
            
            if st.button("Close Image Gen"):
                st.session_state.show_image_gen = False
                st.rerun()

    # --- Main Chat Interface ---
    st.title("🤖 Chat")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "system":
            st.info(msg["text"], icon="✨")
        else:
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])
                if msg["role"] == "bot":
                    if st.button("🔊", key=f"tts_{msg['text'][:20]}"):
                        with st.spinner("Generating audio..."):
                            try:
                                pcm_data, rate = fetch_tts(msg["text"])
                                wav_data = pcm_to_wav_bytes(pcm_data, rate)
                                st.audio(wav_data, format="audio/wav")
                            except Exception as e:
                                st.error(f"TTS failed: {e}")
                
                # Display Sources
                if msg.get("sources"):
                    with st.expander("Sources", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(f"- [{src['title']}]({src['uri']})")

    # Chat Input (bottom)
    col1, col2 = st.columns([10, 1])
    with col1:
        uploaded_file = st.file_uploader("Attach file (Image, Txt, Py, JS, CSV...)", type=["jpg", "jpeg", "png", "txt", "py", "js", "csv", "md", "json"])
    with col2:
        # Mic & Paste features (browser-specific) yahan nahi hain.
        st.write("") # Placeholder
        
    if prompt := st.chat_input("Type your message or attach a file..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "text": prompt, "sources": []})
        with st.chat_message("user"):
            if uploaded_file:
                st.markdown(f"{prompt} (File: {uploaded_file.name})")
            else:
                st.markdown(prompt)

        # Get bot response
        with st.chat_message("bot"):
            with st.spinner("Thinking..."):
                try:
                    file_data = None
                    file_mime = None
                    if uploaded_file:
                        file_data = uploaded_file.getvalue()
                        file_mime = uploaded_file.type
                        
                    response_text, sources = fetch_gemini(prompt, file_data, file_mime)
                    st.markdown(response_text)
                    
                    # Add TTS button (Streamlit UI mein yeh dynamic add karna mushkil hai, isliye upar history loop mein hai)
                    
                    # Display Sources (agar hain)
                    if sources:
                        with st.expander("Sources", expanded=False):
                            for src in sources:
                                st.markdown(f"- [{src['title']}]({src['uri']})")
                                
                    st.session_state.chat_history.append({"role": "bot", "text": response_text, "sources": sources})
                    
                    # (Smart replies yahan add kiye ja sakte hain, lekin UI complex ho jayega)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.chat_history.append({"role": "system", "text": f"Error: {e}", "sources": []})

