import os
import speech_recognition as sr
import time
import google.generativeai as genai
import torch
import pyttsx3
import threading
from faster_whisper import WhisperModel
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import nest_asyncio

# Patch the event loop
nest_asyncio.apply()

# Environment configuration
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Inject CSS for a fixed bottom container for the buttons
st.markdown(
    """
    <style>
    .fixed-bottom {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        padding: 10px;
        text-align: center;
        z-index: 1000;
        border-top: 1px solid #e6e6e6;
    }
    /* Add bottom padding so that content is not hidden behind the fixed container */
    .main .block-container {
        padding-bottom: 120px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for current conversation, previous chats, and selected chat
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = []
if "previous_chats" not in st.session_state:
    st.session_state.previous_chats = []
if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None

# Sidebar: Display previous chats and provide a "New Chat" button
st.sidebar.header("Chats")
if st.sidebar.button("New Chat"):
    if st.session_state.current_conversation:
        # Archive current conversation as a single string separated by line breaks
        st.session_state.previous_chats.append("\n".join(st.session_state.current_conversation))
    st.session_state.current_conversation = []
    st.session_state.selected_chat = None

# Create a radio widget with a "Current Chat" option at the top
options = ["Current Chat"]
for idx, chat in enumerate(st.session_state.previous_chats):
    # Use the first three words as a label
    label = " ".join(chat.split()[:3])
    options.append(f"{idx}: {label}")
selected_label = st.sidebar.radio("Select Chat", options, key="prev_chat_radio")
if selected_label == "Current Chat":
    st.session_state.selected_chat = None
else:
    selected_index = int(selected_label.split(":")[0])
    st.session_state.selected_chat = selected_index

# Main page components
st.title("AI-Powered Voice Assistant")

# Display the assistant image at the top of the interface
st.image(
    "G:/Downloads/camp_lift_project/DALL·E 2025-03-28 01.12.31 - A futuristic AI-powered voice assistant interface for Home.LLC, featuring a sleek and modern chatbot avatar with a home icon subtly integrated into it.webp",
    use_container_width=True
)

# Display either the selected previous chat or the current conversation
if st.session_state.selected_chat is not None:
    st.write("### Previous Chat")
    st.write(st.session_state.previous_chats[st.session_state.selected_chat])
    if st.button("Back to Current Chat"):
        st.session_state.selected_chat = None
else:
    st.write("### Conversation")
    for msg in st.session_state.current_conversation:
        st.write(msg)

# Whisper ASR setup
whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    cpu_threads=os.cpu_count()
)

# Gemini setup
genai.configure(api_key="AIzaSyA_dHI1lZ-nWoMrEpnQu6FF6qruOQjpopc")
if "convo" not in st.session_state or st.session_state.convo is None:
    st.session_state.convo = genai.GenerativeModel("gemini-2.0-flash").start_chat(history=[])
system_message = '''Instructions: You are an advanced AI voice assistant, you are being used to power a voice assistant and should response as so.Your responses must be highly intelligent, well-structured, and concise. Prioritize clarity, logic, and factual accuracy.

Response Guidelines:
- Adapt your response to the complexity of the question. Use direct, short sentences for simple queries and structured, detailed responses for complex ones.
- Use only words of value; avoid filler language.
- Present reasoning in a clear, step-by-step manner when needed.
- Base your responses strictly on facts and logical inference.
- Maintain a professional and neutral tone throughout.
- Do not include asterisks (*) in your response.
'''
st.session_state.convo.send_message(system_message.replace("\n", " "))

# Function to handle text-to-speech using a background thread
def speak(text):
    """Text-to-Speech function using pyttsx3 with proper handling"""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.error(f"TTS Error: {e}")

# Speech recognition setup
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def process_command(audio_data):
    """Enhanced audio processing"""
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data.get_wav_data())
        segments, _ = whisper_model.transcribe(
            "temp_audio.wav",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        return " ".join([seg.text for seg in segments]).strip()
    except Exception as e:
        st.error(f"Processing error: {e}")
        return ""

# Load knowledge base from file
knowledge_base_file = r"G:\Downloads\camp_lift_project\kinjal_knowledge base.txt"
with open(knowledge_base_file, "r") as f:
    knowledge_base_content = f.read()

# Preprocess knowledge base into a dictionary
knowledge_base_dict = {}
kb_sentences = [line.strip() for line in knowledge_base_content.split("\n") if line.strip()]
for i in range(0, len(kb_sentences), 2):
    question = kb_sentences[i]
    answer = kb_sentences[i + 1] if i + 1 < len(kb_sentences) else ""
    knowledge_base_dict[question] = answer

# Initialize Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
kb_questions = list(knowledge_base_dict.keys())
kb_embeddings = embedding_model.encode(kb_questions, convert_to_tensor=True)

def search_data(query):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, kb_embeddings)[0]
    top_index = similarities.argmax().item()
    if similarities[top_index] > 0.5:
        return knowledge_base_dict[kb_questions[top_index]]
    else:
        return "No relevant information found."

# Fixed bottom container for the voice input buttons
st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

# "Activate Voice Assistant" button
if col1.button("🔊 Voice Assistant: What can i help with?"):
    st.write("✅ Activated! Speak your command:")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, phrase_time_limit=10)
            user_input = process_command(audio)
            if user_input:
                st.write(f"👤 You: {user_input}")
                st.session_state.current_conversation.append(f"User: {user_input}")
                response = st.session_state.convo.send_message(user_input).text
                st.write(f"🤖 Assistant: {response}")
                st.session_state.current_conversation.append(f"Assistant: {response}")
                threading.Thread(target=speak, args=(response,), daemon=True).start()
            else:
                st.write("❌ No input detected. Please try again.")
        except sr.WaitTimeoutError:
            st.write("⏳ Listening timed out. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# "Know about me?" button with voice input
# "Know about the developer?" button with voice input
if col2.button("🔊 Voice Assistant: Know about the developer?"):
    st.write("🎤 Listening for your question...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            user_input = process_command(audio)
            if user_input:
                st.write(f"👤 You: {user_input}")
                st.session_state.current_conversation.append(f"User: {user_input}")
                # Retrieve initial answer from the knowledge base
                kb_answer = search_data(user_input)
                # Construct a refined prompt for the LLM using both the user query and KB info
                refined_prompt = (
                    f"User Query: {user_input}\n"
                    f"Knowledge Base Data: {kb_answer}\n\n"
                    "Please generate a creative, comprehensive, and highly professional response that is well-structured and impressive. "
                    "If the knowledge base does not contain relevant information, simply state: 'Information is not present in the knowledge base.'"
                )

                refined_answer = st.session_state.convo.send_message(refined_prompt).text
                if refined_answer:
                    st.write(f"🤖 Assistant: {refined_answer}")
                    st.session_state.current_conversation.append(f"Assistant: {refined_answer}")
                    threading.Thread(target=speak, args=(refined_answer,), daemon=True).start()
                else:
                    st.write("❌ I couldn't generate a refined answer.")
            else:
                st.write("❌ No input detected. Please try again.")
        except sr.WaitTimeoutError:
            st.write("⏳ Listening timed out. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("</div>", unsafe_allow_html=True)
