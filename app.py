import requests
import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from pydub import AudioSegment
import io
import numpy as np
import wave
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import folium

# Set up Streamlit page configuration
st.set_page_config(layout="wide")

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"

# API details
MISTRAL_API_KEY = "EqWANuhXLDyrVh9TWOw5H5K3ppvp3kWS"  # Replace with your API key
MISTRAL_API_URL = "https://codestral.mistral.ai/v1/fim/completions"

# Models Initialization
@st.cache_resource
def load_models():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    tts_model = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    return processor, model, tts_model

processor, model, tts_model = load_models()

# Transcribe audio to text
def transcribe_audio(audio_bytes):
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1)
        raw_audio = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        inputs = processor(raw_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(inputs)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return "Unable to transcribe audio."

# Generate a travel-related response
def generate_response(transcription):
    if not MISTRAL_API_KEY:
        return "MISTRAL_API_KEY is not set."

    try:
        payload = {
            "model": "codestral-latest",
            "prompt": f"You are a travel assistant. Provide detailed travel guidance based on this query:\n'{transcription}'",
            "max_tokens": 1500,
            "temperature": 0.7,
        }
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            else:
                return "Invalid API response format."
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        st.error(f"Error communicating with the API: {e}")
        return "Unable to generate a response."

# Convert text to speech
def text_to_speech(response):
    try:
        speech = tts_model(response)
        audio_array = speech["audio"]
        sample_rate = speech["sampling_rate"]
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_array.tobytes())
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# Geocode a location to get latitude and longitude
def get_location_coordinates(location_name):
    geolocator = Nominatim(user_agent="travel_assistant")
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# Display a map
def display_map(lat, lon):
    if lat and lon:
        map_obj = folium.Map(location=[lat, lon], zoom_start=12)
        folium.Marker([lat, lon], popup="Location").add_to(map_obj)
        st_folium(map_obj, width=700, height=500)
    else:
        st.error("Unable to find the location on the map.")

# Streamlit Interface
st.title("Travel Assistant")

# Initialize session state for audio input and response
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'response' not in st.session_state:
    st.session_state.response = ""

# Text input for user query
st.write("Ask me anything about your travel plans:")
user_input = st.text_area("", height=200, value=st.session_state.transcription)

# Mic symbol for audio input
audio_file = st.file_uploader("Or record/upload an audio file", type=["wav", "mp3"])

# Send button inside the text area
if st.button("Send"):
    if audio_file:
        st.session_state.audio_bytes = audio_file.read()
        st.session_state.transcription = transcribe_audio(st.session_state.audio_bytes)
        st.session_state.response = generate_response(st.session_state.transcription)
    else:
        st.session_state.response = generate_response(user_input)

# Display the response
if st.session_state.response:
    st.markdown(f"**Response:** {st.session_state.response}")

    # Convert the response to speech and play it
    st.write("Converting to Speech...")
    audio_response = text_to_speech(st.session_state.response)
    if audio_response:
        st.audio(audio_response, format="audio/wav")

    # Geocode the destination (if any) mentioned in the response
    location_name = st.session_state.transcription if st.session_state.transcription else user_input
    lat, lon = get_location_coordinates(location_name)
    
    # Display map
    display_map(lat, lon)