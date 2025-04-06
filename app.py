# app.py
import os
import io
import tempfile
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech, speech
from pydub import AudioSegment
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# --- Google Credentials ---
try:
    # ENSURE THIS PATH IS CORRECT FOR YOUR SYSTEM
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\singh\OneDrive\Desktop\Cloud\cloud-translation-service-1ec3fca0d41b.json"
    logging.info("Google Cloud credentials set.")
except Exception as e:
    logging.error(f"Error setting Google Cloud credentials: {e}")

# --- Flask App Setup ---
# Ensure paths match your React build location
react_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cloud-app', 'build'))
app = Flask(__name__, static_folder=os.path.join(react_build_dir, 'static'), template_folder=react_build_dir)
CORS(app) # Enable CORS for all origins

# --- Google Cloud Clients ---
try:
    translate_client = translate.Client()
    tts_client = texttospeech.TextToSpeechClient()
    speech_client = speech.SpeechClient()
    logging.info("Google Cloud clients initialized.")
except Exception as e:
    logging.error(f"Error initializing Google Cloud clients: {e}")
    translate_client = tts_client = speech_client = None

# --- Flask Static Folder (for TTS output) ---
FLASK_STATIC_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
os.makedirs(FLASK_STATIC_FOLDER, exist_ok=True)

# --- Load BLIP Model ---
MODEL_NAME = "Salesforce/blip-image-captioning-large"
logging.info(f"Attempting to load image captioning model: {MODEL_NAME}")
captioning_model_loaded = False
processor = None
model = None
try:
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps" # Apple Silicon check
    else: device = "cpu"
    logging.info(f"Using device for BLIP model: {device}")
    processor = BlipProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    logging.info(f"BLIP image captioning model '{MODEL_NAME}' loaded successfully.")
    captioning_model_loaded = True
except Exception as e:
    logging.error(f"Error loading BLIP model '{MODEL_NAME}': {e}")
    logging.error(traceback.format_exc())

# --- Configure ffmpeg ---
ffmpeg_configured = False
try:
    ffmpeg_path = r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin\ffmpeg.exe"
    ffprobe_path = r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin\ffprobe.exe"
    if not os.path.exists(ffmpeg_path): raise FileNotFoundError(f"ffmpeg not found at {ffmpeg_path}")
    if not os.path.exists(ffprobe_path): raise FileNotFoundError(f"ffprobe not found at {ffprobe_path}")
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path
    logging.info("ffmpeg paths configured for pydub.")
    ffmpeg_configured = True
except Exception as e:
     logging.error(f"Error configuring ffmpeg paths: {e}. Check paths and ensure ffmpeg is installed correctly.")

# --- Helper Functions ---
def generate_caption(image_pil):
    if not captioning_model_loaded or model is None or processor is None:
         raise RuntimeError("Image captioning model is not available.")
    try:
        image_rgb = image_pil.convert("RGB")
        current_device = model.device
        inputs = processor(image_rgb, return_tensors="pt").to(current_device)
        output = model.generate(**inputs, max_length=75, num_beams=5, early_stopping=True)
        caption = processor.decode(output[0], skip_special_tokens=True)
        logging.info(f"Generated caption: {caption}")
        return caption
    except Exception as e:
        logging.error(f"Error during caption generation: {e}")
        logging.error(traceback.format_exc())
        raise e

# --- Flask Routes ---

@app.route("/image-captioning", methods=["POST"])
def image_captioning():
    logging.info("Received request for /image-captioning")
    if not captioning_model_loaded: return jsonify({"error": "Image captioning model not available"}), 500
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        logging.info(f"Processing image for captioning: {file.filename}")
        caption = generate_caption(image)
        return jsonify({"caption": caption or "No caption generated"})
    except Exception as e:
        return jsonify({"error": f"Error during image captioning: {str(e)}"}), 500

@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():
    logging.info("Received request for /speech-to-text")
    if not ffmpeg_configured:
        logging.warning("STT request received, but ffmpeg might not be configured correctly.")

    # --- Get the language code from the form data (sent with the file) ---
    # Default to 'en-US' if not provided by the frontend
    audio_language_code = request.form.get('language_code', 'en-US')
    logging.info(f"Attempting transcription with language code: {audio_language_code}")
    # ----------------------------------------------------------------------

    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    temp_audio_path = None
    try:
        logging.info(f"Processing audio file for STT: {file.filename}")
        audio_bytes = file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = temp_audio.name
            logging.info("Loading audio with pydub...")
            file_ext = os.path.splitext(file.filename)[1].lower().replace('.', '')
            try: audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file_ext if file_ext else None)
            except Exception: audio = AudioSegment.from_file(io.BytesIO(audio_bytes)) # Fallback
            logging.info("Audio loaded. Converting to WAV (16kHz, mono, 16-bit)...")
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            audio.export(temp_audio_path, format="wav")
            logging.info(f"Audio converted and saved to temporary file: {temp_audio_path}")

        with open(temp_audio_path, "rb") as audio_file: audio_content = audio_file.read()
        logging.info("Temporary WAV file read.")
        recognition_audio = speech.RecognitionAudio(content=audio_content)

        # --- UPDATED: Use the language code from the request ---
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=audio_language_code # Use the language selected by the user
            # Consider adding punctuation: enable_automatic_punctuation=True,
        )
        # ------------------------------------------------------

        logging.info(f"Sending audio to Google Speech-to-Text API for language {audio_language_code}...")
        if speech_client is None: raise RuntimeError("Speech client not initialized")
        response = speech_client.recognize(config=config, audio=recognition_audio)
        logging.info("Received response from Google Speech-to-Text API.")

        transcript = " ".join([result.alternatives[0].transcript for result in response.results]) if response.results else ""
        logging.info(f"Transcription result: '{transcript}'")
        return jsonify({"transcript": transcript or "No speech detected"})

    except FileNotFoundError as e:
        logging.error(f"Configuration error (likely ffmpeg/ffprobe path issue): {e}")
        return jsonify({"error": f"Server configuration error related to audio processing: {e}"}), 500
    except Exception as e:
        logging.error(f"Error during Speech-to-Text processing (lang: {audio_language_code}): {e}")
        logging.error(traceback.format_exc())
        error_msg = str(e)
        if "Invalid language code" in error_msg or "LANGUAGE_CODE_INVALID" in error_msg:
            error_msg = f"STT Error: The language code '{audio_language_code}' is invalid or not supported."
        return jsonify({"error": f"Error during Speech-to-Text: {error_msg}"}), 500
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path); logging.info(f"Cleaned up temporary file: {temp_audio_path}")
            except Exception as cleanup_error: logging.error(f"Error cleaning up temp file {temp_audio_path}: {cleanup_error}")


@app.route("/detect-language", methods=["POST"])
def detect_language():
    logging.info("Received request for /detect-language")
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 415
    data = request.get_json()
    text = data.get("text")
    if not text: return jsonify({"error": "No text provided"}), 400
    try:
        if translate_client is None: raise RuntimeError("Translate client not initialized")
        result = translate_client.detect_language(text)
        detected_info = {"detected_language": result.get("language"), "confidence": result.get("confidence", "N/A")}
        logging.info(f"Language detection result: {detected_info}")
        return jsonify(detected_info)
    except Exception as e:
        logging.error(f"Error during language detection: {e}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/translate", methods=["POST"])
def translate_text():
    logging.info("Received request for /translate")
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 415
    data = request.get_json()
    text = data.get("text")
    target_language = data.get("target_language")
    if not text or not target_language:
        return jsonify({"error": "Missing text or target language"}), 400
    try:
        logging.info(f"Translating text to target language: {target_language}")
        if translate_client is None: raise RuntimeError("Translate client not initialized")
        result = translate_client.translate(text, target_language=target_language)
        translated_info = {"translated_text": result["translatedText"]}
        logging.info(f"Translation result: {translated_info}")
        return jsonify(translated_info)
    except Exception as e:
        logging.error(f"Error during translation to {target_language}: {e}")
        logging.error(traceback.format_exc())
        error_msg = str(e)
        if "invalid" in error_msg.lower() and "target language" in error_msg.lower():
             error_msg = f"Invalid or unsupported target language code: '{target_language}'"
        return jsonify({"error": f"Translation error: {error_msg}"}), 500


@app.route("/text-to-speech", methods=["POST"])
def text_to_speech():
    logging.info("Received request for /text-to-speech")
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 415
    data = request.get_json()
    text = data.get("text")
    language_code = data.get("language_code", "en-US") # Get language for TTS voice

    if not text: return jsonify({"error": "No text provided"}), 400
    if not isinstance(language_code, str) or len(language_code) < 2:
         logging.warning(f"Invalid TTS language code received: '{language_code}', defaulting to en-US")
         language_code = "en-US"

    logging.info(f"Requesting TTS for text: '{text[:50]}...' in language: {language_code}")
    try:
        if tts_client is None: raise RuntimeError("Text-to-Speech client not initialized")
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code=language_code) # Use specified language
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        logging.info("Sending text to Google Text-to-Speech API...")
        response = tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
        logging.info("Received audio response from TTS API.")
        output_filename = "output.mp3"
        file_path = os.path.join(FLASK_STATIC_FOLDER, output_filename)
        with open(file_path, "wb") as out: out.write(response.audio_content)
        logging.info(f"TTS audio saved to {file_path}")
        # Use the correct prefix for Flask's static files
        return jsonify({"audio_url": f"/flask-static/{output_filename}"})
    except Exception as e:
        logging.error(f"Error during Text-to-Speech for lang '{language_code}': {e}")
        logging.error(traceback.format_exc())
        error_msg = str(e)
        if "Unsupported language_code" in error_msg or "could not synthesize" in error_msg.lower():
             error_msg = f"TTS Error: Language '{language_code}' may not be supported or no suitable voice found."
        elif "INVALID_ARGUMENT" in error_msg:
             error_msg = f"TTS Error: Invalid request for language '{language_code}'. Check text or parameters."
        return jsonify({"error": f"Text-to-Speech error: {error_msg}"}), 500


# --- Serve React App and Flask Static Files ---
@app.route("/flask-static/<path:filename>")
def serve_flask_static(filename):
    return send_from_directory(FLASK_STATIC_FOLDER, filename)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(react_build_dir, path)):
        return send_from_directory(react_build_dir, path)
    else:
        index_path = os.path.join(react_build_dir, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(react_build_dir, 'index.html')
        else:
             logging.error(f"React index.html not found at expected location: {index_path}")
             return jsonify({"error": "Frontend application index not found."}), 404

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting Flask application...")
    if translate_client is None or tts_client is None or speech_client is None:
         logging.critical("One or more Google Cloud clients failed to initialize. API calls will fail.")
    if not captioning_model_loaded:
        logging.warning(f"BLIP model '{MODEL_NAME}' failed to load. Captioning endpoint will fail.")
    if not ffmpeg_configured:
         logging.warning("ffmpeg may not be configured correctly. STT endpoint might fail if audio conversion is needed.")

    app.run(debug=True, host='127.0.0.1', port=5000)