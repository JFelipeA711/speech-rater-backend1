import torch
import librosa
import language_tool_python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
import re
import os
import openai
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configurar API de OpenAI
openai.api_key = "TU_API_KEY"

# Cargar modelos de IA
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
tool = language_tool_python.LanguageTool("en-US")

# Función para transcribir el audio
def transcribe_audio(audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()

# Evaluar gramática
def evaluate_grammar(text):
    matches = tool.check(text)
    return len(matches), [match.ruleIssueType for match in matches]

# Evaluar pronunciación
def evaluate_pronunciation(transcription, reference_text):
    reference_words = reference_text.lower().split()
    transcript_words = transcription.split()
    matched = sum(1 for w in transcript_words if w in reference_words)
    accuracy = matched / max(len(transcript_words), 1)
    score = accuracy * 4
    return max(0, min(4, score))

# Evaluar fluidez con pausas y muletillas
def evaluate_fluency(audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)
    duration = len(speech) / rate
    word_count = len(transcribe_audio(audio_path).split())
    words_per_minute = (word_count / duration) * 60
    pauses = len(re.findall(r'\b(uh|um|like)\b', transcribe_audio(audio_path)))
    fluency_score = min(4, max(0, words_per_minute / 30 - pauses * 0.5))
    return fluency_score

# Generar feedback con ChatGPT
def generate_chatgpt_feedback(transcription, grammar_errors, pronunciation_score, fluency_score):
    prompt = f"""
    You are an AI language tutor evaluating an English speech recording. The transcript is:
    "{transcription}"
    
    Evaluation criteria:
    - Grammar errors: {grammar_errors}
    - Pronunciation score: {pronunciation_score}/4
    - Fluency score: {fluency_score}/4
    
    Provide a detailed analysis with strengths and areas for improvement.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]

# Generar feedback detallado
def generate_feedback(audio_path):
    transcription = transcribe_audio(audio_path)
    grammar_errors, grammar_issues = evaluate_grammar(transcription)
    pronunciation_score = evaluate_pronunciation(transcription, transcription)
    fluency_score = evaluate_fluency(audio_path)

    chatgpt_feedback = generate_chatgpt_feedback(transcription, grammar_errors, pronunciation_score, fluency_score)

    return {
        "transcription": transcription,
        "grammar_errors": grammar_errors,
        "pronunciation_score": pronunciation_score,
        "fluency_score": fluency_score,
        "chatgpt_feedback": chatgpt_feedback
    }

# Ruta para subir archivos y procesarlos
@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    feedback = generate_feedback(filepath)
    return jsonify(feedback)

import os 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=5000)
