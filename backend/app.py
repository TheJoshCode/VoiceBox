from flask import Flask, request, send_file, jsonify, send_from_directory
from voicebox_core import VoiceBox
import os
import threading

app = Flask(__name__, static_folder="../frontend", static_url_path="")
voicebox = VoiceBox()

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        text = request.form.get("text")
        txt_file = request.files.get("file")
        audio_file = request.files.get("audio")
        batch_size = int(request.form.get("batch_size", 30))
        exaggeration = float(request.form.get("exaggeration", 0.5))
        cfg_weight = float(request.form.get("cfg_weight", 0.5))

        if not audio_file:
            return "Error: Provide a reference audio.", 400

        audio_path = os.path.join("output", "audio_prompt.wav")
        audio_file.save(audio_path)

        if txt_file:
            text_path = os.path.join("output", "input.txt")
            txt_file.save(text_path)
            result_path = voicebox.process_text_file(text_path, audio_path, batch_size, exaggeration, cfg_weight)
        elif text:
            result_path = voicebox.generate_audio(text, audio_path, exaggeration, cfg_weight)
        else:
            return "Error: Provide either text or a text file.", 400

        if os.path.exists(result_path):
            return send_file(result_path, mimetype="audio/wav")
        return "Error: No audio generated.", 500

    except Exception as e:
        return str(e), 500

@app.route("/stop", methods=["POST"])
def stop():
    voicebox.stop()
    return "Stopped", 200

if __name__ == "__main__":
    import webview

    os.makedirs("output", exist_ok=True)

    def run_flask():
        app.run(host="127.0.0.1", port=5000, threaded=True)

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    webview.create_window("VoiceBox", "http://127.0.0.1:5000")
