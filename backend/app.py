from flask import Flask, request, send_file, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from voicebox_core import VoiceBox
import os

app = Flask(__name__, static_folder="../frontend", static_url_path="")
voicebox = VoiceBox()

os.makedirs("output", exist_ok=True)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/test")
def test():
    return "OK", 200

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
            return jsonify({"error": "Provide a reference audio."}), 400

        audio_filename = secure_filename(audio_file.filename or "audio_prompt.wav")
        audio_path = os.path.join("output", audio_filename)
        audio_file.save(audio_path)

        if txt_file:
            text_filename = secure_filename(txt_file.filename or "input.txt")
            text_path = os.path.join("output", text_filename)
            txt_file.save(text_path)
            result_path = voicebox.process_text_file(text_path, audio_path, batch_size, exaggeration, cfg_weight)
        elif text:
            result_path = voicebox.generate_audio(text, audio_path, exaggeration, cfg_weight)
        else:
            return jsonify({"error": "Provide either text or a text file."}), 400

        if isinstance(result_path, str) and os.path.exists(result_path):
            return send_file(result_path, mimetype="audio/wav", as_attachment=True)
        elif isinstance(result_path, str) and "stopped" in result_path.lower():
            return jsonify({"message": result_path}), 200

        return jsonify({"error": "No audio generated."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stop", methods=["POST"])
def stop():
    voicebox.stop()
    return jsonify({"message": "Stopped"}), 200

@app.route("/output/<path:filename>")
def download_file(filename):
    return send_from_directory("output", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
