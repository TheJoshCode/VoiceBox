import os
import torch
import torchaudio as ta
import threading

from pywebio.input import textarea, input_group, file_upload, actions, slider
from pywebio.output import put_text, put_html, put_success, put_file
from pywebio.platform.tornado_http import start_server
from chatterbox.tts import ChatterboxTTS
import webview

def voicebox_app():
    put_html("""
    <style>
        body {
            font-family: "Segoe UI", sans-serif;
            background-color: #f4f6f9;
            padding: 30px;
        }
        .pywebio-container {
            max-width: 700px;
            margin: auto;
            background: #ffffff;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 0 12px rgba(0,0,0,0.05);
        }
        h2 {
            text-align: center;
            color: #333;
        }
        .button {
            background-color: #1e88e5;
            border: none;
            color: white;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #1565c0;
        }
    </style>
    """)

    put_html("<h2>🎙️ VoiceBox TTS</h2>")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)

    form = input_group("Generate Speech", [
        textarea("Text to synthesize", name="text", rows=8, required=True, placeholder="Enter text..."),
        slider("Batch size (characters)", name="batch_size", value=200, min_value=50, max_value=500, step=10),
        file_upload("Speaker WAV (optional)", name="speaker", accept=".wav", required=False),
        actions("", buttons=["Generate"], name="action")
    ])

    text = form["text"]
    batch_size = form["batch_size"]
    speaker_file = form["speaker"]

    os.makedirs("chunks", exist_ok=True)
    chunks = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]
    out_files = []

    for idx, chunk in enumerate(chunks, start=1):
        kwargs = {}
        if speaker_file:
            speaker_path = f"temp_speaker_{idx}.wav"
            with open(speaker_path, "wb") as f:
                f.write(speaker_file["content"])
            kwargs["audio_prompt_path"] = speaker_path

        wav = model.generate(chunk, **kwargs)
        out_path = f"chunks/voicebox_out_{idx}.wav"
        ta.save(out_path, wav, model.sr)
        out_files.append(out_path)

    put_success(f"✅ Generated {len(out_files)} files in 'chunks/'.")

    for path in out_files:
        with open(path, "rb") as f:
            put_file(os.path.basename(path), f.read())


def start_voicebox_webview():
    # Start PyWebIO server in a separate thread
    threading.Thread(target=lambda: start_server(voicebox_app, port=8080, auto_open_webbrowser=False), daemon=True).start()

    # Start native window
    webview.create_window("VoiceBox TTS", "http://localhost:8080", width=800, height=700)
    webview.start()


if __name__ == "__main__":
    start_voicebox_webview()
