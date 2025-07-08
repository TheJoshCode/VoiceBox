import os
import re
import torch
import torchaudio as ta
import threading
import subprocess
import platform
from pywebio.input import textarea, input_group, file_upload, actions, slider
from pywebio.output import put_html, put_success, put_file, put_processbar, set_processbar, clear
from pywebio.platform.tornado_http import start_server
from chatterbox.tts import ChatterboxTTS
import webview

def get_next_run_folder(base_dir="chunks"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    run_nums = [
        int(match.group(1)) for name in existing
        if (match := re.match(r"run_(\d{3})", name))
    ]
    next_run_num = max(run_nums, default=0) + 1
    next_folder = os.path.join(base_dir, f"run_{next_run_num:03}")
    os.makedirs(next_folder, exist_ok=True)
    return next_folder

def open_folder(path):
    path = os.path.abspath(path)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])

def voicebox_app():
    put_html("""
    <style>
        html, body {
            margin: 0;
            padding: 0;
            background-color: #f4f6f9;
            overflow: hidden;
        }
        .pywebio-container {
            margin: 0 !important;
            padding: 30px !important;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(0,0,0,0.05);
            font-family: "Segoe UI", sans-serif;
            max-width: 640px;
            margin-left: auto;
            margin-right: auto;
        }
        h2 {
            text-align: center;
            color: #333;
            margin-top: 0;
        }
        input, textarea, button {
            font-size: 16px !important;
        }
    </style>
    <h2>🎙️ VoiceBox TTS</h2>
    """)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)

    form = input_group("Generate Speech", [
        textarea("Text to synthesize", name="text", rows=8, required=True, placeholder="Enter text..."),
        slider("Batch size (characters)", name="batch_size", value=100, min_value=1, max_value=500, step=1),
        file_upload("Speaker WAV (optional)", name="speaker", accept=".wav", required=False),
        actions("", buttons=["Generate"], name="action")
    ])

    text = form["text"]
    batch_size = form["batch_size"]
    speaker_file = form["speaker"]

    out_dir = get_next_run_folder()
    chunks = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]

    kwargs = {}
    if speaker_file:
        speaker_path = os.path.join(out_dir, "speaker.wav")
        with open(speaker_path, "wb") as f:
            f.write(speaker_file["content"])
        kwargs["audio_prompt_path"] = speaker_path

    put_processbar('progress', label="Generating Audio", init=0)

    for idx, chunk in enumerate(chunks, start=1):
        set_processbar('progress', idx / len(chunks))
        wav = model.generate(chunk, **kwargs)
        out_path = os.path.join(out_dir, f"voicebox_out_{idx}.wav")
        ta.save(out_path, wav, model.sr)

    clear()
    put_html("<h2>✅ Generation Complete!</h2>")
    open_folder(out_dir)

def start_voicebox_webview():
    threading.Thread(
        target=lambda: start_server(voicebox_app, port=8080, auto_open_webbrowser=False),
        daemon=True
    ).start()

    window = webview.create_window(
        "VoiceBox TTS",
        "http://localhost:8080",
        width=700,
        height=650,
        frameless=True,
        easy_drag=False,
        resizable=False
    )
    webview.start()

if __name__ == "__main__":
    start_voicebox_webview()
