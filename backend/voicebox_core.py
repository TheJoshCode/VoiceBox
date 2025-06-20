import os
import datetime
import threading
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

class VoiceBox:
    def __init__(self):
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        self.stop_flag = threading.Event()

    def stop(self):
        self.stop_flag.set()

    def generate_audio(self, text, audio_prompt_path, exaggeration=0.5, cfg_weight=0.5, output_path="output/output.wav"):
        try:
            wav = self.model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            ta.save(output_path, wav, self.model.sr)
            return output_path
        except Exception as e:
            return f"Error: {str(e)}"

    def process_text_file(self, file_path, audio_prompt_path, batch_size, exaggeration=0.5, cfg_weight=0.5):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        sentences = text.split('. ')
        batches, current = [], ""
        for s in sentences:
            current += s + ". "
            if len(current) / 5 > batch_size:
                batches.append(current.strip())
                current = ""
        if current:
            batches.append(current.strip())

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        first_output = None
        for i, batch in enumerate(batches):
            if self.stop_flag.is_set():
                self.stop_flag.clear()
                return "Generation stopped."

            output_path = os.path.join(output_dir, f"batch_{i}.wav")
            result = self.generate_audio(batch, audio_prompt_path, exaggeration, cfg_weight, output_path)
            if not first_output and os.path.exists(result):
                first_output = result

        return first_output or "output/none.wav"
