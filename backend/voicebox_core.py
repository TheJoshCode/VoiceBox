import os
import datetime
import threading
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

class VoiceBox:
    def __init__(self, output_dir="output"):
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        self.stop_flag = threading.Event()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def stop(self):
        self.stop_flag.set()

    def _save_audio(self, waveform, path):
        ta.save(path, waveform, self.model.sr)

    def generate_audio(self, text, audio_prompt_path, exaggeration=0.5, cfg_weight=0.5, output_path=None):
        try:
            waveform = self.model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )

            if not output_path:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.output_dir, f"output_{timestamp}.wav")

            self._save_audio(waveform, output_path)
            return output_path

        except Exception as e:
            return f"Error generating audio: {str(e)}"

    def _split_text(self, text, batch_size):
        """Split long text into manageable batches."""
        sentences = text.replace('\n', ' ').split('. ')
        batches, current = [], ""

        for sentence in sentences:
            current += sentence.strip() + ". "
            if len(current.strip().split()) >= batch_size:
                batches.append(current.strip())
                current = ""

        if current.strip():
            batches.append(current.strip())

        return batches

    def process_text_file(self, file_path, audio_prompt_path, batch_size=30, exaggeration=0.5, cfg_weight=0.5):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                return "Error: Input file is empty."

            batches = self._split_text(content, batch_size)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(self.output_dir, f"run_{timestamp}")
            os.makedirs(session_dir, exist_ok=True)

            first_output = None

            for i, batch_text in enumerate(batches):
                if self.stop_flag.is_set():
                    self.stop_flag.clear()
                    return "Generation stopped."

                output_path = os.path.join(session_dir, f"batch_{i + 1}.wav")
                result = self.generate_audio(
                    text=batch_text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    output_path=output_path
                )

                if isinstance(result, str) and os.path.exists(result) and not first_output:
                    first_output = result

            return first_output or "Error: No valid output generated."

        except Exception as e:
            return f"Error processing text file: {str(e)}"
