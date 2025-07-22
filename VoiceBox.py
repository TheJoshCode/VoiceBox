import gradio as gr
import torchaudio as ta
from pathlib import Path
import os
import subprocess
from datetime import datetime
import re
import logging

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from chatterbox.tts import ChatterboxTTS

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)

def load_phonics(file_path="phonic_lines.txt"):
    path = Path(file_path)
    if not path.exists():
        logging.error(f"Phonics file not found: {file_path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [re.sub(r"[‘’]", "'", line.strip()) for line in f.readlines() if line.strip()]
    logging.info(f"Loaded {len(lines)} phonics lines")
    return lines

phonic_list = load_phonics()

device = "cuda"
model = ChatterboxTTS.from_pretrained(device=device)

def generate_dataset(audio_prompt_file=None, progress=gr.Progress()):
    if not phonic_list:
        logging.error("Phonics list is empty or not found")
        return None, "Error: phonic_lines.txt not found or empty."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(f"runs/run_{timestamp}")
    wav_folder = run_folder / "wavs"
    wav_folder.mkdir(parents=True, exist_ok=True)
    meta_file = run_folder / "metadata.csv"

    logging.info(f"Generating dataset in {run_folder}")

    with meta_file.open("w", encoding="utf-8") as f:
        for i, txt in enumerate(phonic_list):
            progress((i + 1) / len(phonic_list), desc=f"Generating utterance {i+1}/{len(phonic_list)}")
            try:
                if audio_prompt_file:
                    wav = model.generate(txt, audio_prompt_path=audio_prompt_file)
                else:
                    wav = model.generate(txt)
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                wav_path = wav_folder / f"utt_{i}.wav"
                ta.save(str(wav_path), wav, model.sr)
                f.write(f"wavs/utt_{i}.wav|{txt}|{txt}\n")
            except Exception as e:
                logging.error(f"Error generating wav for '{txt}': {str(e)}")
                return None, f"Error generating wav: {str(e)}"

    logging.info("Dataset generation completed")
    return str(run_folder), f"✅ Dataset generated in {run_folder}"

class ProgressTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, progress=gr.Progress()):
        total_epochs = self.config.epochs
        logging.info(f"Starting training for {total_epochs} epochs")
        for epoch in range(total_epochs):
            self.train_epoch()
            progress((epoch + 1) / total_epochs, desc=f"Training epoch {epoch + 1}/{total_epochs}")
            logging.debug(f"Completed epoch {epoch + 1}")
        logging.info("Training complete")
        return "✅ Training complete."

def train_model(dataset_path, epochs):
    output_path = os.path.abspath("tts_output")
    os.makedirs(output_path, exist_ok=True)

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logging.error(f"Dataset path does not exist: {dataset_path}")
        return f"Error: Dataset folder {dataset_path} does not exist."
    
    metadata_path = dataset_path / "metadata.csv"
    wavs_path = dataset_path / "wavs"
    if not metadata_path.exists():
        logging.error(f"metadata.csv not found in {dataset_path}")
        return f"Error: metadata.csv not found in {dataset_path}"
    if not wavs_path.exists() or not wavs_path.is_dir():
        logging.error(f"wavs folder not found or is not a directory in {dataset_path}")
        return f"Error: wavs folder not found in {dataset_path}"

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=str(dataset_path)
    )

    config = GlowTTSConfig(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=int(epochs),
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
    )

    try:
        logging.info("Initializing AudioProcessor and Tokenizer")
        ap = AudioProcessor.init_from_config(config)
        tokenizer, config = TTSTokenizer.init_from_config(config)

        logging.info("Loading TTS samples")
        train_samples, eval_samples = load_tts_samples(
            dataset_config,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size,
        )
        if not train_samples:
            logging.error("No training samples loaded.")
            return "Error: No training samples loaded. Check dataset formatting."

        logging.info("Initializing GlowTTS model")
        model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

        logging.info("Initializing Trainer")
        trainer = Trainer(
            TrainerArgs(), config, output_path,
            model=model, train_samples=train_samples, eval_samples=eval_samples
        )

        logging.info(f"Starting training for {epochs} epochs")
        trainer.fit()
        logging.info("Training complete")
        return "✅ Training complete."

    except Exception as e:
        logging.error(f"Exception during training: {str(e)}", exc_info=True)
        return f"Error during training: {str(e)}"

def synthesize(text, audio_prompt_file=None):
    text = re.sub(r"[‘’]", "'", text)
    out_path = "tts_output/output.wav"
    try:
        if audio_prompt_file:
            wav = model.generate(text, audio_prompt_path=audio_prompt_file)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            ta.save(out_path, wav, model.sr)
        else:
            model_path = "tts_output/best_model.pth"
            config_path = "tts_output/coqui_config.json"
            if Path(model_path).exists() and Path(config_path).exists():
                cmd = (
                    f'tts --text "{text}" --model_path {model_path} '
                    f'--config_path {config_path} --out_path {out_path}'
                )
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    logging.error(f"Synthesis subprocess error: {result.stderr}")
                    return f"Error in synthesis: {result.stderr}"
            else:
                wav = model.generate(text)
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                ta.save(out_path, wav, model.sr)
        logging.info(f"Synthesis complete, saved to {out_path}")
        return out_path
    except Exception as e:
        logging.error(f"Error in synthesis: {str(e)}", exc_info=True)
        return f"Error in synthesis: {str(e)}"

def run_pipeline(audio_prompt_file=None, dataset_path=None, epochs=50, text="This is a test of the trained model.", progress=gr.Progress()):
    logging.info("Starting full pipeline")
    if not dataset_path:
        dataset_path, dataset_result = generate_dataset(audio_prompt_file, progress)
        if dataset_path is None:
            return dataset_result, None, None
    else:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            return f"Error: Dataset folder {dataset_path} does not exist.", None, None
        dataset_result = f"Using provided dataset: {dataset_path}"

    train_result = train_model(dataset_path, epochs)
    if "Error" in train_result:
        return dataset_result, train_result, None

    if not text.strip():
        return dataset_result, train_result, "Error: No text provided for synthesis."

    audio_out = synthesize(text)
    return dataset_result, train_result, audio_out

with gr.Blocks() as app:
    gr.Markdown("# 🗣️ VoiceBox Pipeline")

    with gr.Tab("Run Pipeline"):
        audio_prompt_upload = gr.Audio(
            label="Upload speaker audio for dataset generation (optional)",
            type="filepath"
        )
        dataset_path_input = gr.Textbox(
            label="Dataset Folder Path",
            placeholder="Enter path to existing dataset folder (e.g., runs/run_20250721_223600) or leave blank to generate a new dataset",
        )
        epochs_slider = gr.Slider(
            minimum=1,
            maximum=5000,
            value=50,
            step=1,
            label="Number of Training Epochs",
            info="Select the number of epochs for training the model."
        )
        text_input = gr.Textbox(
            label="Text to Synthesize",
            value="This is a test of the trained model.",
            placeholder="Enter text to synthesize after training"
        )
        run_btn = gr.Button("Run All")
        dataset_out = gr.Textbox(label="Dataset Generation Status")
        train_out = gr.Textbox(label="Training Status", lines=5)
        audio_out = gr.Audio(label="Synthesized Audio")

        run_btn.click(
            fn=run_pipeline,
            inputs=[audio_prompt_upload, dataset_path_input, epochs_slider, text_input],
            outputs=[dataset_out, train_out, audio_out],
            show_progress=True
        )

app.launch()

