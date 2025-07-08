import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import torchaudio as ta
import os
from chatterbox.tts import ChatterboxTTS

class VoiceBoxApp:
    def __init__(self, root):
        self.root = root
        root.title("VoiceBox")

        self.speaker_wav_path = None

        self.text = tk.Text(root, wrap='word', height=10)
        self.text.pack(expand=True, fill='both', padx=10, pady=10)

        self.slider = tk.Scale(root, from_=1, to=500, orient='horizontal',
                               label='Batch size (characters)',
                               length=300)
        self.slider.set(200)
        self.slider.pack(padx=10, pady=5)

        self.speaker_button = tk.Button(root, text="Select Speaker WAV", command=self.select_speaker_wav)
        self.speaker_button.pack(pady=(10, 0))

        self.speaker_label = tk.Label(root, text="No speaker selected", fg="gray")
        self.speaker_label.pack(pady=(0, 10))

        self.gen_button = tk.Button(root, text="Generate", command=self.on_generate)
        self.gen_button.pack(pady=10)

        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = ChatterboxTTS.from_pretrained(device="cpu")
        except Exception as e:
            messagebox.showerror("Error", f"Failed loading model:\n{e}")

    def select_speaker_wav(self):
        file_path = filedialog.askopenfilename(
            title="Select Speaker WAV File",
            filetypes=[("WAV files", "*.wav")]
        )
        if file_path:
            self.speaker_wav_path = file_path
            self.speaker_label.config(text=os.path.basename(file_path), fg="black")
        else:
            self.speaker_label.config(text="No speaker selected", fg="gray")

    def on_generate(self):
        text = self.text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input needed", "Please enter some text.")
            return
        batch_size = self.slider.get()
        self.gen_button.config(state='disabled')
        threading.Thread(target=self.generate_batch, args=(text, batch_size), daemon=True).start()

    def generate_batch(self, text, batch_size):
        try:
            os.makedirs("chunks", exist_ok=True)
            chunks = [text[i:i+batch_size] for i in range(0, len(text), batch_size)]
            out_files = []
            for idx, chunk in enumerate(chunks, start=1):
                wav = self.model.generate(chunk, speaker_wav=self.speaker_wav_path)
                file_name = os.path.join("chunks", f"voicebox_out_{idx}.wav")
                ta.save(file_name, wav, self.model.sr)
                out_files.append(file_name)
            messagebox.showinfo("Done", f"Generated {len(out_files)} files in /chunks")
        except Exception as e:
            messagebox.showerror("Error", f"Generation failed:\n{e}")
        finally:
            self.gen_button.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceBoxApp(root)
    root.geometry("500x500")
    root.mainloop()
