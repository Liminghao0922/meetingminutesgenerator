
import noisereduce as nr
import librosa
import numpy as np
import soundfile as sf
import whisper
from pyannote.audio import Pipeline
import torch
import tempfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import librosa.display
from jiwer import wer
import threading
import os

class AudioProcessor:
    def __init__(self, whisper_model_path="/Users/liminghao/Documents/Models/Whisper", status_callback=None):
        """Initialize AudioProcessor"""
        self.whisper_model_path = whisper_model_path
        self.status_callback = status_callback
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.whisper_model = None
        self.diarization_pipeline = None
        self.model_lock = threading.Lock()  # Lock for model loading

    def _update_status(self, message):
        """Call status callback function (thread-safe)"""
        with threading.Lock():  # Ensure status updates are thread-safe
            if self.status_callback:
                self.status_callback(message)

    def _butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def _lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self._butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def _compress_audio(self, y, threshold=-20.0, ratio=4.0):
        y_db = 20 * np.log10(np.abs(y) + 1e-10)
        gain = np.where(y_db > threshold, threshold + (y_db - threshold) / ratio, y_db)
        y_compressed = np.sign(y) * (10 ** (gain / 20))
        return y_compressed

    def calculate_snr(self, signal, noise):
        """Calculate Signal-to-Noise Ratio (SNR) in dB"""
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)

    def load_models(self):
        """Load Whisper model and speaker diarization pipeline (thread-safe)"""
        with self.model_lock:
            if self.whisper_model is None:
                self._update_status(f"Loading Whisper model (device: {self.device})...")
                self.whisper_model = whisper.load_model(
                    name="turbo",
                    device="cpu",
                    #device="mps" if torch.backends.mps.is_available() else "cpu",
                    download_root="~/Models/Whisper")
            if self.diarization_pipeline is None:
                self._update_status("Loading speaker diarization model...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token="AUTH_TOKEN").to(torch.device(self.device))
        self._update_status("Models loaded successfully")

    def transcribe_with_whisper(self, audio_path, prompt=""):
        """Transcribe audio using Whisper (thread-safe)"""
        self._update_status(f"Starting Whisper transcription for {os.path.basename(audio_path)}...")
        with self.model_lock:  # Ensure model access is thread-safe
            if self.whisper_model is None:
                self.load_models()
            result = self.whisper_model.transcribe(
                audio_path,
                language="ja",
                prompt=prompt
            )
        self._update_status(f"Whisper transcription completed for {os.path.basename(audio_path)}")
        return result["segments"]

    def check_audio_distortion(self, audio_path):
        """Check and fix audio distortion"""
        self._update_status("Checking audio distortion...")
        y, sr = librosa.load(audio_path, sr=None)
        max_amplitude = np.max(np.abs(y))
        if max_amplitude > 1.0:
            y = y / max_amplitude
            sf.write(audio_path, y, sr)
            self._update_status("Original audio has distortion, preprocessed")
            return True
        self._update_status("No distortion in original audio")
        return False

    def plot_spectrogram(self, audio_path, title):
        """Generate spectrogram"""
        y, sr = librosa.load(audio_path, sr=None)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        plt.figure()
        librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, cmap='viridis')
        plt.title(title)
        return plt

    def evaluate_transcription(self, original_audio, clean_audio, reference_text):
        """Evaluate transcription accuracy, return WER"""
        self._update_status("Evaluating transcription accuracy...")
        orig_segments = self.transcribe_with_whisper(original_audio)
        clean_segments = self.transcribe_with_whisper(clean_audio)
        orig_result = " ".join(segment["text"] for segment in orig_segments)
        clean_result = " ".join(segment["text"] for segment in clean_segments)
        orig_wer = wer(reference_text, orig_result)
        clean_wer = wer(reference_text, clean_result)
        self._update_status("Transcription evaluation completed")
        return orig_wer, clean_wer

    def convert_and_resample(self, audio_path, target_sr=16000):
        """Convert audio to WAV and resample"""
        self._update_status("Converting audio format and resampling...")
        audio, sr = librosa.load(audio_path, sr=None)
        if sr != target_sr:
         self._update_status(f"Resampling from {sr} to {target_sr} Hz")
         audio = librosa.resample(audio, orig_sr= sr,target_sr= target_sr)
        #audio = AudioSegment.from_file(audio_path)
        #audio = audio.set_frame_rate(target_sr)
        output_path = tempfile.mktemp(suffix=".wav")
        #audio.export(output_path, format="wav")
        sf.write(output_path, audio, target_sr)
        self._update_status(f"Audio converted to WAV and resampled to {target_sr} Hz")
        return output_path

    def split_audio(self, audio_path, chunk_duration=300):
        """Split large audio file into chunks"""
        self._update_status("Starting to split large audio file...")
        y, sr = librosa.load(audio_path, sr=None)
        total_duration = librosa.get_duration(y=y, sr=sr)
        chunk_samples = int(chunk_duration * sr)
        chunks = [y[i:i + chunk_samples] for i in range(0, len(y), chunk_samples)]
        chunk_paths = []
        for i, chunk in enumerate(chunks):
            chunk_path = tempfile.mktemp(suffix=f"_chunk_{i}.wav")
            sf.write(chunk_path, chunk, sr)
            chunk_paths.append(chunk_path)
        self._update_status(f"Audio split into {len(chunk_paths)} chunks")
        return chunk_paths

    def reduce_noise(self, audio_path, output_path=None, prop_decrease=0.75, high_freq_cutoff=4000, chunk_duration=300):
        """Noise reduction with chunking for large files"""
        self._update_status("Starting noise reduction...")

        # Convert to WAV and resample
        wav_path = self.convert_and_resample(audio_path)

        # Check if chunking is needed
        total_duration = librosa.get_duration(filename=wav_path)
        if total_duration > chunk_duration:
            chunk_paths = self.split_audio(wav_path)
            clean_chunks = []
            for i, chunk_path in enumerate(chunk_paths):
                self._update_status(f"Reducing noise for chunk {i + 1}/{len(chunk_paths)}...")
                y, sr = librosa.load(chunk_path, sr=None)
                y_clean = nr.reduce_noise(y, sr=sr, stationary=False, prop_decrease=prop_decrease)
                y_clean = self._lowpass_filter(y_clean, cutoff=high_freq_cutoff, fs=sr)
                y_clean = self._compress_audio(y_clean)
                y_clean = y_clean / np.max(np.abs(y_clean))
                clean_chunk_path = tempfile.mktemp(suffix=f"_clean_chunk_{i}.wav")
                sf.write(clean_chunk_path, y_clean, sr)
                clean_chunks.append(clean_chunk_path)
                os.unlink(chunk_path)
            # Merge chunks
            y_full = []
            for clean_chunk in clean_chunks:
                y_chunk, _ = librosa.load(clean_chunk, sr=sr)
                y_full.append(y_chunk)
                os.unlink(clean_chunk)
            y_clean = np.concatenate(y_full)
        else:
            y, sr = librosa.load(wav_path, sr=None)
            y_clean = nr.reduce_noise(y, sr=sr, stationary=False, prop_decrease=prop_decrease)
            y_clean = self._lowpass_filter(y_clean, cutoff=high_freq_cutoff, fs=sr)
            y_clean = self._compress_audio(y_clean)
            y_clean = y_clean / np.max(np.abs(y_clean))

        # Calculate SNR
        noise_len = int(0.5 * sr)
        noise = y[:noise_len]
        original_snr = self.calculate_snr(y[noise_len:], noise)
        cleaned_snr = self.calculate_snr(y_clean[noise_len:], noise)
        self._update_status(f"Noise reduction result - Original SNR: {original_snr:.2f} dB, Cleaned SNR: {cleaned_snr:.2f} dB")

        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")
        sf.write(output_path, y_clean, sr)
        self._update_status("Noise reduction completed")
        if wav_path != audio_path:
            os.unlink(wav_path)
        return output_path

    def transcribe_audio_with_speakers(self, audio_path, prompt="", max_threads=2):
        """Transcribe audio and identify speakers using multi-threading"""
        self._update_status("Starting audio processing...")

        # Check if chunking is needed
        total_duration = librosa.get_duration(filename=audio_path)
        if total_duration > 300:
            chunk_paths = self.split_audio(audio_path)
            use_chunks = True
        else:
            chunk_paths = [audio_path]
            use_chunks = False

        # Load models
        self.load_models()

        final_text = []
        threads = []
        results = [None] * len(chunk_paths)
        result_lock = threading.Lock()

        def process_chunk(index, chunk_path):
            """Process a single chunk: noise reduction, transcription, and diarization"""
            self._update_status(f"Processing chunk {index + 1}/{len(chunk_paths)}...")
            clean_chunk_path = self.reduce_noise(chunk_path)
            segments = self.transcribe_with_whisper(clean_chunk_path, prompt)
            self._update_status(f"Chunk {index + 1} speaker diarization started...")
            diarization = self.diarization_pipeline(clean_chunk_path)

            annotated_text = []
            current_speaker = None
            for segment in segments:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]

                speaker = None
                for turn, _, spk in diarization.itertracks(yield_label=True):
                    if turn.start <= start_time and turn.end >= end_time:
                        speaker = spk
                        break
                if not speaker:
                    speaker = "Unknown"

                if speaker != current_speaker:
                    annotated_text.append(f"[Speaker_{speaker[-2:]}] {text}")
                    current_speaker = speaker
                else:
                    annotated_text[-1] += f" {text}"

            with result_lock:
                results[index] = annotated_text

            if chunk_path != audio_path:
                os.unlink(chunk_path)
            os.unlink(clean_chunk_path)

        for i, chunk_path in enumerate(chunk_paths):
            process_chunk(i,chunk_path=chunk_path)
        # # Limit the number of concurrent threads
        # for i, chunk_path in enumerate(chunk_paths):
        #     thread = threading.Thread(target=process_chunk, args=(i, chunk_path))
        #     threads.append(thread)
        #     thread.start()
        #     # Control max concurrent threads
        #     if len([t for t in threads if t.is_alive()]) >= max_threads:
        #         for t in threads:
        #             t.join()

        # # Wait for all threads to complete
        # for thread in threads:
        #     thread.join()

        # Combine results in order
        final_text = [item for sublist in results for item in sublist if sublist is not None]
        result = "\n".join(final_text)
        self._update_status("Audio transcription and speaker identification completed")
        return result, use_chunks