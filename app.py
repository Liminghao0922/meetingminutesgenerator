import streamlit as st
import os
import tempfile
import time
import json
from src.audio_processor import AudioProcessor
from src.text_optimizer import TextOptimizer
from src.doc_generator import generate_minutes
import librosa

CONFIG_FILE = "config.json"

def load_config():
    """Load configuration from config.json or return default values"""
    default_config = {
        "AudioProcess": {
            "prop_decrease": 0.75,
            "high_freq_cutoff": 4000,
            "target_sr": 16000,
            "chunk_duration": 300
        },
        "STT": {
            "whisper_model": "medium",
            "whisper_prompt": "artificial intelligence, machine learning, deep learning"
        },
        "LLM": {
            "deepseek_model_path": "~/MeetingMinutesGenerator/models/deepseek/DeepSeek-R1-Distill-Qwen-14B-Japanese-Q4_K_M.gguf",
            "n_ctx": 32768,
            "max_tokens": 1000,
            "temperature": 0.6,
            "top_p": 0.9,
            "domain_terms": "人工知能, 機械学習",
            "misrecognized_words": "AI→人工知能, ML→機械学習"
        }
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            loaded_config = json.load(f)
            default_config["AudioProcess"].update(loaded_config.get("AudioProcess", {}))
            default_config["STT"].update(loaded_config.get("STT", {}))
            default_config["LLM"].update(loaded_config.get("LLM", {}))
            return default_config
    return default_config

def save_config(config):
    """Save configuration to config.json"""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def main():
    st.title("Meeting Minutes Generator - Wizard Mode")
    st.write("Follow the steps below to generate meeting minutes from an audio file.")

    # Load initial config
    config = load_config()

    # Initialize session state for wizard steps and results
    if "step" not in st.session_state:
        st.session_state.step = 1
        st.session_state.audio_path = None
        st.session_state.clean_audio_path = None
        st.session_state.raw_text = None
        st.session_state.optimized_text = None
        st.session_state.text_optimizer = None

    # Wizard steps
    steps = {
        1: "Step 1: Preprocess Audio",
        2: "Step 2: Speech-to-Text (STT)",
        3: "Step 3: Optimize Transcription"
    }

    # Sidebar for step navigation
    st.sidebar.title("Progress")
    for i in range(1, 4):
        if st.session_state.step >= i:
            st.sidebar.write(f"✅ {steps[i]}")
        else:
            st.sidebar.write(f"⬜ {steps[i]}")

    # Status placeholder
    status_placeholder = st.empty()
    def update_status(message):
        status_placeholder.write(f"Current Status: {message}")

    processor = AudioProcessor(status_callback=update_status)

    # Step 1: Preprocess Audio
    if st.session_state.step == 1:
        st.header(steps[1])
        st.write("Upload your audio file and preprocess it.")

        uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"], key="step1_upload")
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.audio_path = tmp_file.name

            st.subheader("Original Audio")
            st.audio(st.session_state.audio_path, format=f"audio/{uploaded_file.name.split('.')[-1]}")

            if uploaded_file.name.split('.')[-1] != "wav":
                st.write("Transcoding to WAV...")
                st.session_state.audio_path = processor.convert_and_resample(st.session_state.audio_path)
                processor._process_status_queue(status_placeholder)
                st.write("Transcoding completed. Updated audio:")
                st.audio(st.session_state.audio_path, format="audio/wav")

            st.subheader("Preprocessing Options")
            prop_decrease = st.slider("Noise Reduction Strength (prop_decrease)", 0.0, 1.0, config["AudioProcess"]["prop_decrease"])
            high_freq_cutoff = st.slider("High Frequency Cutoff (Hz)", 1000, 8000, config["AudioProcess"]["high_freq_cutoff"])
            target_sr = st.selectbox("Sample Rate (Hz)", [16000, 22050, 44100], index=[16000, 22050, 44100].index(config["AudioProcess"]["target_sr"]))
            chunk_duration = st.slider("Chunk Duration (seconds)", 60, 600, config["AudioProcess"]["chunk_duration"])

            if st.button("Preprocess Audio"):
                st.session_state.clean_audio_path = processor.reduce_noise(
                    st.session_state.audio_path,
                    prop_decrease=prop_decrease,
                    high_freq_cutoff=high_freq_cutoff,
                    chunk_duration=chunk_duration
                )
                processor._process_status_queue(status_placeholder)
                st.subheader("Preprocessed Audio")
                st.audio(st.session_state.clean_audio_path, format="audio/wav")

                config["AudioProcess"]["prop_decrease"] = prop_decrease
                config["AudioProcess"]["high_freq_cutoff"] = high_freq_cutoff
                config["AudioProcess"]["target_sr"] = target_sr
                config["AudioProcess"]["chunk_duration"] = chunk_duration
                save_config(config)

                st.session_state.step = 2
                st.experimental_rerun()

    # Step 2: Speech-to-Text (STT)
    elif st.session_state.step == 2:
        st.header(steps[2])
        st.write("Convert the preprocessed audio to text using Whisper.")

        if st.session_state.clean_audio_path:
            st.subheader("Preprocessed Audio")
            st.audio(st.session_state.clean_audio_path, format="audio/wav")

            st.subheader("STT Options")
            whisper_model = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large"], index=["tiny", "base", "small", "medium", "large"].index(config["STT"]["whisper_model"]))
            whisper_prompt = st.text_input("Whisper Prompt (comma-separated terms)", config["STT"]["whisper_prompt"])

            if st.button("Run STT"):
                processor.whisper_model_path = f"~/MeetingMinutesGenerator/models/whisper/{whisper_model}"
                st.session_state.raw_text, _ = processor.transcribe_audio_with_speakers(
                    st.session_state.clean_audio_path,
                    prompt=whisper_prompt,
                    max_threads=4
                )
                processor._process_status_queue(status_placeholder)
                st.subheader("STT Result")
                st.text_area("Raw Transcription", st.session_state.raw_text, height=200)

                config["STT"]["whisper_model"] = whisper_model
                config["STT"]["whisper_prompt"] = whisper_prompt
                save_config(config)

                st.session_state.step = 3
                st.experimental_rerun()

    # Step 3: Optimize Transcription
    elif st.session_state.step == 3:
        st.header(steps[3])
        st.write("Optimize the transcription text using DeepSeek.")

        if st.session_state.raw_text:
            st.subheader("Raw Transcription")
            st.text_area("Raw Transcription", st.session_state.raw_text, height=200)

            st.subheader("Optimization Options")
            model_path = st.text_input("DeepSeek Model Path", config["LLM"]["deepseek_model_path"])
            n_ctx = st.number_input("Context Window Size", min_value=2048, max_value=32768, value=config["LLM"]["n_ctx"], step=2048)
            max_tokens = st.number_input("Max Tokens", min_value=1000, max_value=32768, value=config["LLM"]["max_tokens"])
            temperature = st.slider("Temperature", 0.1, 1.0, config["LLM"]["temperature"])
            top_p = st.slider("Top P", 0.1, 1.0, config["LLM"]["top_p"])
            domain_terms = st.text_input("Domain-Specific Terms (comma-separated)", config["LLM"]["domain_terms"], help="例: 人工知能, 機械学習")
            misrecognized_words = st.text_input("Misrecognized Words (format: wrong→correct, comma-separated)", config["LLM"]["misrecognized_words"], help="例: AI→人工知能, ML→機械学習")

            if st.button("Optimize Transcription"):
                # Initialize TextOptimizer if not already done
                if st.session_state.text_optimizer is None:
                    try:
                        st.session_state.text_optimizer = TextOptimizer(pretrained_model_name_or_path=model_path)
                        update_status("TextOptimizer initialized successfully.")
                    except Exception as e:
                        st.error(f"Failed to initialize TextOptimizer: {e}")
                        return

                st.session_state.optimized_text = st.session_state.text_optimizer.optimize_text(
                    raw_text=st.session_state.raw_text,
                    domain_terms=domain_terms,
                    misrecognized_words=misrecognized_words,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                processor._process_status_queue(status_placeholder)
                st.subheader("Optimized Transcription")
                st.text_area("Optimized Text", st.session_state.optimized_text, height=200)

                if st.button("Generate Meeting Minutes (Optional)"):
                    output_path = os.path.join("data/output", "meeting_minutes.docx")
                    os.makedirs("data/output", exist_ok=True)
                    generate_minutes(st.session_state.optimized_text, output_path)
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="Download Meeting Minutes",
                            data=file,
                            file_name="meeting_minutes.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )

                config["LLM"]["deepseek_model_path"] = model_path
                config["LLM"]["n_ctx"] = n_ctx
                config["LLM"]["max_tokens"] = max_tokens
                config["LLM"]["temperature"] = temperature
                config["LLM"]["top_p"] = top_p
                config["LLM"]["domain_terms"] = domain_terms
                config["LLM"]["misrecognized_words"] = misrecognized_words
                save_config(config)

                if st.button("Start Over"):
                    st.session_state.step = 1
                    st.session_state.audio_path = None
                    st.session_state.clean_audio_path = None
                    st.session_state.raw_text = None
                    st.session_state.optimized_text = None
                    st.session_state.text_optimizer = None
                    st.experimental_rerun()

if __name__ == "__main__":
    main()