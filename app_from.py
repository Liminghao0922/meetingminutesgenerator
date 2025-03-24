# import streamlit as st
# import os
# import tempfile
# from src.audio_processor import transcribe_audio_with_speakers
# from src.audio_processor import preprocess_audio
# from src.text_optimizer import optimize_text_with_deepseek
# from src.doc_generator import generate_minutes

# def main():
#     st.title("会議記録生成ツール")
#     st.write("オーディオファイルをアップロードして、自動的に会議記録を生成します。")

#     # 用户输入模型路径
#     model_path = st.text_input("DeepSeek R1モデルパス", "/Users/liminghao/Documents/Models/cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-Q4_K_M.gguf")
#     if not os.path.exists(model_path):
#         st.error("モデルファイルが見つかりません。正しいパスを指定してください。")
#         return

#     # 上传音频文件
#     uploaded_file = st.file_uploader("オーディオファイルをアップロード", type=["mp3", "wav", "m4a"])
#     if uploaded_file is not None:
#         with tempfile.NamedTemporaryFile() as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             audio_path = preprocess_audio(tmp_file.name)
        
#         st.write(f"アップロードされたファイル: {uploaded_file.name}")
#        # audio_path = preprocess_audio(audio_path)
#         if st.button("会議記録を生成"):
#             with st.spinner("処理中..."):
#                 # Step 1: 转录并识别发言人
#                 st.write("音频转录开始...")
#                 raw_text = transcribe_audio_with_speakers(audio_path)
#                 st.subheader("転写結果（发言人标签付き）")
#                 st.text_area("Raw Text", raw_text, height=200)

#                 # Step 2: 使用DeepSeek R1优化
#                 st.write("DeepSeek R1优化开始...")
#                 optimized_text = optimize_text_with_deepseek(raw_text, model_path)
#                 st.subheader("最適化結果")
#                 st.text_area("Optimized Text", optimized_text, height=200)

#                 # Step 3: 生成会议记录并提供下载
#                 output_path = os.path.join("data/output", "meeting_minutes.docx")
#                 os.makedirs("data/output", exist_ok=True)
#                 generate_minutes(optimized_text, output_path)
#                 with open(output_path, "rb") as file:
#                     st.download_button(
#                         label="会議記録をダウンロード",
#                         data=file,
#                         file_name="meeting_minutes.docx",
#                         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#                     )
                
#                 # 清理临时文件
#                 #os.unlink(audio_path)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import os
# import tempfile
# import time
# import matplotlib.pyplot as plt
# import librosa
# import librosa.display
# from jiwer import wer
# import audio_processor
# from src.audio_processor import transcribe_audio_with_speakers, reduce_noise
# from src.text_optimizer import optimize_text_with_phi4
# from src.doc_generator import generate_minutes
# import numpy as np
# def check_audio_distortion(audio_path):
#     y, sr = librosa.load(audio_path, sr=None)
#     max_amplitude = np.max(np.abs(y))
#     if max_amplitude > 1.0:
#         y = y / max_amplitude
#         sf.write(audio_path, y, sr)
#         return True
#     return False

# def plot_spectrogram(audio_path, title, col):
#     y, sr = librosa.load(audio_path, sr=None)
#     D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
#     librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, cmap='viridis')
#     plt.title(title)
#     col.pyplot(plt)

# def evaluate_transcription(original_audio, clean_audio, reference_text):
#     # whisper_model = whisper.load_model("~/MeetingMinutesGenerator/models/whisper/medium")
#     # orig_result = whisper_model.transcribe(original_audio, language="ja")["text"]
#     # clean_result = whisper_model.transcribe(clean_audio, language="ja")["text"]
#     orig_result = audio_processor.transcribe_audio(original_audio)["text"]
#     clean_result = audio_processor.transcribe_audio(clean_audio)["text"]

#     orig_wer = wer(reference_text, orig_result)
#     clean_wer = wer(reference_text, clean_result)
#     return orig_wer, clean_wer

# def main():
#     st.title("会議記録生成ツール")
#     st.write("オーディオファイルをアップロードして、自動的に会議記録を生成します。")

#     model_path = st.text_input("Phi-4モデルパス", "~/MeetingMinutesGenerator/models/phi4/")
#     whisper_prompt = st.text_input("Whisper 提示词（专业术语，用逗号分隔）", "人工知能, 機械学習, ディープラーニング")
#     phi4_keywords = st.text_input("Phi-4 关键字（需修正的术语，用逗号分隔）", "人工知能, 機械学習, ディープラーニング, クラウドコンピューティング, API, ブロックチェーン")
#     uploaded_file = st.file_uploader("オーディオファイルをアップロード", type=["mp3", "wav","m4a"])

#     status_placeholder = st.empty()
#     def update_status(message):
#         status_placeholder.write(f"現在の処理状況: {message}")

#     if uploaded_file is not None:
#         with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             audio_path = tmp_file.name

#         if check_audio_distortion(audio_path):
#             st.warning("原始音频存在失真，已预处理")
#         st.write(f"アップロードされたファイル: {uploaded_file.name}")
#         st.audio(audio_path, format="audio/mp3")

#         clean_audio_path = reduce_noise(audio_path, status_callback=update_status)
#         st.write("降噪後のオーディオ:")
#         st.audio(clean_audio_path, format="audio/wav")

#         col1, col2 = st.columns(2)
#         plot_spectrogram(audio_path, "原始音频频谱", col1)
#         plot_spectrogram(clean_audio_path, "降噪后音频频谱", col2)

#         reference_text = st.text_area("参考文本（用于评估降噪效果，可选）", "")
#         if reference_text:
#             orig_wer, clean_wer = evaluate_transcription(audio_path, clean_audio_path, reference_text)
#             st.write(f"原始音频 WER: {orig_wer:.2f}, 降噪后 WER: {clean_wer:.2f}")

#         if st.button("会議記録を生成"):
#             start_time = time.time()
#             with st.spinner("処理中..."):
#                 raw_text = transcribe_audio_with_speakers(audio_path, whisper_prompt, update_status)
#                 st.subheader("転写結果（发言人标签付き）")
#                 st.text_area("Raw Text", raw_text, height=200)

#                 optimized_text = optimize_text_with_phi4(raw_text, model_path, phi4_keywords, update_status)
#                 st.subheader("最適化結果")
#                 st.text_area("Optimized Text", optimized_text, height=200)

#                 output_path = os.path.join("data/output", "meeting_minutes.docx")
#                 os.makedirs("data/output", exist_ok=True)
#                 generate_minutes(optimized_text, output_path)
#                 with open(output_path, "rb") as file:
#                     st.download_button(
#                         label="会議記録をダウンロード",
#                         data=file,
#                         file_name="meeting_minutes.docx",
#                         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#                     )

#                 end_time = time.time()
#                 st.write(f"全体の処理時間: {end_time - start_time:.2f} 秒")

#             os.unlink(audio_path)
#             os.unlink(clean_audio_path)

# if __name__ == "__main__":
#     main()

import streamlit as st
import os
import tempfile
import time
from src.audio_processor import AudioProcessor
from src.text_optimizer import optimize_text_with_deepseek
from src.doc_generator import generate_minutes
import librosa

def main():
    st.title("Meeting Minutes Generator")
    st.write("Upload an audio file to automatically generate meeting minutes.")

    model_path = st.text_input("DeepSeek R1 Model Path", "/Users/liminghao/Documents/Models/cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-Q4_K_M.gguf")
    whisper_prompt = st.text_input("Whisper Prompt (comma-separated terms)", "artificial intelligence, machine learning, deep learning")
    phi4_keywords = st.text_input("DeepSeek Keywords (comma-separated terms to correct)", "artificial intelligence, machine learning, deep learning, cloud computing, API, blockchain")
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])

    status_placeholder = st.empty()
    def update_status(message):
        status_placeholder.write(f"Current Status: {message}")

    processor = AudioProcessor(status_callback=update_status)

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name

        if processor.check_audio_distortion(audio_path):
            st.warning("Original audio has distortion, preprocessed")
        st.write(f"Uploaded File: {uploaded_file.name}")
        st.audio(audio_path, format=f"audio/{uploaded_file.name.split('.')[-1]}")

        clean_audio_path = processor.reduce_noise(audio_path)
        st.write("Audio After Noise Reduction:")
        st.audio(clean_audio_path, format="audio/wav")

        col1, col2 = st.columns(2)
        orig_plot = processor.plot_spectrogram(audio_path, "Original Audio Spectrogram")
        clean_plot = processor.plot_spectrogram(clean_audio_path, "Cleaned Audio Spectrogram")
        col1.pyplot(orig_plot)
        col2.pyplot(clean_plot)

        reference_text = st.text_area("Reference Text (optional, for noise reduction evaluation)", "")
        if reference_text:
            total_duration = librosa.get_duration(filename=audio_path)
            if total_duration <= 300:
                orig_wer, clean_wer = processor.evaluate_transcription(audio_path, clean_audio_path, reference_text)
                st.write(f"Original Audio WER: {orig_wer:.2f}, Cleaned Audio WER: {clean_wer:.2f}")
            else:
                st.warning("Audio has been chunked, cannot evaluate full transcription accuracy")

        if st.button("Generate Meeting Minutes"):
            start_time = time.time()
            with st.spinner("Processing..."):
                raw_text, use_chunks = processor.transcribe_audio_with_speakers(audio_path, whisper_prompt, max_threads=4)
                st.subheader("Transcription Result (with Speaker Tags)")
                st.text_area("Raw Text", raw_text, height=200)

                optimized_text = optimize_text_with_deepseek(raw_text, model_path, phi4_keywords, update_status)
                st.subheader("Optimized Result")
                st.text_area("Optimized Text", optimized_text, height=200)

                output_path = os.path.join("data/output", "meeting_minutes.docx")
                os.makedirs("data/output", exist_ok=True)
                generate_minutes(optimized_text, output_path)
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Meeting Minutes",
                        data=file,
                        file_name="meeting_minutes.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

                end_time = time.time()
                st.write(f"Total Processing Time: {end_time - start_time:.2f} seconds")

            os.unlink(audio_path)
            os.unlink(clean_audio_path)

if __name__ == "__main__":
    main()