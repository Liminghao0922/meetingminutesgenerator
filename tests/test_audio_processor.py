import unittest
from src.audio_processor import transcribe_audio_with_speakers

class TestAudioProcessor(unittest.TestCase):
    def test_transcription(self):
        result = transcribe_audio_with_speakers("data/input/meeting_audio.mp3")
        self.assertTrue(isinstance(result, str))
        self.assertIn("[Speaker_", result)

if __name__ == "__main__":
    unittest.main()