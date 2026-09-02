import requests
from loguru import logger
from app.config import settings

class STTService:
    def __init__(self):
        self.endpoint = "https://api.sarvam.ai/speech-to-text"

    def transcribe(self, audio_bytes: bytes, filename: str = "query.wav") -> str:
        """
        Sends audio bytes to Sarvam Speech-to-Text API.
        If no API key is configured, defaults to a mock transcription for demo/test purposes.
        """
        api_key = settings.SARVAM_API_KEY
        if not api_key or "your_sarvam" in api_key.lower() or api_key == "":
            logger.warning("SARVAM_API_KEY is not configured or is placeholder. Returning mock transcription.")
            return "कॉर्पोरेशन क्या है?"

        headers = {
            "api-subscription-key": api_key
        }
        
        # Sarvam expects multipart/form-data
        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        
        data = {
            "model": "saaras:v3",
            "mode": "transcribe"
        }

        try:
            logger.info("Sending STT request to Sarvam...")
            response = requests.post(
                self.endpoint,
                headers=headers,
                files=files,
                data=data,
                timeout=15.0  # Safe timeout for audio processing
            )
            if response.status_code == 402:
                logger.warning("Sarvam STT quota exhausted (402). Falling back to Groq Whisper...")
                return self._transcribe_with_groq(audio_bytes, filename)
            
            if response.status_code != 200:
                logger.warning(f"Sarvam STT returned status {response.status_code}. Falling back to Groq Whisper...")
                return self._transcribe_with_groq(audio_bytes, filename)

            res_json = response.json()
            transcript = res_json.get("transcript", "").strip()
            if not transcript:
                logger.warning("Sarvam returned empty transcript, trying Whisper...")
                return self._transcribe_with_groq(audio_bytes, filename)
            
            logger.info(f"Successfully transcribed audio via Sarvam: '{transcript}'")
            return transcript
        except Exception as e:
            logger.warning(f"Sarvam STT failed ({e}), attempting Groq Whisper fallback...")
            try:
                return self._transcribe_with_groq(audio_bytes, filename)
            except Exception as fe:
                logger.error(f"All STT providers failed: {fe}")
                raise fe

    def _transcribe_with_groq(self, audio_bytes: bytes, filename: str) -> str:
        """
        Fallback speech-to-text using Groq Whisper (whisper-large-v3-turbo).
        Ensures voice recognition remains 100% operational even if Sarvam credits run out.
        """
        api_key = settings.GROQ_API_KEY
        if not api_key or "your_groq" in api_key.lower():
            raise Exception("Sarvam AI quota exhausted and Groq API key is not configured.")

        from groq import Groq
        groq_client = Groq(api_key=api_key)
        logger.info("Transcribing audio via Groq Whisper fallback (whisper-large-v3-turbo)...")
        res = groq_client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            language="hi"
        )
        transcript = res.text.strip()
        if not transcript:
            raise Exception("No clear speech detected in audio.")
        logger.info(f"Groq Whisper transcribed successfully: '{transcript}'")
        return transcript

# Singleton instance
stt_service = STTService()
