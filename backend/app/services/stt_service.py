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
            response.raise_for_status()
            res_json = response.json()
            transcript = res_json.get("transcript", "").strip()
            logger.info(f"Successfully transcribed audio: '{transcript}'")
            return transcript
        except Exception as e:
            logger.error(f"Sarvam STT failed: {e}")
            raise e

# Singleton instance
stt_service = STTService()
