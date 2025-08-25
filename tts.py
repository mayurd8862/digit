import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
# Initialize the Groq client
client = Groq()


def transcribe_audio(client, audio_file_path):
    """Transcribe audio using Groq API"""
    with open(audio_file_path, "rb") as file:
        try:
            translation = client.audio.translations.create(
                file=(audio_file_path, file.read()),
                model="whisper-large-v3",
                response_format="json",
                temperature=0.0
            )
            return translation.text
        except Exception as e:
            return f"Error during transcription: {str(e)}"

# print(transcribe_audio(client, "conversation.mp3"))