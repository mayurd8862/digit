# import os
# from groq import Groq
# from dotenv import load_dotenv
# load_dotenv()
# # Initialize the Groq client
# client = Groq()


# def transcribe_audio(client, audio_file_path):
#     """Transcribe audio using Groq API"""
#     with open(audio_file_path, "rb") as file:
#         try:
#             translation = client.audio.translations.create(
#                 file=(audio_file_path, file.read()),
#                 model="whisper-large-v3",
#                 response_format="json",
#                 temperature=0.0
#             )
#             return translation.text
#         except Exception as e:
#             return f"Error during transcription: {str(e)}"

# # print(transcribe_audio(client, "conversation.mp3"))


from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def transcribe_audio(client, audio_file_path):
    """Transcribe audio using Google API"""
    with open(audio_file_path, "rb") as file:
        audio_bytes = file.read()
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    "Transcribe this audio clip and transcribe it in English:",
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type="audio/mp3",
                    ),
                ],
            )
            return response.text
        except Exception as e:
            return f"Error during transcription: {str(e)}"
