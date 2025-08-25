# import os
# import pandas as pd
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# import streamlit as st

# EXCEL_FILE = "data.xlsx"

# # 1. Define response schema with updated sentiment labels
# response_schemas = [
#     ResponseSchema(name="name", description="Person's name in the conversation"),
#     ResponseSchema(name="age", description="Age of the person if mentioned"),
#     ResponseSchema(name="policytype", description="Type of insurance policy"),
#     ResponseSchema(name="address_with_pin", description="Full address including pincode if available"),
#     ResponseSchema(
#         name="sentiment",
#         description="Overall sentiment: Interested but Occupied, Interested, Not Interested"
#     ),
#     ResponseSchema(name="summary", description="Short summary of the conversation")
# ]

# # 2. Create parser
# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# # 3. Get format instructions
# format_instructions = output_parser.get_format_instructions()

# # 4. Build prompt
# prompt = PromptTemplate(
#     template="""
# Extract the following features from the conversation in ENGLISH:
# - Name
# - Age
# - Policy Type
# - Address with pin
# - Sentiment (choose one: Interested but Occupied, Interested, Not Interested)
# - Summary

# Conversation: {conversation}

# {format_instructions}
# """,
#     input_variables=["conversation"],
#     partial_variables={"format_instructions": format_instructions}
# )

# # 5. LLM (Gemini)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# def extract_features(conversation: str):
#     _input = prompt.format_prompt(conversation=conversation)
#     output = llm(_input.to_messages())
#     parsed = output_parser.parse(output.content)
#     return parsed  # returns dict with all fields

# def save_to_excel(data: dict):
#     if not os.path.exists(EXCEL_FILE):
#         raise FileNotFoundError(f"{EXCEL_FILE} not found. Please make sure it exists.")

#     df_existing = pd.read_excel(EXCEL_FILE)
#     df_new = pd.DataFrame([data])

#     df_combined = pd.concat([df_existing, df_new], ignore_index=True)
#     df_combined.to_excel(EXCEL_FILE, index=False)
#     print(f"Data inserted into {EXCEL_FILE}")


# # st.set_page_config(page_title="Insurance Conversation Extractor", layout="wide")

# st.title("Insurance Conversation Feature Extractor")

# conversation_input = st.text_area("Conversation", height=200)

# if st.button("Extract & Save"):
#     if conversation_input.strip() == "":
#         st.warning("Please enter a conversation!")
#     else:
#         try:
#             # Extract features
#             extracted_data = extract_features(conversation_input)
            
#             # Display extracted data
#             st.subheader("Extracted Data")
#             # st.json(extracted_data)
            
#             # Save to Excel
#             save_to_excel(extracted_data)
#             st.success(f"Data inserted into {EXCEL_FILE} successfully!")
#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# # Optionally, display the existing Excel file
# if st.checkbox("Show Existing Data"):
#     try:
#         df_existing = pd.read_excel(EXCEL_FILE)
#         st.subheader("Existing Data in Excel")
#         st.dataframe(df_existing)
#     except FileNotFoundError:
#         st.warning(f"{EXCEL_FILE} not found.")





import os
import tempfile
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import streamlit as st
import whisper  # Using OpenAI Whisper locally
from utils.tts import transcribe_audio, client

EXCEL_FILE = "data.xlsx"

# ------------------ Define Response Schema ------------------
response_schemas = [
    ResponseSchema(name="name", description="Person's name in the conversation"),
    ResponseSchema(name="age", description="Age of the person if mentioned"),
    ResponseSchema(name="policytype", description="Type of insurance policy"),
    ResponseSchema(name="address_with_pin", description="Full address including pincode if available"),
    ResponseSchema(
        name="sentiment",
        description="Overall sentiment: Interested but Occupied, Interested, Not Interested"
    ),
    ResponseSchema(name="summary", description="Short summary of the conversation")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="""
Extract the following features from the conversation in ENGLISH:
- Name
- Age
- Policy Type
- Address with pin
- Sentiment (choose one: Interested but Occupied, Interested, Not Interested)
- Summary

Conversation: {conversation}

{format_instructions}
""",
    input_variables=["conversation"],
    partial_variables={"format_instructions": format_instructions}
)

# ------------------ LLM ------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# ------------------ Functions ------------------
def extract_features(conversation: str):
    _input = prompt.format_prompt(conversation=conversation)
    output = llm(_input.to_messages())
    parsed = output_parser.parse(output.content)
    return parsed

def save_to_excel(data: dict):
    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(f"{EXCEL_FILE} not found. Please make sure it exists.")
    df_existing = pd.read_excel(EXCEL_FILE)
    df_new = pd.DataFrame([data])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_excel(EXCEL_FILE, index=False)
    print(f"Data inserted into {EXCEL_FILE}")

# ------------------ Streamlit UI ------------------
st.title("Insurance Conversation Feature Extractor")

# Add radio buttons for input choice
input_type = st.radio("Choose input type:", ("Text Input", "Upload MP3"))

conversation_input = ""

if input_type == "Text Input":
    conversation_input = st.text_area("Enter conversation text", height=200)
else:
    audio_file = st.file_uploader("Upload MP3 file", type=["mp3"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        conversation_input = transcribe_audio(client, tmp_file_path)
        st.info("Transcribed Text:")
        st.text(conversation_input)
        os.remove(tmp_file_path)  # remove temp file after use

if st.button("Extract & Save"):
    if conversation_input.strip() == "":
        st.warning("Please provide input!")
    else:
        try:
            # Extract features
            extracted_data = extract_features(conversation_input)
            
            # Display extracted data
            st.subheader("Extracted Data")
            st.json(extracted_data)
            
            # Save to Excel
            save_to_excel(extracted_data)
            st.success(f"Data inserted into {EXCEL_FILE} successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Show existing Excel data
if st.checkbox("Show Existing Data"):
    try:
        df_existing = pd.read_excel(EXCEL_FILE)
        st.subheader("Existing Data in Excel")
        st.dataframe(df_existing)
    except FileNotFoundError:
        st.warning(f"{EXCEL_FILE} not found.")
