import os
import pdfplumber
from docx import Document
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

# Initialize GPT-2 model and tokenizer for offline use
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure pad_token_id is set for GPT-2
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text  # Extract text from each page
    return text

# Function to extract text from Word Document
def extract_text_from_word(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"  # Combine text from all paragraphs
    return text

# Preprocess the extracted text
def preprocess_text(text):
    text = re.sub(r'\n+', '\n', text)  # Remove extra newlines
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove non-alphanumeric characters
    text = text.strip()  # Remove leading and trailing spaces
    return text

# Function to chunk long text into smaller pieces
def chunk_text(text, chunk_size=1024):
    # Split the text into chunks of the specified size
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to generate a response from GPT-2 based on a given prompt
def generate_response(prompt):
    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)

    # Generate output with GPT-2 model
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=150,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.2
    )

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to process files in the folder and generate responses
def process_folder(folder_path):
    # List all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if it's a PDF or Word document
        if file_name.endswith('.pdf'):
            print(f"Processing PDF: {file_name}")
            file_text = extract_text_from_pdf(file_path)
        elif file_name.endswith('.docx'):
            print(f"Processing Word document: {file_name}")
            file_text = extract_text_from_word(file_path)
        else:
            # Skip unsupported file formats
            print(f"Skipping unsupported file: {file_name}")
            continue

        # Preprocess the extracted text
        preprocessed_text = preprocess_text(file_text)

        # Print out the extracted and preprocessed text for debugging
        print(f"Extracted Text:\n{preprocessed_text[:1000]}...")  # Print the first 1000 characters for review

        # Chunk the text if it's too long
        chunks = chunk_text(preprocessed_text)

        # Generate a response based on the file's content
        for chunk in chunks:
            response = generate_response(chunk)
            print(f"Response for {file_name}:\n{response}\n")

# Main function for interactive chatbot
def interactive_chatbot():
    print("Welcome to the Offline GPT-2 Chatbot!")
    print("You can start chatting now. Type 'exit' to end the conversation.")

    while True:
        # Get user input (question)
        user_input = input("\nYou: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Process the input and generate a response based on local context
        response = generate_response(user_input)
        print(f"Chatbot: {response}")

# Run the interactive chatbot or process files
if __name__ == "__main__":
    mode = input("Choose mode: \n1. Interactive Chatbot\n2. Process Files in Folder\nChoose: ")

    # Specify your custom folder path here
    folder_path = "/Users/kanna/voice/data/"

    if mode == "1":
        interactive_chatbot()  # Start the interactive chatbot
    elif mode == "2":
        process_folder(folder_path)  # Process all files in the folder
    else:
        print("Invalid choice. Please run again and select a valid mode.")
