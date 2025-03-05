import os
import pdfplumber
from docx import Document
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

# Initialize GPT-2 model and tokenizer
model_name = "gpt2"  # You can try "gpt2-medium" or "gpt2-small" if you face memory issues
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

# Preprocess extracted text
def preprocess_text(text):
    text = re.sub(r'\n+', '\n', text)  # Remove extra newlines
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove non-alphanumeric characters
    text = text.strip()  # Remove leading and trailing spaces
    return text

# Chunk text into smaller sections if it's too long
def chunk_text(text, chunk_size=512):
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to generate a response from the chatbot model
def generate_response(prompt):
    # Tokenize the input text with padding and attention_mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate output with temperature, top_k, and repetition_penalty to control output behavior
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],  # Pass the attention mask
        max_new_tokens=150,  # Generate 150 new tokens after the input tokens
        num_return_sequences=1,
        temperature=0.7,  # Controls randomness of responses (0.7 is a good balance)
        top_k=50,  # Limits the model's possible next tokens to the top 50
        repetition_penalty=1.2,  # Penalizes repetition in the model's response
    )

    # Decode the generated response to readable text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Main function for interactive chatbot
def interactive_chatbot():
    print("Welcome to the GPT-2 Chatbot!")
    print("You can start chatting now. Type 'exit' to end the conversation.")

    while True:
        # Get user input (question)
        user_input = input("\nYou: ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Preprocess user input and generate a response
        processed_input = preprocess_text(user_input)
        response = generate_response(processed_input)
        print(f"Chatbot: {response}")

# Main function to process files in the folder (for background processing)
def process_folder(folder_path):
    # Check if the path exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        return

    # List all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if it's a PDF or Word document
        if file_name.endswith('.pdf'):
            print(f"Processing PDF: {file_name}")
            try:
                text = extract_text_from_pdf(file_path)
            except Exception as e:
                print(f"Failed to process PDF {file_name}. Error: {e}")
                continue
        elif file_name.endswith('.docx'):
            print(f"Processing Word document: {file_name}")
            try:
                text = extract_text_from_word(file_path)
            except Exception as e:
                print(f"Failed to process Word document {file_name}. Error: {e}")
                continue
        else:
            # Skip non-PDF/Word files
            print(f"Skipping unsupported file: {file_name}")
            continue

        # Preprocess extracted text
        preprocessed_text = preprocess_text(text)

        # Split into smaller chunks if the text is too long
        chunks = chunk_text(preprocessed_text)

        # Generate responses for each chunk
        for chunk in chunks:
            response = generate_response(chunk)
            print(f"Response for {file_name}:\n{response}\n")

# Main function
if __name__ == "__main__":
    mode = input("Choose mode: \n1. Interactive Chatbot\n2. Process Files in Folder\nChoose: ")

    if mode == "1":
        interactive_chatbot()  # Start the interactive chatbot
    elif mode == "2":
        folder_path = input("Enter folder path to process: ")  # For dynamic folder path input
        process_folder(folder_path)  # Process files in the folder
    else:
        print("Invalid choice. Please run again and select a valid mode.")
