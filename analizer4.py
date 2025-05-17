import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import fitz  # PyMuPDF
import re
import nltk 
from nltk.tokenize import sent_tokenize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer

nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_text(text):
    # Remove headers and footers 
    text = re.sub(r'Header text pattern', '', text)
    text = re.sub(r'Footer text pattern', '', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

pdf_path = r"D:\EL studies\term9\pdf for project\clustering_2.pdf"    #set the pdf address in your PC
text = extract_text_from_pdf(pdf_path)

# Preprocess the extracted text
cleaned_text = preprocess_text(text)

# Tokenize text into sentences
sentences = sent_tokenize(cleaned_text)

# Vectorize sentences using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)

def process_question(question):
    return [question]

question = "What is K_means algorithm?"
processed_question = process_question(question)

# Vectorize question
question_vector = vectorizer.transform(processed_question)

# Compute cosine similarity between question and sentences
cosine_similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()

# Select top 100 relevant sentences based on cosine similarity
relevant_indices = cosine_similarities.argsort()[-100:][::-1]
relevant_sentences = [sentences[i] for i in relevant_indices]

# Sentiment Analysis
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', revision='714eb0f')

def search_by_sentiment(question, sentences, sentiment_analyzer):
    question_sentiment = sentiment_analyzer(question)[0]['label']
    relevant_sentences = []
    for sentence in sentences:
        sentence_sentiment = sentiment_analyzer(sentence)[0]['label']
        if sentence_sentiment == question_sentiment:
            relevant_sentences.append(sentence)
    return relevant_sentences

# A list of sentences
relevant_sentences_by_sentiment = search_by_sentiment(question, relevant_sentences, sentiment_analyzer)

# Generate a human-like answer based on the relevant sentences
context = " ".join(relevant_sentences_by_sentiment)

# Initialize the tokenizer and text generation pipeline
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
generator = pipeline('text-generation', model='distilgpt2')

inputs = tokenizer(context, return_tensors='pt', truncation=True, max_length=1024, padding='max_length')

generated_answer = generator(context, max_length=1024, num_return_sequences=1, truncation=True, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']

print(generated_answer)
