# Question Answering implementation
Nowadays, due to the large volume of textual data across various fields and the need to process them, NLP (Natural Language Processing) has gained significant attention.
This topic has multiple subfields, and here we focus on question answering.

In many scientific and non-scientific textual datasets, we might seek an answer to a specific question within a large amount of text. This method helps us extract our answer from an entire article or dataset using semantic analysis and search, ultimately providing a coherent, human-like response.

In the implementation, the textual file is considered in PDF format.

## steps:

**Extracting Text from a PDF:** 
The text is extracted from a PDF file using the PuMuPDF library.


**Text Preprocessing:**
A function processes the text using the re library. This includes:

-Removing headers, footers and special characters.
-Reducing multiple spaces to a single space.


**Tokenization:**
To analyze the processed text, we tokenize it. Each token can be a word, sentence, or other unit depending on the need. Here, we use a sentence tokenizer (tokenizer_sent) from the NLTK library, which splits the text into individual sentences, forming a list of sentences.
Converting Text to Numeric Values: To enable search operations, we convert each sentence into numerical values for comparison and similarity detection.
We use TfidfVectorizer (from Scikit-learn) to transform the list of strings into a two-dimensional array: Each row corresponds to a sentence. Each column represents a term defined by the library.
A numeric value is assigned to each term based on its importance in the sentence. The question is also transformed into a numeric array in the same way.


**Similarity Metrics for Search:**
Two similarity metrics are defined to find the most relevant sentence:
Cosine Similarity: Measures the similarity between two vectors by calculating the cosine angle between them.
Sentiment-Based Similarity: Uses a pre-trained model to assess sentence similarity based on semantic meaning.

Pros & Cons of both:

Cosine similarity emphasizes matching words in sentences, which might ignore semantically related sentences that don’t share exact words.
Semantic similarity measures similarity based on context, making it more effective in capturing meaning. A combination of both approaches enhances accuracy.

Model Used for Semantic Similarity: english-2-sst-finetuned-uncased-base-distilBERT.


**Finding the Most Relevant Sentences:**
In cosine similarity, the similarity score is calculated between the question and each sentence. The output is a list of scores for each sentence.
The top 100 sentences with the highest similarity are stored in sentences_relevant.

Semantic Search for Answers: The sentiment_by_search function uses the question, sentences_relevant, and the semantic similarity model to extract contextually similar sentences.

**Generating a Human-Like Response:**
Finally, to create a human-like answer, a trained model (distilGPT-2) generates the response. The maximum response length is set based on the length of the extracted text.
This method ensures accurate question-answering by using both statistical and semantic approaches.
