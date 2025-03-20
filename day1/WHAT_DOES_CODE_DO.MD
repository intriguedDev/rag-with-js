### Explanation

1) generateEmbedding - Uses openai library to generate an embedding from text
2) buildVectorStore - Uses generateEmbedding to create vectors and store it in in-memory vector store
3) findRelevantDocs - Similarity search (Cosine similarity)
4) generateAnswer - Does the following
    - Takes user query and generates an embedding
    - uses the embedded query to search similar date in vector store
    - picks top k matches
    - Sends the matched docs and user query to LLM to finaly select the most appropriate answer and respond
