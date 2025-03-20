Day 2: Preparing and Indexing Your Documents
--------------------------------------------

**Objective:**

1.  **Preprocess** and **chunk** your documents.
2.  **Create** embeddings for each chunk.
3.  **Store** these embeddings in a vector store (or an in-memory data structure, if you're still prototyping).

* * * * *

### 1\. Document Preprocessing & Chunking

**Why Chunk Documents?**

-   Large documents---like lengthy PDFs, articles, or user manuals---tend to dilute relevance if treated as a single text block.
-   Splitting (chunking) them into smaller sections (e.g., 200--500 tokens) makes retrieval more precise.
-   During retrieval, the system can grab the *specific* chunk relevant to the user's query instead of retrieving an entire 10,000-word document.

**Steps to Chunk Your Documents**

1.  **Acquire Documents**
    -   Pick a small sample set: these might be PDFs, Word docs, or any text files relevant to your domain.
2.  **Convert to Plain Text**
    -   If you have PDFs or other formats, convert them to text using a tool or library (e.g., [pdfplumber (Python)](https://pypi.org/project/pdfplumber/), or a local script in Node.js).
    -   Make sure to handle or remove extraneous formatting, such as HTML tags.
3.  **Split into Chunks**
    -   Decide on a chunk size. A typical rule of thumb: 200--500 words (or tokens) per chunk.

    -   If using a library like LangChain's text splitter (in Python), you could do something like:

        ```from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(your_long_text)```

    -   Or manually split by paragraphs, bullet points, or a fixed number of words.

**Tips & Best Practices**

-   **Overlap**: Sometimes having an overlap of ~50--100 tokens between chunks helps keep context intact.
-   **Metadata**: Store extra info with each chunk, like the source filename, page number, or section header. This helps you identify the chunk's origin later on.

* * * * *

### 2\. Generating Embeddings

**Embedding Basics Recap**

-   Embeddings transform each text chunk into a vector that captures its semantic meaning.
-   Tools you can use:
    -   **OpenAI Embeddings** (`text-embedding-ada-002`).
    -   **Sentence-BERT** or other local embedding models (if you prefer open-source).

**Implementation Outline**

1.  **Load your Chunks**
    -   Suppose you have a list of text chunks: `chunks = [chunk1, chunk2, ...]`.
2.  **Embed Each Chunk**
    -   Using **OpenAI** (Node.js example):

        ```const response = await openai.embeddings.create({
          model: "text-embedding-ada-002",
          input: chunkText,
        });
        const embedding = response.data.data[0].embedding;```
3.  **Store**
    -   Save each embedding along with the chunk text and metadata in a structure ready for indexing.

**Batching Tip**

-   If you have many chunks, it's often more efficient (and cheaper) to embed them in batches (e.g., up to a certain batch size) rather than one by one.

* * * * *

### 3\. Building a Vector Store

**What is a Vector Store?**

-   A specialized database or indexing solution designed for storing embeddings and performing similarity search.
-   Examples: Pinecone, Weaviate, Milvus, Qdrant, FAISS (in-memory or local file).
-   If you're just **prototyping**, you could store your embeddings in a simple array or an in-memory structure, then do a brute-force similarity search.

#### Option A: Quick & Dirty (In-Memory)

-   Store an array of objects: `[{ id, chunkText, metadata, embedding }]`.
-   Perform similarity search by computing cosine similarity between the query embedding and each stored embedding.
-   **Pros**: Easiest to set up.
-   **Cons**: Slow for larger datasets.

#### Option B: Dedicated Vector Database

-   Pinecone, Weaviate, Qdrant, etc.
-   You embed each chunk once, "upsert" into the vector database, then query for top-k matches.
-   **Pros**: Scalable, quick, built-in approximate nearest neighbor (ANN) search.
-   **Cons**: Requires hosting or service setup.

* * * * *

### 4\. Hands-On Example (Node.js + Pinecone or In-Memory)

Below is a very short outline you can follow to code your pipeline for Day 2:

1.  **Chunk Documents**

    ```// Pseudocode for chunking in JS
    function chunkText(text, chunkSize = 400, overlap = 50) {
      // We'll do a naive approach: just slice by characters or words
      const chunks = [];
      let startIndex = 0;
      while (startIndex < text.length) {
        const endIndex = Math.min(startIndex + chunkSize, text.length);
        const chunk = text.slice(startIndex, endIndex);
        chunks.push(chunk);
        startIndex += (chunkSize - overlap); // step forward by chunkSize - overlap
      }
      return chunks;
    }```

2.  **Generate Embeddings** (OpenAI v4 syntax):

    js

    CopyEdit

    `const response = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: chunk,
    });
    const embedding = response.data.data[0].embedding;`

3.  **Store Them**

    -   **In-memory** array:
        ```vectorStore.push({
          id: `doc${i}-chunk${j}`,
          text: chunk,
          embedding: embedding
        });```

    -   **Pinecone** (or Weaviate), do an "upsert."

* * * * *

### 5\. Validate by Testing Similarity

Even on Day 2, do a quick test:

1.  **Choose a sample query** that you believe should match one of your chunk's contents.
2.  **Embed** that query.
3.  **Compute** the similarity with each chunk embedding in your store (cosine similarity).
4.  **Check** if the top match is indeed the chunk you expected.

This is your first rough check that your pipeline is working before hooking up a language model.

* * * * *

### Day 2 Wrap-Up

By the end of Day 2, you should have:

1.  **Document Chunks**: Each with an ID, text snippet, and possibly metadata.
2.  **Embeddings**: Generated for each chunk (and stored in an array or vector DB).
3.  **Basic Similarity Query**: A quick test or function to retrieve the top-k relevant chunks for a given query.

**Day 3** will focus on building a retrieval pipeline around these embeddings---essentially writing the code that, given a user query, fetches the top relevant chunks (and eventually feeds them to an LLM).