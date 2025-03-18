// index.js

require('dotenv').config();         // If you're using dotenv
const OpenAI = require('openai');
const cosineSimilarity = require('cosine-similarity');

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// 2. Sample Documents / Text Chunks
// In a real scenario, you'd parse actual files or PDFs and split them into small chunks.
const documents = [
    { id: 1, text: "JavaScript is a programming language commonly used in web development." },
    { id: 2, text: "Node.js is a runtime environment that lets you run JavaScript on the server side." },
    { id: 3, text: "Python is often used for AI tasks, data analysis, and scientific computing." },
    { id: 4, text: "React is a popular JavaScript library for building user interfaces." }
];

// 3. We'll store embeddings here (in memory) after we generate them
let vectorStore = [];

/**
 * generateEmbedding - uses OpenAI API to get embedding for a given text
 * @param {string} text
 * @returns {Promise<number[]>} embedding vector
 */
async function generateEmbedding(text) {
    const response = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text
    });

    // The API returns an array of embeddings; we only sent 1 input, so take the first
    return response.data[0].embedding;
}

/**
 * buildVectorStore - generates and stores embeddings for our documents
 */
async function buildVectorStore() {
    for (const doc of documents) {
        const embedding = await generateEmbedding(doc.text);
        vectorStore.push({
            id: doc.id,
            text: doc.text,
            embedding,
        });
    }
    console.log("Vector store built successfully.");
}

/**
 * findRelevantDocs - given a query, generate an embedding, then compute similarity to each document
 * @param {string} query
 * @param {number} topK
 */
async function findRelevantDocs(query, topK = 2) {
    // 1. Embed the query
    const queryEmbedding = await generateEmbedding(query);

    // 2. Compute similarity with each document
    const similarities = vectorStore.map((doc) => {
        const score = cosineSimilarity(queryEmbedding, doc.embedding);
        return { ...doc, similarity: score };
    });

    // 3. Sort by similarity, descending
    similarities.sort((a, b) => b.similarity - a.similarity);

    // 4. Return top K
    return similarities.slice(0, topK);
}

/**
 * Use retrieved docs to create a final answer from ChatGPT.
 */
async function generateAnswer(userQuery) {
    // 1. Find relevant docs
    const relevantDocs = await findRelevantDocs(userQuery, 2);

    // 2. Build the prompt
    const context = relevantDocs.map(doc => doc.text).join("\n\n");
    const systemPrompt = `
      You are a helpful assistant. Use the following context to answer the question:
      Context:
      ${context}
    `;

    // 3. Call OpenAI Chat API
    const response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userQuery },
        ]
    });

    return response.choices[0].message.content.trim();
}

(async () => {
    // Build the vector store
    await buildVectorStore();

    // Query
    const userQuery = "How can I run JavaScript outside a browser?";
    const answer = await generateAnswer(userQuery);

    console.log("User Query:", userQuery);
    console.log("LLM Answer:\n", answer);
})();