import dotenv from 'dotenv';
dotenv.config();
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { doc1, doc2, doc3 } from "./data.mjs";
import cosineSimilarity from 'cosine-similarity';
import OpenAI from 'openai';
import { v4 } from 'uuid';

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

async function generateEmbedding(text) {
    const response = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text
    });

    // The API returns an array of embeddings; we only sent 1 input, so take the first
    return response.data[0].embedding;
}

const createChunks = async (text) => {
    return splitter.splitText(text);
}

async function buildVectorStore() {
    let documents = [];

    let chunks1 = await createChunks(doc1.text);
    let chunks2 = await createChunks(doc2.text);
    let chunks3 = await createChunks(doc3.text);

    chunks1.forEach((chunk, index) => {
        documents.push({ id: v4(), text: chunk, title: doc1.title });
    });

    chunks2.forEach((chunk, index) => {
        documents.push({ id: v4(), text: chunk, title: doc2.title});
    });

    chunks3.forEach((chunk, index) => {
        documents.push({ id: v4(), text: chunk, title: doc3.title});
    });


    for (const doc of documents) {
        const embedding = await generateEmbedding(doc.text);
        vectorStore.push({
            id: doc.id,
            text: doc.text,
            embedding,
            title: doc.title
        });
    }
    console.log("Vector store built successfully.");
}


async function findRelevantDocs(query, topK = 3) {
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


async function generateAnswer(userQuery) {
    // 1. Find relevant docs
    const relevantDocs = await findRelevantDocs(userQuery, 4);

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

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    separators: ["\n\n", " "],
});

(async () => {
    // Build the vector store
    await buildVectorStore();

    // Query
    const userQuery = "physical strenght and sports?";
    const answer = await generateAnswer(userQuery);

    console.log("User Query:", userQuery);
    console.log("LLM Answer:\n", answer);
})();