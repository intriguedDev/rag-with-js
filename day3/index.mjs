import dotenv from 'dotenv';
dotenv.config();
import OpenAI from 'openai';
import { OpenAIEmbeddings } from "@langchain/openai";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";

// 1. Provide your Neo4j connection details
const NEO4J_URL = process.env.NEO4J_URL || "bolt://localhost:7687";
const NEO4J_USERNAME = process.env.NEO4J_USERNAME || "neo4j";
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || "test1234";

const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY
});

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

async function findRelevantDocs(query, topK = 3) {
    let vectorStore = await Neo4jVectorStore.initialize(embeddings, {
        url: NEO4J_URL,
        username: NEO4J_USERNAME,
        password: NEO4J_PASSWORD,
        nodeLabel: "Chunk", // Optional: Customize node label
        textProperty: "info", // Optional: Customize text property
        embeddingProperty: "vector", // Optional: Customize embedding property
    });

    const results = await vectorStore.similaritySearch(query, topK);
    vectorStore.close();
    return results;
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

(async () => {
    // Query
    const userQuery = "physical strenght and sports?";
    const answer = await generateAnswer(userQuery);

    console.log("User Query:", userQuery);
    console.log("LLM Answer:\n", answer);
})();