import { config } from "dotenv";
config(); // If you want to load .env variables

import { OpenAIEmbeddings } from "@langchain/openai";
import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { doc1, doc2, doc3 } from "./data.mjs";
import { Document } from "@langchain/core/documents";

// Provide your Neo4j connection details
const NEO4J_URL = process.env.NEO4J_URL || "bolt://localhost:7687";
const NEO4J_USERNAME = process.env.NEO4J_USERNAME || "neo4j";
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || "test1234";


// Define your embeddings model
const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY
});

async function load() {
    try {
        // Upsert documents (with embeddings) into Neo4j
        //    The community integration may expose a helper like addDocuments or similar.
        //    The method name might differ depending on the exact version:
        const documents = [new Document(doc1), new Document(doc2), new Document(doc3)];
        await Neo4jVectorStore.fromDocuments(
            documents,
            embeddings,
            {
                url: NEO4J_URL,
                username: NEO4J_USERNAME,
                password: NEO4J_PASSWORD,
                nodeLabel: "Chunk", // Optional: Customize node label
                textProperty: "info", // Optional: Customize text property
                embeddingProperty: "vector", // Optional: Customize embedding property
            }
        );

        console.log("Documents have been upserted to Neo4j.");
    } catch (err) {
        console.error("Error:", err);
    } finally {
        // 7. Clean up or close the connection if needed
        // e.g., await graph.close();
    }
}

load();