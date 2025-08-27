//This script to be run after running "rag.js" once.

import * as dotenv from 'dotenv';
import readlineSync from 'readline-sync';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";
dotenv.config();

const ai = new GoogleGenAI({}); // No need to pass api key, it will automatically look for env variable named GEMINI_API_KEY
const History = []

/*
    This script takes user queries, rewrites them to be standalone questions, retrieves relevant document chunks from Pinecone, and generates answers using Google Gemini. It is needed to handle follow-up questions:
        User: What is a binary tree?
        Bot: <Explains binary tree>
        User: What are its types?
        Bot: <Cannot answer, because "What are its types?" does not match any context in Pinecone. It needs to be rewritten to "What are the types of binary trees?" This is done using the transformQuery function below.>
*/

async function transformQuery(question){
    History.push({
        role:'user',
        parts:[{text:question}]
    })  

    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: History,
        config: {
            systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
            Only output the rewritten question and nothing else.`,
            },
    });
    
    History.pop()
    return response.text
}

async function getAnswer(question) {
    // First, transform the question to a standalone query
    const standalone_query = await transformQuery(question);
    // console.log('Standalone Query:', standalone_query);

    // Convert the query to vector using same embedding model
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
    });
    const queryVector = await embeddings.embedQuery(standalone_query);   

    // Query Pinecone to get top 3 most similar chunks 
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    const searchResults = await pineconeIndex.query({
        topK: 10, 
        vector: queryVector,
        includeMetadata: true,
    });
    // console.log(searchResults.matches[0].metadata); // This will print the most relevant chunk from the document

    // Prepare the context to be sent to LLM
    const context = searchResults.matches
                    .map(match => match.metadata.text)
                    .join("\n\n---\n\n");

    // Now, send the query and context to LLM to get the answer
    History.push({
        role:'user',
        parts:[{text:standalone_query}]
    })  
    const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
        systemInstruction: `You have to behave like a Data Structure and Algorithm Expert.
                            You will be given a context of relevant information and a user question.
                            Your task is to answer the user's question based ONLY on the provided context.
                            If the answer is not in the context, you must say "I could not find the answer in the provided document."
                            Keep your answers clear, concise, and educational.
                            
                            Context: ${context}
                            `,
        },
    }); 
    History.push({
        role:'model',
        parts:[{text:response.text}]
    })

    console.log("\n");
    console.log(response.text);
}

async function main(){
   const userProblem = readlineSync.question("Ask me anything--> ");
   await getAnswer(userProblem);
   main();
}

main();