// This script loads a PDF, chunks its content, generates embeddings using Google Gemini, and uploads them to Pinecone for vector search.
// It needs to be just run ONCE to set up the vector database with document embeddings.  
//After running this, user can then ask queries in the "query.js" file.

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import * as dotenv from 'dotenv';
dotenv.config();

async function loadPDF() {
    // load the PDF file
    const PDF_PATH = './dsa.pdf';
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();
    // console.log('rawDocs length', rawDocs.length); // 112; This me ans 112 pages in the pdf
    console.log('PDF loaded');
    
    
    // Chunking kro 
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200 ,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log('chunkedDocs length', chunkedDocs.length); // 227; So after chunking, we have 227 chunks  
    console.log('PDF chunked');
    
    
    // Get your vector embedding model
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
    });
    console.log('Embeddings model loaded');

    // Initialize Pinecone Client to connect to your Pinecone DB  
    const pinecone = new Pinecone(); //This will look for env variables automatically, no need to pass api key and env explicitly
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    // Now, LangChain does the dirty work: it takes the chunked documents, generates embeddings for each chunk, and uploads them to Pinecone 
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
        pineconeIndex,
        maxConcurrency: 5, // how many parallel requests to make to Pinecone (So here, 5 vectors would be pushed to Pinecone at once). In free tier of Pinecone , max limit is 5
    });
    console.log('All chunks embedded and uploaded to Pinecone');
}

loadPDF();