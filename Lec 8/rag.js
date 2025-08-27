import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import * as dotenv from 'dotenv';
dotenv.config();

// STEP1: load the PDF file
async function loadPDF() {
    const PDF_PATH = './dsa.pdf';
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();
    // console.log('rawDocs length', rawDocs.length); // 112; This means 112 pages in the pdf

    // Chunking kro 
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200 ,
      });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log('chunkedDocs', chunkedDocs);

    // Get your vector embedding model

}

loadPDF();