// 1. Import necessary modules and libraries
import { OpenAI } from 'langchain/llms';
import { RetrievalQAChain } from 'langchain/chains';
import { HNSWLib } from 'langchain/vectorstores';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from 'fs';
import * as dotenv from 'dotenv';

import express from 'express'

dotenv.config();

const txtFilename = "AI_Superpowers";
//const question = "Tell me about “The Experimenter and the Finisher";
const question = "Who purchased kaixin ?";
const txtPath = `./${txtFilename}.txt`;

const VECTOR_STORE_PATH = `${txtFilename}.index`;

const app = express();
app.use(express.json());
const port = process.env.PORT || 5056;

app.post("/ask", async (req, res) => {
    const question = req.body.prompt;
    let result = await runWithEmbeddings(question);

    console.log(result.text);

    return res.status(200).json({
        success: true,
        message: result.text,
    });


})

app.listen(port, () => console.log(`Server is running on port ${port}!!`));

// 4. Define the main function runWithEmbeddings
async function runWithEmbeddings(question) {
    // 5. Initialize the OpenAI model with an empty configuration object
    const model = new OpenAI({});

    // 6. Check if the vector store file exists
    let vectorStore;
    if (fs.existsSync(VECTOR_STORE_PATH)) {
        // 6.1. If the vector store file exists, load it into memory
        console.log('Vector Exists..');
        vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
    } else {
        // 6.2. If the vector store file doesn't exist, create it
        // 6.2.1. Read the input text file
        const text = fs.readFileSync(txtPath, 'utf8');
        // 6.2.2. Create a RecursiveCharacterTextSplitter with a specified chunk size
        const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
        // 6.2.3. Split the input text into documents
        const docs = await textSplitter.createDocuments([text]);
        // 6.2.4. Create a new vector store from the documents using OpenAIEmbeddings
        vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
        // 6.2.5. Save the vector store to a file
        await vectorStore.save(VECTOR_STORE_PATH);
    }

    // 7. Create a RetrievalQAChain by passing the initialized OpenAI model and the vector store retriever
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

    // 8. Call the RetrievalQAChain with the input question, and store the result in the 'res' variable
    const res = await chain.call({
        query: question,
    });

    // 9. Log the result to the console
    //console.log({ res });
    return res;
};
