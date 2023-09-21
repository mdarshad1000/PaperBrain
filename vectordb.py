from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import pinecone
import os
from uuid import uuid4
import openai
from typing import List

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

MODEL = 'text-embedding-ada-002'

openai.api_key = OPENAI_API_KEY


# Initialize Pinecone Index
def initialize_pinecone():
    index_name = 'ai-journal'
    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT  # find next to api key in console
    )
    # check if 'openai' index already exists (only create index if not)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)
    # connect to index
    index = pinecone.Index(index_name)
    return index


def split_pdf_into_chunks(paper_id: str):
    
    # Create a splitter object
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
    )
    print("Split function got paper id as", paper_id)
    # load the PDF
    loader = PyMuPDFLoader(f"arxiv_papers/{paper_id}")

    # split the loaded PDF
    pages = loader.load_and_split(
        text_splitter=text_splitter)

    # Initialize Data structures for for data prep
    texts = [] 
    metadatas = [] 

    # Adding text chunks, creating chunk IDs and preparing Metadata
    for item in range(len(pages)):

        texts.append(pages[item].page_content)
        print(len(texts), chr(10), texts)
        metadata = {
            'paper-id': pages[item].metadata['source'],
            'source': pages[item].page_content,
        }

        record_metadatas = [{
            "chunk": j, "text": text, **metadata
        } for j, text in enumerate(texts)]

        metadatas.extend(record_metadatas)

        print(metadatas)

    return texts, metadatas, 


def embed_and_upsert(paper_id: str, texts: List[str], metadatas: List[str]):
    
    # Create an embedding object
    embed = OpenAIEmbeddings(
        model=MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    index = initialize_pinecone()
    ids = [str(uuid4()) for _ in range(len(texts))]

    embeds = embed.embed_documents(texts)
    print(embeds)

    index.upsert(vectors=zip(ids, embeds, metadatas), namespace=paper_id)

    return "Embedded the Arxiv Paper"


def ask_questions(question: str, paper_id: int):

    text_field = "source"

    # Create an embedding object
    embed = OpenAIEmbeddings(
        model=MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    # switch back to normal index for langchain
    index = pinecone.Index('ai-journal')

    vectorstore = Pinecone(
        index, embed.embed_query, text_field, namespace=paper_id
    )

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )

    return qa.run(f'question: {question}')



# index = initialize_pinecone()

# index.delete(delete_all=True, namespace="1234.pdf")

# paper_ida, textsa, metadatasa = split_pdf_into_chunks('1234.pdf')

# embed_and_upsert(paper_id=paper_ida, index=index, texts=textsa, metadatas=metadatasa)

# print(ask_questions('what is this paper about', paper_id='121.pdf'))


