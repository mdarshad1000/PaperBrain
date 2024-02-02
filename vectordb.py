from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from uuid import uuid4
from typing import List
import itertools
import pinecone
import openai
import os


PROMPT_TEMPLATE = """
You are an expert assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
Use four sentences maximum and keep the answer concise unless the user asks you to be detailed.
Be very respectful and polite.

Question: {question} 

Context: {context} 

Answer:
"""

prompt = PromptTemplate.from_template(template=PROMPT_TEMPLATE,)

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

MODEL = 'text-embedding-ada-002'

openai.api_key = OPENAI_API_KEY


# Initialize Pinecone Index
def initialize_pinecone():
    '''
    Initialize the Pinecone Index with the Given Index Name.
    '''
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
        chunk_size=1500,
        chunk_overlap=150,
    )  
    print("Split function got paper id as", paper_id)
    # load the PDF
    loader = PyPDFLoader(f"arxiv_papers/{paper_id}.pdf")

    # split the loaded PDF
    pages = loader.load()
    

    # Initialize Data structures for for data prep
    texts = [] 
    metadatas = [] 

    # Adding text chunks and preparing Metadata
    for item, page in enumerate(pages):
        texts.append(page.page_content)
        metadata = {
            'paper-id': page.metadata['source'],
            'source': page.page_content,
            'page_no':int(page.metadata['page']) + 1
        }
        record_metadata = {
            "chunk": item, "text": texts[item], **metadata
        }
        metadatas.append(record_metadata)
    
    return texts, metadatas


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))
        

def embed_and_upsert(paper_id: str, texts: List[str], metadatas: List[str]) -> str:
    # Create an embedding object
    embed = OpenAIEmbeddings(
        model=MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    # Create pinecone.Index with pool_threads=30 (limits to 30 simultaneous requests)
    index = pinecone.Index('ai-journal', pool_threads=30)
    ids = [str(uuid4()) for _ in range(len(texts))]

    # Send upsert requests in parallel
    async_results = [
        index.upsert(vectors=zip(ids_chunk, embeds_chunk, metadatas_chunk), namespace=paper_id, async_req=True)
        for ids_chunk, embeds_chunk, metadatas_chunk in zip(chunks(ids), chunks(embed.embed_documents(texts)), chunks(metadatas))
    ]
    # Wait for and retrieve responses (this raises in case of error)
    [async_result.get() for async_result in async_results]

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
        temperature=0.0,
        streaming=True,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k':4}),
        # verbose=True,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": True,
                        }
    )
    answer_w_metadata = qa(question)

    answer = answer_w_metadata['result']

    page_no = [(int(answer_w_metadata['source_documents'][i].metadata['page_no'])) for i in range(len(answer_w_metadata['source_documents']))]
    # source_text = [answer_w_metadata['source_documents'][i].metadata['text'] for i in range(len(answer_w_metadata['source_documents']))]

    return answer, page_no[:2]


def check_namespace_exists(paper_id):
    index = pinecone.Index('ai-journal') 
    index_stats_response = index.describe_index_stats()
    index_stats_response = str(index_stats_response)
    return paper_id in index_stats_response


def delete_namespace():
    index = pinecone.Index('ai-journal') 
    delete_response = index.delete(delete_all=True, namespace='abcd')


initialize_pinecone()
index = pinecone.Index('ai-journal')


# delete_response = index.delete(delete_all=True, namespace='2204.04477v1')

# textsa, metadatasa = split_pdf_into_chunks('abcd')

# embed_and_upsert(paper_id='abcd', texts=textsa, metadatas=metadatasa)
# print(split_pdf_into_chunks('POA'))
# print(len(ask_questions('what is this paper about', paper_id='1802.06593v1')))
# print('\n\n\n\n\n')

# print(ask_questions('what is this paper about', paper_id='1802.06593v1')['query'])
# print('\n\n\n\n\n')

# print(ask_questions('what is this paper about', paper_id='1802.06593v1')['result'])
# print('\n\n\n\n\n')

# print(ask_questions('what is metastable electron pairs in electron', paper_id='1802.06593v1')['source_documents'])
initialize_pinecone()