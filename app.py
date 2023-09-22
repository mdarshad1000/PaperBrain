# Import dependencies
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor
from langchain import OpenAI
from vectordb import embed_and_upsert, split_pdf_into_chunks, ask_questions, check_namespace_exists, initialize_pinecone
from flask_cors import CORS, cross_origin
from urllib.parse import urlparse
from flask import Flask, request, jsonify
import requests
import openai
import shutil
import arxiv
import uuid
import os

# Set API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app,)


@app.route('/check')
def check():
    return 'This is working'


# Sort by relevance
@cross_origin('*')
@app.route('/', methods=['GET', 'POST'])
def relevance():
    if request.method == 'POST':
        # Get query from user
        user_query = request.json["query"]

        # Search for papers
        search_paper = arxiv.Search(
            query=user_query,
            max_results=100,
            sort_by=arxiv.SortCriterion.Relevance
        )

        # List comprehension to store required paper details
        papers_list = [
            {
                'paper_title': result.title,
                'paper_url': result.pdf_url,
                'paper_summary': result.summary,
                'paper_authors': ", ".join([author.name for author in result.authors]),
            }
            for result in search_paper.results()
        ]

        res = {"papers": papers_list}
        return res, 200, {'Access-Control-Allow-Origin': '*'}
        
# Sort by last updated
@cross_origin('*')
@app.route('/lastupdated', methods=['GET', 'POST'])
def lastUpdated():

    if request.method == 'POST':
        # Get query from user
        user_query = request.json["query"]
        
        # Search for papers
        search_paper = arxiv.Search(
            query=user_query,
            max_results=100,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        # List to store required paper details
        papers_list = []
        
        # Iterate through search results and append paper details to list
        for result in search_paper.results():
            papers_json = {
                'paper_title':result.title,
                'paper_url':result.pdf_url,
                'paper_summary':result.summary,
                'paper_authors':", ".join([author.name for author in result.authors]),
            }
    
            # print(papers_json)
            papers_list.append(papers_json)

        res = {"papers": papers_list}
        return res, 200, {'Access-Control-Allow-Origin': '*'}


@cross_origin(supports_credentials=True)
@app.route('/indexpaper', methods=['POST'])
def index_paper():

    if request.method == 'POST':
        paper_url = request.json['paperurl'] if request.json['paperurl'] else ''

        # Extract the paper ID
        paper_id = os.path.splitext(os.path.basename(paper_url))[0]

        # Check if a namespace exists in Pinecone with this paper ID
        flag = check_namespace_exists(paper_id=paper_id)

        if flag:
            print("Already Indexed")
        
        else:
            print("Not Indexed, indexing now...")

            response = requests.get(paper_url)

            # Check if the request was successful and download the pdf
            if response.status_code == 200:
                with open(f'arxiv_papers/{paper_id}.pdf', 'wb') as f:
                    f.write(response.content)

            # Split PDF into chunks
            texts, metadatas = split_pdf_into_chunks(paper_id=paper_id)
            print("chunked",paper_id)

            # Create embeddings and upsert to Pinecone
            embed_and_upsert(paper_id=paper_id, texts=texts, metadatas=metadatas)

    return {"paper_id": paper_id}


@cross_origin(supports_credentials=True)
@app.route('/explain-new', methods=['POST'])
def ask_arxiv():
    
    if request.method == 'POST':
        
        paper_id = request.json["f_path"]
        question = request.json["message"]

        answer = ask_questions(question=question, paper_id=paper_id)

        return {"answer": answer}


@cross_origin(supports_credentials=True)
@app.route('/getpdf', methods=['GET', 'POST'])
def get_pdf():

    # Download the uploaded pdf from Firebase link
    if request.method == 'POST':
        url = request.json["pdfURL"]
        parsed_url = urlparse(url)
        pdf = os.path.basename(parsed_url.path)
        f_path = str(uuid.uuid4())   # unique identifier for each user
        response = requests.get(url)

        if not os.path.exists(f'static/pdfs/{f_path}'):
            os.makedirs(f'static/pdfs/{f_path}')

        # Check if the request was successful
        if response.status_code == 200:
            with open(f'static/pdfs/{f_path}/{pdf}.pdf', 'wb') as f:
                f.write(response.content)

        return {"f_path":f_path}
    
# @cross_origin(supports_credentials=True)
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    f_path = request.json["f_path"] if request.json["f_path"] else ""
    query = request.json["message"] if request.json["message"] else ""

    if os.path.exists(f'static/index/{f_path}.json'):
        print("Loading index loop")

        # remove the uploaded pdf once indexed
        directory_to_delete = f'static/pdfs/{f_path}'
        try:
            # Use shutil.rmtree() to remove the directory and its contents
            shutil.rmtree(directory_to_delete)
            print(f"Directory '{directory_to_delete}' and its contents have been successfully deleted.")
        except OSError as e:
            print(f"Error: {e}")


        # load from disk
        loaded_index = GPTSimpleVectorIndex.load_from_disk(f'static/index/{f_path}.json')
        response = loaded_index.query(query, verbose=True, response_mode="default")
        final_answer = str(response)
        return {"answer":final_answer}
    
    else:
        print("Creating index loop")
        # Set path of indexed jsons

        index_path = f"static/index/{f_path}.json"

        documents = SimpleDirectoryReader(f'static/pdfs/{f_path}').load_data()

        # builds an index over the documents in the data folder
        index = GPTSimpleVectorIndex(documents)

        # save the index to disk
        index.save_to_disk(index_path)

        # define the llm to be used
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

        # load from disk
        loaded_index = GPTSimpleVectorIndex.load_from_disk(index_path, llm_predictor=llm_predictor)
        response = loaded_index.query(query, verbose=True, response_mode="default")

        final_answer = str(response)
        print(f"This is the the response for your question\nQuestion: {query}\nAnswer{final_answer}")
        return {"answer":final_answer}


@app.route('/clearjsons', methods=['POST'])
def clear_pdfs():
    json_dir = 'static/index'
    arxiv_dir = 'arxiv_papers'
    exception_json = 'i.json'
    exception_paper = 'p.pdf'


    try:
        for filename in os.listdir(json_dir):
            if filename != exception_json and filename.endswith('.json'):
                file_path = os.path.join(json_dir, filename)
                os.remove(file_path)

        for filename in os.listdir(arxiv_dir):
            if filename != exception_paper and filename.endswith('.json'):
                file_path = os.path.join(arxiv_dir, filename)
                os.remove(file_path)

        return jsonify(message='JSONs and ArXiv cleared successfully'), 200
    
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route('/viewstatus', methods=['POST'])
def view_status():
    folder_path = 'arxiv_papers'
    files = os.listdir(folder_path)
    return {"filename": files}


if __name__ == '__main__':
    initialize_pinecone()
    app.run(debug=True, port=5000)