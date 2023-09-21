# Import dependencies
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor
from langchain import OpenAI
from vectordb import embed_and_upsert, split_pdf_into_chunks, ask_questions, check_namespace_exists
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
    # Get query from user
    user_query = request.json["query"] if request.json["query"] else ""

    # Search for papers
    search_paper = arxiv.Search(
        query=user_query,
        max_results=100,
        sort_by=arxiv.SortCriterion.Relevance
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

        papers_list.append(papers_json)

    res = {"papers": papers_list}
    return res, 200, {'Access-Control-Allow-Origin': '*'}


# Sort by last updated
@cross_origin('*')
@app.route('/lastupdated', methods=['GET', 'POST'])
def lastUpdated():

    # Get query from user
    user_query = request.json["query"] if request.json["query"] else ""
    
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

        # Extract the paper ID using string slicing
        start_index = paper_url.rfind("/") + 1  # Find the index of the last "/"
        end_index = paper_url.rfind(".pdf")  # Find the index of ".pdf"

        paper_id = None
        if start_index != -1 and end_index != -1:
            paper_id = paper_url[start_index:end_index]

        flag = check_namespace_exists(paper_id=paper_id)

        if flag is True:
            print("already indexed")
            return {"paper_id": paper_id}
        
        else:
            print("not indexed")

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
    
    paper_id = request.json["paper_id"] if request.json["paper_id"] else ""
    question = request.json["question"] if request.json["question"] else ""

    answer = ask_questions(question=question, paper_id=paper_id)

    return {"answer": answer}


@cross_origin(supports_credentials=True)
@app.route('/explain', methods=['POST'])
def explain():

    # Explain the text for Papers loaded via arXiv
    if request.method == 'POST':
        # excerpt = request.json
        # print(excerpt)
        # response = openai.Completion.create(
        # model="text-davinci-002",
        # prompt=f"The user is a novice reading a research paper. Explain the following text:\n{excerpt}",
        # temperature=0.8,
        # max_tokens=293,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0
        # )
        # final_response = response["choices"][0]["text"].lstrip()

        return {"answer":'Sorry! You cannot chat with the paper right now. Please upload your own paper for now :)'}
        
    return "<h1>This is working as well</h1>"


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


@app.route('/clearpdfs', methods=['POST'])
def clear_pdfs():
    pdfs_dir = 'static/index'
    exception_file = 'i.json'

    try:
        for filename in os.listdir(pdfs_dir):
            if filename != exception_file and filename.endswith('.json'):
                file_path = os.path.join(pdfs_dir, filename)
                os.remove(file_path)

        return jsonify(message='JSONs cleared successfully'), 200
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)