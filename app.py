# Import dependencies
from flask import Flask, request
import arxiv
from flask_cors import CORS, cross_origin
import openai
import os
import requests
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor
from langchain import OpenAI
import uuid
from urllib.parse import urlparse
import shutil


# Set API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app,)


# Sort by relevance
@cross_origin('*')
@app.route('/', methods=['GET', 'POST'])
def home():
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
@app.route('/lastUpdated', methods=['GET', 'POST'])
def index():

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
@app.route('/explain', methods=['POST'])
def explain():

    # Explain the text for Papers loaded via arXiv
    if request.method == 'POST':
        # excerpt = request.json
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
        # directory_to_delete = f'static/pdfs/{f_path}'
        # try:
        #     # Use shutil.rmtree() to remove the directory and its contents
        #     shutil.rmtree(directory_to_delete)
        #     print(f"Directory '{directory_to_delete}' and its contents have been successfully deleted.")
        # except OSError as e:
        #     print(f"Error: {e}")


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
        return {"answer":final_answer}

if __name__ == '__main__':
    app.run(debug=True, port=5000)