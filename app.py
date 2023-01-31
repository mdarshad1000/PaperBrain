# Import dependencies
from flask import Flask, request
import arxiv
from flask_cors import CORS, cross_origin
import openai
import os
import requests
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper
from langchain import OpenAI
# from azure.storage.blob import BlobServiceClient

# connect_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
# container_name = "pdfs"
# container2_name = "index"

# Create a blob service client to interact with the storage account
# blob_service_client = BlobServiceClient.from_connection_string(conn_str=connect_str)

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
        print(papers_json)

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
        print(papers_json)

        
        # print(papers_json)
        papers_list.append(papers_json)

    res = {"papers": papers_list}
    return res, 200, {'Access-Control-Allow-Origin': '*'}


@cross_origin(supports_credentials=True)
@app.route('/explain', methods=['POST'])
def explain():

    # Explain the text for Papers loaded via arXiv
    if request.method == 'POST':
        excerpt = request.json
        response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"The user is a novice reading a research paper. Explain the following text:\n{excerpt}",
        temperature=0.8,
        max_tokens=293,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        final_response = response["choices"][0]["text"].lstrip()

        print(final_response)

        return {"answer":final_response}
        
    return "<h1>This is working as well</h1>"

file_name = None
folder_path = None

@cross_origin(supports_credentials=True)
@app.route('/getpdf', methods=['GET', 'POST'])
def get_pdf():

    # Download the uploaded pdf from Firebase link
    if request.method == 'POST':

        global folder_path
        folder_path = "static/pdfs"
        
        # Get all the pdf files in the folder
        files = os.listdir(folder_path)
        
        # Remove all the files in the folder before downloading the new one
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Get firebase url of the pdf
        url = request.json["pdfURL"]
        print("This is the pdf url", url)

        global file_name
        # Get the pdf file name
        file_name = url.split("/")[-1]

        # Check if the file name has .pdf extension
        if not file_name.endswith(".pdf"):
            file_name += ".pdf"
        
        folder_path = "static/pdfs"

        # Check if the folder exists else create one
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            
        # Create the file path
        file_path = os.path.join(folder_path, file_name)

        # Download the pdf
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)

        return "PDF Downloaded!"

    return "<h1>This is working as well!</h1>"


@cross_origin(supports_credentials=True)
@app.route('/chat', methods=['GET', 'POST'])
def chat():

    # Get question from user about the uploaded pdf 
    query = request.json["message"] if request.json["message"] else ""

    global folder_path

    # Set path of indexed jsons
    global file_name
    index_path = f"static/index/{file_name}.json"

    # Get all the index.json in the folder
    files = os.listdir(folder_path)
        
    # Remove all the files in the folder before downloading the new one
    for index_file in files:
        file_path = os.path.join("static/index/", index_file)
        if os.path.isfile(file_path):
            os.remove(file_path) 


    
    # Get the path of the index
    index_path = f"static/index/{file_name}.json"
    
    print("this is the index path",index_path)

    # Loads all the data from the pdfs folder
    documents = SimpleDirectoryReader(folder_path).load_data()

    # builds an index over the documents in the data folder
    index = GPTSimpleVectorIndex(documents)

    # save the index to disk
    index.save_to_disk(index_path)

    # define the llm to be used
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

    # load from disk
    index = GPTSimpleVectorIndex.load_from_disk(index_path, llm_predictor=llm_predictor)

    # Get the response
    response = index.query(query, verbose=False, response_mode="default")
    final_answer = str(response)
    
    return {"answer":final_answer}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
