# Import dependencies
from vectordb import embed_and_upsert, split_pdf_into_chunks, ask_questions, check_namespace_exists, initialize_pinecone
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO
from flask import Flask, request, jsonify, Response
import requests
import openai

import os
initialize_pinecone()
# Set API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app,)


# Index Arxiv Paper
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
            print("chunked", paper_id)

            # Create embeddings and upsert to Pinecone
            embed_and_upsert(paper_id=paper_id, texts=texts,
                             metadatas=metadatas)

    return {"paper_id": paper_id}


# Chat with arxiv paper
@cross_origin(supports_credentials=True)
@app.route('/explain-new', methods=['POST'])
def ask_arxiv():

    paper_id = request.json["f_path"]
    question = request.json["message"]

    answer, page_no = ask_questions(question=question, paper_id=paper_id) 
    # return Response(ask_questions(paper_id=paper_id, question=question), mimetype='text/event-stream')


    return {
        "answer": answer,
        "page_no": page_no,
        # "source_text": source_text
        }



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
    app.run(debug=True, port=5000)