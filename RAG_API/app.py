from flask import Flask, request, jsonify
from rag import rag_on_s3 # Replace 'your_script_name' with the name of your Python file containing the rag_on_doc class
from llama_index import  load_index_from_storage,StorageContext
from flask_cors import CORS, cross_origin
from model import *
app = Flask(__name__)

# Initialize your rag_on_doc instance
doc_processor = rag_on_s3()


@app.route('/',methods=['GET'])
def hello():
    response = jsonify({
        "message":"Your API is working"
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# @app.route("/get_models",methods=['GET'])
# @cross_origin()
# def get_models():
#     response = doc_processor.get_model_name()
#     return jsonify({"model_names":response})

# @app.route("/select_model",methods=['POST'])
# @cross_origin()
# def select_model():
#     data = request.json
#     model = data.get('model_name')
#     response = doc_processor.model_configuration(model)
#     return jsonify({"response":response})

@app.route("/get_industry_names", methods=['GET'])
@cross_origin()
def get_industry_names():
    response = doc_processor.get_industry_names()
    return jsonify({"industry_names":response})

@app.route("/get_index", methods=['POST'])
@cross_origin()
def get_index():
    data = request.json
    industry_name = data.get('industry_name')
    response = doc_processor.get_indexes(industry_name)
    return jsonify({"response": response})

@app.route("/get_docs",methods=['POST'])
@cross_origin()
def get_docs():
    data = request.json
    industry_name = data.get('industry_name')
    total_docs, questions, file_types = doc_processor.get_docs(industry_name)
    response = jsonify({"total_docs":total_docs, "questions": questions, "file_types": file_types})
    return response

@app.route("/open_file", methods=['POST'])
@cross_origin()
def open_file():
    data = request.json
    filename = data.get('filename')
    url = doc_processor.open_file(filename)
    response = jsonify({'url':url})
    return response

@app.route("/display_csv_files", methods=['GET'])
@cross_origin()
def get_csv_file():
    csv_files = doc_processor.display_csv_files()
    return jsonify({"files":csv_files})


@app.route("/get_csv_files", methods=['POST'])
@cross_origin()
def get_csv_files():
    data = request.json
    file_name = data.get('csv_file')
    response,columns = doc_processor.get_csv_file(file_name)
    return response
    # return jsonify({"columns":columns,"response":response})

@app.route("/query_csv", methods=['POST'])
@cross_origin()
def query_csv():
    data = request.json
    query = data.get('query')
    response= doc_processor.csv_query(query)
    return jsonify({"response":response})


@app.route('/query_txt', methods=['POST'])
@cross_origin()
def process_query():
    data = request.json
    query = data.get('query')
    model_name = data.get('model_name')
    industry_name = data.get('industry_name')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400

    ind_name = industry_specific_bucket[industry_name]
    index_dir = ind_name + "-index"
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_dir))
    response, source = doc_processor.query_text(query, index,model_name) # Process the query
    return jsonify({"response": response, "source": source})


if __name__ == '__main__':
    app.run(host='172.31.25.189',debug=True)