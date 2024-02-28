import boto3
import os
import json 
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.llms import Bedrock
from llama_index.node_parser import SimpleNodeParser
from llama_index.embeddings import LangchainEmbedding
from langchain_community.embeddings import BedrockEmbeddings
from llama_index import ServiceContext, OpenAIEmbedding, PromptHelper
from llama_index import VectorStoreIndex
from llama_index import download_loader
from llama_index.postprocessor.cohere_rerank import CohereRerank
from credentials import cohere_api_key , openai_api_key
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from llama_index.llms import OpenAI as llama_openai
from model import *
from questions import *
import random
import logging
from collections import defaultdict
import shutil
import numpy as np
from llama_index.node_parser import SentenceWindowNodeParser, SimpleNodeParser



logging.basicConfig(filename='rag.log', encoding='utf-8', level=logging.INFO)

class rag_on_s3:

    def __init__(self):
        session = boto3.Session()
        self.bedrock_runtime = boto3.client("bedrock-runtime")
        self.s3 = session.client('s3')
        self.cohere_api_key = cohere_api_key
        self.S3Reader = download_loader("S3Reader")
        self.root_bucket = "demo-industry-specific"
        os.environ['OPENAI_API_KEY'] = openai_api_key
        self.model_configuration("OpenAI")


    def get_model_name(self):
        model_names = list(models_ids.keys())
        return model_names

    def get_industry_names(self):
        return list(industry_specific_bucket.keys())
        

    def model_configuration(self,model_name):
        model_id = models_ids.get(model_name)
        model_kwargs = model_parameters.get(model_name)

        if model_id == "OpenAI":

            llm = llama_openai(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
            embed_model = OpenAIEmbedding()
            
        else:
            llm = Bedrock(
                client=self.bedrock_runtime,
                model_id=model_id,
                model_kwargs=model_kwargs
            )
            # bedrock_embedding = BedrockEmbeddings(
            #     client=self.bedrock_runtime,
            #     model_id="amazon.titan-embed-text-v1",
            # )
            # embed_model = LangchainEmbedding(bedrock_embedding)
            embed_model = OpenAIEmbedding()

        self.service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        system_prompt="You are an AI assistant answering questions"
        )

        response = f"{model_name} successfully loaded"
        return response


    def get_indexes(self, industry_name):
        self.industry_name = industry_specific_bucket[industry_name]

        logging.info("getting indexes")
        folder_path = "/home/ubuntu/RAG_API/"+self.industry_name+"-index"
        logging.info(folder_path)
        if os.path.exists(folder_path):
            logging.info("folder exists")
            if os.path.isdir(folder_path):
                logging.info("removing dir")
                shutil.rmtree(folder_path, ignore_errors=True)
        
        logging.info("creting indexes")
        loader = self.S3Reader(bucket=self.industry_name)
        docs = loader.load_data()

        sentence_node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text")
        # base_node_parser = SimpleNodeParser()

        nodes = sentence_node_parser.get_nodes_from_documents(docs)
        # base_nodes = base_node_parser.get_nodes_from_documents(docs)
    

        self.service_context = ServiceContext.from_service_context(
            service_context = self.service_context,
            node_parser = nodes
            )


        index = VectorStoreIndex(
            nodes,
            service_context=self.service_context)

        index.storage_context.persist(folder_path)
        logging.info("indexing successful")
        response = "Index successfully created"
        return response


# to handle text data eg : .txt , .pdf, .docx              
    def get_docs(self, industry_name):
        self.industry_name = industry_specific_bucket[industry_name]
        folder_path = "/home/ubuntu/RAG_API/"+self.industry_name+"-index"
        if (os.path.exists(folder_path)==False):
            self.get_indexes(industry_name)

        filenames = []
        files = self.s3.list_objects_v2(Bucket=self.industry_name, Delimiter='/')
        for item in files['Contents']:
            f = item['Key']
            filenames.append(f)
        logging.info(filenames)

        response = str(len(filenames)) + " Documents sucessfully indexed"
        ques = question_list[industry_name]
        
        ques_list = random.sample(ques, 4)
        total_docs = len(filenames)

        filetypes = defaultdict()
        for fn in filenames:
            ftype = fn.split('.')[-1]
            if ftype not in filetypes:
                filetypes[ftype] = []
            filetypes[ftype].append(fn)

        logging.info(filetypes)
        filetype_response = "{total_docs} documents loaded. ".format(total_docs=total_docs)
        for key in filetypes:
            l = len(filetypes[key])
            filetype_response += "{l} {key} files ".format(l=l, key=key)

        filetypes["All"] = filenames
        logging.info(filetype_response)
        return filetype_response, ques_list, filetypes



    def open_file(self, filename):
        response = self.s3.generate_presigned_url('get_object',
                                                    Params = {'Bucket': self.industry_name,
                                                                'Key':filename},
                                                    ExpiresIn=3000)
        return response


    def query_text(self,query,index,model_name):
        cohere_rerank = CohereRerank(api_key=self.cohere_api_key, top_n=2)
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            node_postprocessors=[cohere_rerank],
            service_context=self.service_context
        )
        if model_name == "Anthropic":
            response = query_engine.query(f"\n\nHuman:{query}\n\nAssistant:")
        else:
            response = query_engine.query(query)
        
        file_names = [details['file_name'] for key, details in response.metadata.items()]
        score = [node_with_score.score for node_with_score in response.source_nodes]


        logging.info(response.metadata)
        source = set()
        if(score[0] < 0.8 and score[0] > 0.5):
            source.add(file_names[0])
        else:
            for i in range(len(score)):
                if score[i] >= 0.8 :
                    source.add(file_names[i])

        print(score)
        source = list(source)
        if len(source) == 0:
            final_source = "Not Found"
            final_response = "Sorry, I couldn't find the relevant information to the given query !!"
        else : 
            final_source = source
            final_response = response.response

        return final_response,final_source


# to handle .csv data 
    def display_csv_files(self):
        response = self.s3.list_objects_v2(Bucket="berrywise-database")
        object_keys = [obj['Key'] for obj in response['Contents']]
        return object_keys


    def get_csv_file(self,csv_file_key):
        directory_path = "/home/ubuntu/RAG_API/csv_file"
        files_in_directory = os.listdir(directory_path)
        for filename in files_in_directory: 
            file_path = os.path.join(directory_path, filename)  # Construct full path
            os.remove(file_path)
        self.s3.download_file('berrywise-database', csv_file_key, "/home/ubuntu/RAG_API/csv_file/"+csv_file_key)
        file_name= os.listdir(directory_path)
        df = pd.read_csv(f"/home/ubuntu/RAG_API/csv_file/{file_name[0]}")
        data = df.head(5)
        columns = df.columns.tolist()
        
        # logging.info(data)
        
        json_output = data.to_json(orient='records', force_ascii=False)
        parsed_json = json.loads(json_output)
        output = json.dumps(parsed_json)

            
        logging.info(output)
        print(output)
        return output, columns
        

    def csv_query(self,query):
        folder_path = "/home/ubuntu/RAG_API/csv_file"
        file_name= os.listdir(folder_path)
        df = pd.read_csv(f"/home/ubuntu/RAG_API/csv_file/{file_name[0]}")
        llm = OpenAI(api_token=openai_api_key)
        df = SmartDataframe(df, config={"llm": llm})
        result = df.chat(query)

        if(type(result) == np.integer):
            return int(result)

        if (type(result) == int or type(result) == np.float64): 
            json_output = json.dumps(result, default=str)
            return json_output

        if(type(result) == SmartDataframe):
            result = result.to_dict(orient='records')
            return result

        return result
        # if (type(result) == int or type(result) == np.float64): 
        #     json_output = json.dumps(result)
        #     return json_output
        # elif type(result) == str : 
        #     json_output = json.dumps(result)
        #     return json_output
        # else: 
        #     json_output = json.dumps(result.to_dict())
        #     # json_output = json.dumps(result)
        #     return json_output

    
        # # llm = OpenAI(api_token=openai_api_key)
        # agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
        # prompt = {
        #     "tool_names" : ["pandas"],
        #     "tools" : {
        #         "pandas" : {
        #             "df":df.to_dict()
        #         }
        #     },
        #     "input":query
        # }
        # result = agent.run(prompt)


