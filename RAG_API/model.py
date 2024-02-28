models_ids = {
    "Anthropic" : "anthropic.claude-v2",
    "Cohere":"cohere.command-text-v14",
    "Ai21Labs" : "ai21.j2-ultra-v1",
    # "Amazon Titan" : "amazon.titan-text-express-v1",
    # "Llama 2":  "meta.llama2-13b-chat-v1",
    "OpenAI":"OpenAI"
}

industry_specific_bucket = {
    "Healthcare" : "gen-ai-for-healthcare",
    "Manufacturing" : "gen-ai-for-manufacturing",
    "Lifescience" : "gen-ai-for-lifescience",
    "Public Sector" : "gen-ai-for-public-sector",
    "Automotive": "gen-ai-for-automotive"
}


model_parameters = {

     "Anthropic" : { 
                    "max_tokens_to_sample": 4096,
                    "temperature": 0.5,
                    "top_k": 250,
                    "top_p": 1,
                    "stop_sequences": ["\n\nHuman:"],
                },

    "Cohere":  { 
                    "max_tokens": 4000,
                    "temperature": 0.8,
    
                },

    "Ai21Labs" : { 
                    "maxTokens":4000,
                    "temperature":0,
                    "topP":1,
                    "countPenalty":{"scale":0},
                    "presencePenalty":{"scale":0},
                    "frequencyPenalty":{"scale":0}
                },

    # "Amazon Titan" : { 
    #                 "maxTokenCount": 8192,
    #                 "stopSequences": [],
    #                 "temperature":0,
    #                 "topP":1
    #             },

    "Llama 2":  { 
                    "max_gen_len": 512,
                    "temperature": 0.2,
                    "top_p": 0.9
                }
}
