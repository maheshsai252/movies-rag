from fastapi import FastAPI
from models.pydantic_models import ChatRequest, ChatResponse
import os
from fastapi import FastAPI, HTTPException
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
import uvicorn
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

#Setup keys and prompts

OPENAI_KEY = os.getenv('openai_key')
PINECONE_API_KEY = os.getenv("pinecone_key")
PINECONE_INDEX_NAME = "movies-index"


os.environ["OPENAI_API_KEY"] = OPENAI_KEY

llm = ChatOpenAI(
    api_key=OPENAI_KEY,
    temperature=0.7,
    max_tokens=150
    )
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specializing in movies."),
    ("user", 
     """
        Role: You are a helpful movie recommendation assistant. Don't bring new movies from your knowledge

        Primary Tasks:
        Recommend given movies ordering by user query when user asked for suggestions or recommendations.
        Answer questions related to movie/movies.

        Guidelines:

        Use only the information provided in the "Movies" section.
        Always summarize the plot of recommended movies when query is recommendation.
        
        When recommending Movies:
        Sort them based on how well they match the user's question.
        Don't completely ignore any aspect of the user's question.

        If there isn't enough information to answer a movie-related question, respond with: "I don't have enough information to answer this question."
        If it is question about single movie, fetch from movies section if available else answer  "I don't have enough information to answer this question."
        Movies: {context}

        Question: {question}

        Response:
        Give movie name, plot and genres of each movie you are recommending.
     """)
])

client = OpenAI(
    api_key =  OPENAI_KEY,
)

model_embedding = SentenceTransformer("Snowflake/snowflake-arctic-embed-l")


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("movies-index")

# Initialize FastAPI app
app = FastAPI()

# Set up OpenAI LLM
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Function to refine the user's query based on the provided conversation log
def query_refiner(conversation, query):
    # Using the OpenAI GPT-3.5 instruct model, formulate a question that would be most relevant to provide an answer from a knowledge base.
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(response)
    # Extract and return the refined query from the response
    return response.choices[0].text

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Refine Quey
        query = query_refiner(conversation=request.conversation_string, query=request.query)
        
         # Fetch Context
        query_encode = model_embedding.encode(query, prompt_name="query").tolist()
        results = index.query(
            namespace="movie-desc",
            vector=query_encode,
            top_k=3,
            include_values=True,
            include_metadata=True,
        )

        print(results['matches'][0]['metadata'])

        context = ""
        for match in results['matches']:
            metadata = match['metadata']
            context += f"Title: {metadata.get('original_title', '')}\n"
            context += f"\nTagline: {metadata.get('tagline', '')}\n"
            context += f"\nOverview: {metadata.get('overview', '')}\n"
            context += f"\nGenres: {metadata.get('genre_names','')}\n\n"
            context += f"\Rating: {metadata.get('vote_average','')}\n\n"
            context += f"\Release Date: {metadata.get('release_date','')}\n\n"


        chain = prompt | llm 

        message = chain.invoke({
            "question": query,
            "context": context
            })

        return ChatResponse(response=message.content, movies_desc=context)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
