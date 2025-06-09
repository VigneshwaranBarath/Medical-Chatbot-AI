
from langchain import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.helper import load_pdf_file
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from flask import Flask, render_template, request

import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = "pcsk_3fw9Cf_82M3f1scBV4xC9AsaLm895SnPQrd21HkvFUZrFnJ8S4agseHXpwogFQ3ukPYGPW"

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Download the Embeddings from Hugging Face
embedding_model = HuggingFaceEmbeddings()



index_name = "mediapp"

print("Embedding text chunks and upserting into Pinecone index...")
vector_store = PineconeVectorStore(index_name="mediapp", embedding=embedding_model,pinecone_api_key=PINECONE_API_KEY)


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

llm=OpenAI(temperature=0)
   

# Define the prompt variable (you can customize this as needed)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
])

combine_documents_chain = create_stuff_documents_chain(llm, prompt)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port= 5000)











