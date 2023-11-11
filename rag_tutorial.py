# from https://python.langchain.com/docs/integrations/vectorstores/pinecone
# and https://python.langchain.com/docs/use_cases/question_answering/
# adapted to use Vertex AI embeddings

import os
import getpass

os.environ["PINECONE_API_KEY"] = getpass.getpass("Pinecone API Key:")
os.environ["PINECONE_ENV"] = getpass.getpass("Pinecone Environment:")

from langchain.embeddings import VertexAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader


# Load documents

from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

# Split documents

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
splits = text_splitter.split_documents(loader.load())

embeddings = VertexAIEmbeddings()
import pinecone

# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

index_name = "rag-demo"

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
      name=index_name,
      metric='cosine',
      dimension=768  
)

#vectorstore = Pinecone(index_name, embeddings.embed_query, splits)
print("past index creation for index ", index_name)
# Vertex AI embedding model  uses 768 dimensions`
vectorstore = Pinecone.from_documents(splits, embeddings, index_name=index_name)

retriever = vectorstore.as_retriever()

# Prompt 
# https://smith.langchain.com/hub/rlm/rag-prompt

from langchain import hub
rag_prompt = hub.pull("rlm/rag-prompt")

print("past index rag_prompt creation ")
from langchain.prompts import ChatPromptTemplate
from langchain.llms import VertexAI
llm = VertexAI()
 

# RAG chain 

from langchain.schema.runnable import RunnablePassthrough
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | rag_prompt 
    | llm 
)

rci_output = rag_chain.invoke("What is Task Decomposition?")
print("rci_output is: ",rci_output)



