from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import os

os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "853f2604-1610-4e29-9af1-50b73dc36ecf"
os.environ["PINECONE_ENV"] = "gcp-starter"

def loadText():
    loader = TextLoader("awedfirstpaper.txt")
    documents = loader.load()
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
    )


    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()


    # initialize pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("quickstart")

    index_name = "langchain-demo"

    # First, check if our index already exists. If it doesn't, we create it
    if index_name not in Pinecone.list_indexes():
        # we create a new index
        Pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
    # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
    # docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

def search():
    embeddings = OpenAIEmbeddings()

    # initialize pinecone
    Pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    index_name = "langchain-demo"
    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    query = "What is a distributed pointcut"
    docs = docsearch.similarity_search(query)

    print(docs[0].page_content)

loadText()
search()