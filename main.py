
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


from dotenv import load_dotenv
load_dotenv()

# Load the webpage
from langchain_community.document_loaders import PyPDFLoader
files = input("enter the path of the pdf eg: C:/Users/mgsuk/Downloads/usmanlom.pdf: ")
file_path = (
    files
)
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()



# Create and persist a Chroma vector store
vectorstore = Chroma.from_documents(documents = pages, embedding=OpenAIEmbeddings())

# Create a retriever
retriever = vectorstore.as_retriever()

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="mixtral-8x7b-32768",  # llama3-70b-8192
    temperature=0,
)


# Create a custom prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)


# Function to ask questions
def ask_question(question):
    result = qa_chain({"query": question})
    return result["result"]


# Example usage
question = input("Enter your question: ")
print(ask_question(question))
