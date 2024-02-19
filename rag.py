from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Weaviate
from langchain.schema import Document
import os, weaviate

class ChatPrivate:
    vector_store = None
    retriever = None
    chain = None
    
    def __init__(self):
        self.model = ChatOllama(base_url='http://localhost:11434', model='llama2', temperature=0)
        self.text_splitter = NLTKTextSplitter(chunk_size=1024)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use 3 sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def is_file_pdf(self, file_path):
        with open(file_path, 'rb') as file:
            return file.read(4) == b'%PDF'

    def is_file_md(self, file_path):
        with open(file_path, 'r') as file:
            first_line = file.readline()
            return True if first_line.strip().startswith('#') else False

    def ingest(self, file_path: str):

        if self.is_file_pdf(file_path):
            data = PyPDFLoader(file_path=file_path).load()
        elif self.is_file_md(file_path):
            data = UnstructuredMarkdownLoader(file_path=file_path).load()
        else:
            data = TextLoader(file_path=file_path).load()

        chunks = self.text_splitter.split_documents(data)
        chunks = filter_complex_metadata(chunks)

        client = weaviate.Client(
            url='http://localhost:8080'
        )

        vector_store = Weaviate.from_documents(
            documents=chunks, embedding=FastEmbedEmbeddings(), client=client, by_text=False
        )

        self.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 6, 'lambda_mult': 0.25}
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
