import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from PyPDF2 import PdfReader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import openai
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import io
import pickle
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv('api_key')
if "last_sources" not in st.session_state:
    st.session_state.last_sources = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None 
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
    
if "store" not in st.session_state:
    st.session_state.store = None    
       
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)    

llm=ChatOpenAI(temperature=0, model="gpt-4",openai_api_key = openai.api_key)

def prepare_documents_for_embedding(documents):
    # Splits documents into chunks while preserving metadata
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Prepare texts and metadatas separately
    texts = []
    metadatas = []
    
    for doc in documents:
        # Split the text from each document
        text_chunks = text_splitter.split_text(doc['text'])
        
        # Create metadata for each chunk
        for chunk in text_chunks:
            texts.append(chunk)
            metadatas.append({
                'source': doc['metadata']['source'],
                'page': doc['metadata']['page']
            })
    
    return texts, metadatas

def get_pdf_text_with_metadata(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            # Create a document with both text and metadata
            doc = {
                'text': text,
                'metadata': {
                    'source': pdf.name,
                    'page': page_num + 1  # Page numbers start from 1
                }
            }
            documents.append(doc)
    return documents

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def clear_chat_history():
    st.session_state.memory.clear()
    st.session_state.last_sources=None
    st.session_state.chat_history=[]
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]
    st.success("Chat history cleared!")
embedding_model_id = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_id,
            )    
 
st.title("PDF Chatbot :robot_face:")
st.subheader("Welcome to the chat!")


with st.sidebar:

    st.title("Upload Your PDF File")
    docs=st.file_uploader("### Upload your PDF here",type=['pdf'], accept_multiple_files=True)

    but=st.sidebar.button('Submit and Process')
    if but:
        with st.spinner("Processing..."):
            original_pdfs = docs
            raw_documents = get_pdf_text_with_metadata(docs)
            
            # Prepare documents for embedding
            texts, metadatas = prepare_documents_for_embedding(raw_documents)
            
            
            # Saving Embeddings in VectorStore
            vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

            st.session_state.vector_store=vector_store
            if(vector_store is not None):
                retriever=vector_store.as_retriever()
                #memory = st.session_state.memory    
                ### Contextualize question ###
                contextualize_q_system_prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""
                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )
                qa_system_prompt = """You are an assistant for question-answering tasks. \
                Use the following pieces of retrieved context to answer the question. \
                If you don't find the information to answer the question present in the retrieved context, just output 'Sorry, I didn’t understand your question. Do you want to connect with a live agent?' and nothing else. \
                    
                {context}"""

                qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", qa_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                store = {}

                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
                st.session_state.conversational_rag_chain = conversational_rag_chain
                st.session_state.store=store
            st.sidebar.success("Done")
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
st.write("""
#### Ask your Question.
 """)
q1=st.text_area("Write your Question here.")
but1=st.button('Submit')
question=""

vector_store=st.session_state.vector_store
res=""

if but1:
    if vector_store==None:
        st.write("Please Submit the PDF First!")
    else:
        question = q1
        if(q1==""):
            st.write("Please type your question!")
        else:    
            conversational_rag_chain = st.session_state.conversational_rag_chain
            result = conversational_rag_chain.invoke(
                        {"input": question},
                        config={
                            "configurable": {"session_id": "abc123"}
                        },
                    )["answer"]
            last_sources = conversational_rag_chain.invoke(
                                {"input": question},
                                config={
                                    "configurable": {"session_id": "abc123"}
                                },
                            )["context"]
            st.session_state.last_sources=last_sources
            st.session_state.chat_history.append(list([question,result]))
            st.write(f"##### {result}")
            
        
if st.button('Show Source of the last Answer in the Document'):
    if(st.session_state.last_sources==None):
        st.write("Please submit your Question First")
    else:
        for i in range(len(st.session_state.last_sources)):    
            st.write(f'''
            Source No. {i+1}: \n
            Name of the PDF file: {st.session_state.last_sources[i].metadata['source']}
        
            Page No. in PDF: {st.session_state.last_sources[i].metadata['page']}
        
            Content: {st.session_state.last_sources[i].page_content}
        
                     ''')            
if st.button('Show Conversation History'):
        store=st.session_state.store
        if(store==None):
            st.write("Please Upload the PDF!")    
        else:
    
            st.write("### Chat History:")
            if(st.session_state.chat_history==[] or st.session_state.chat_history==None ):
                st.write("#### No Chat History to show.")
            else:    
                for entry in st.session_state.chat_history:
                    user_question = entry[0]  # Question is at index 0
                    chatbot_response = entry[1]  # Answer is at index 1
                    st.write(f'''
                        **User:** {user_question}  
                        **Chatbot:** {chatbot_response}
                    ''')


      

        

