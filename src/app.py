import os
import streamlit as st
import pandas as pd
import json
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI,AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, create_structured_output_runnable,LLMChain
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import tool,create_openai_tools_agent,AgentExecutor,create_tool_calling_agent,create_structured_chat_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain import hub
from dotenv import load_dotenv, find_dotenv, dotenv_values
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
 
config = dotenv_values(find_dotenv())
 
global llm, parser, format_instructions, docs, dfs, openai_api_key
 
#Azure Credentials
azure_endpoint = config['AZURE_OPENAI_ENDPOINT']
deployment_name = config['CHAT_COMPLETIONS_DEPLOYMENT_NAME']
openai_api_version = config['OPENAI_API_VERSION']
embeddings_deployment_name = config['EMBEDDINGS_DEPLOYMENT_NAME']
 
#LLM Model
def setup_llm(openai_api_key):
    """instantiates llm after users passes the open api key"""
    llm = AzureChatOpenAI(azure_endpoint=azure_endpoint,
                        deployment_name = deployment_name,
                        openai_api_key=openai_api_key,
                        openai_api_version=openai_api_version,
                        temperature=0.0,
                        verbose=True
                        )
    return llm
 
def fe_setup():
    #App configuration
    chevron_icon = Image.open("deloitte-logo.png")
    st.set_page_config(page_title="Chat with your file",page_icon=chevron_icon)
    col1, mid, col2 = st.columns([1,1,20])
    with col1:
        st.image(chevron_icon, width=60)
    with col2:
        st.header('Deloitte Tabular Question Answering Chatbot')
 

@tool("query_dataframe", return_direct=True)
def query_dataframe(user_input):
    """parses data stored as pandas dataframe containing event logs"""
    prefix = """
    If the user asks you to answer a question based on browser downloads consider the following schema:
    - Download Source: URL the file has been downloaded from.
    """
   
    agent = create_pandas_dataframe_agent(llm,
                                          st.session_state.dfs,
                                          prefix = prefix,
                                          verbose=True,
                                          )
    response = agent.invoke({
       "input":user_input,
       "chat_history":st.session_state.chat_history,
    })
    return response['output']
 

# add RAG tool retriever
def rag_tool_setup(openai_api_key):
    """Generates vectorstore and set rag tool as global variable"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    splits = text_splitter.split_text(st.session_state.docs)
    embeddings = AzureOpenAIEmbeddings(azure_endpoint=azure_endpoint,
                                        deployment = embeddings_deployment_name,
                                        openai_api_key=openai_api_key,
                                        openai_api_version=openai_api_version,
                                        )
    vectorstore = FAISS.from_texts(texts=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    rag_tool = create_retriever_tool(
            retriever,
            "text_retriever",
            "Searches and returns excerpts from the emails or chat protocols.",
            )
    return rag_tool
 
# instantiate session state variables
session_variables = ['dfs', 'docs']
for var in session_variables:
    if not var in st.session_state:
        st.session_state[var] = None
 
#Call fe_setup function
fe_setup()
starter_message = "Hello, I am your assistant. Please upload the file(s) & ask me a question regarding your file(s)"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
                    AIMessage(content=starter_message),
                ]
   
#Sidebar and data processing
with st.sidebar:
    # instantiate llm afer user passes open api key
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    # freeze tool if now key is passed
    if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
    else:
        llm = setup_llm(openai_api_key)
    # start recording chat history
    if "chat_history" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["chat_history"] = [AIMessage(content=starter_message)]
    # file uploads
    st.subheader("Your documents")
    files = st.file_uploader("Upload your csv files",accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            #Reading .csv file
            dfs = []
            docs = []
            for file in files:
                if file.name[-4:] == '.csv':
                    pdf = pd.read_csv(file)
                    pdf['loaded_file'] = file.name
                    dfs.append(pdf)
                elif file.name[-5:] == '.xlsx':
                    pdf = pd.read_excel(file)
                    pdf['loaded_file'] = file.name
                    dfs.append(pdf)
                elif file.name[-4:] == '.txt':
                    docs.append(file.read())
        # buffer list of files as session state if files are present
        if dfs:
            st.session_state['dfs'] = pd.concat(dfs)
        if docs:
            st.session_state['docs'] = " ".join([str(doc) for doc in docs])
 
# add container to browse excel file and show visuals
if st.session_state.dfs is not None:
    with st.container():
        st.write("File overview")
        bytes_dl = st.session_state['dfs']['Bytes Downloaded']
        st.dataframe(st.session_state['dfs'])
        st.line_chart(data=bytes_dl)
 
#Main Function
user_input = st.chat_input("Ask a question about your files")
if user_input is not None and user_input != "":    
   
    prompt=hub.pull("hwchase17/structured-chat-agent")
 
    tools = [query_dataframe]
    # append rag_tool if text files are present
    if st.session_state.docs is not None:
        rag_tool = rag_tool_setup(openai_api_key)
        tools.append(rag_tool)
    agent = create_structured_chat_agent(llm,tools=tools,prompt=prompt)
    agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True,handle_parsing_errors=True)
    response = agent_executor.invoke(
                            {
                                "input": user_input,
                                "chat_history": st.session_state.chat_history,
                            }
                        )
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=str(response['output'])))
   
 
#Conversation
for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)