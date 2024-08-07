#import necessary libraries
import streamlit as st
import tempfile
import logging
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
#from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.agents.agent_toolkits import (VectorStoreToolkit,VectorStoreInfo)
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
#from langchain.callbacks import StreamlitCallbackHandler

from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from APIKEY import GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
#==================----------------

#Read and divide pdf data
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(file_path=tmp_file_path)
    pages = loader.load_and_split() 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    os.unlink(tmp_file_path)  # Delete the temporary file
    return chunks

#store pdf data in chromadb
def process_and_store_paragraphs(chunks):
    google_ef = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v1")
    store = Chroma.from_documents(documents=chunks,
                                embedding=google_ef,
                                collection_name="report")
    return store, google_ef

#get context from chromadb for query
def get_context(store, google_ef, query):
    context = ""
    query_emb = google_ef.embed_query(query)
    result = store.similarity_search_by_vector_with_relevance_scores(query_emb, k=5)
    for doc, score in result:
        context += doc.page_content + "\n\n"
    print(context)
    return context

#define tool for summarization
def summarize_tool(toolkit, llm, chunks):
    summarize_template = """You are a helpful AI assistant. Your task is to summarize the given text based on the following points:
                        1. What is the document about?
                        2. What does it contain?

                        Please provide a concise summary and also explain the second point briefly and its key terms in the final output .

                        Text to summarize:
                        {chunks}

                        Summary:
                        """
    
    summarize_prompt = PromptTemplate(
        input_variables=["chunks"],
        template=summarize_template
    )

    summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

    def summarize_with_chain(query):
        return summarize_chain.run(chunks)

    summarize_tool = Tool(
            name="Summarize Document",
            func=summarize_with_chain,
            description="Use this tool when asked to summarize the document or provide an overview else use other tools.")
    
    vector_store_tools = toolkit.get_tools()
    tools = vector_store_tools + [summarize_tool]

    return tools

#get LLM response for query
def get_llm_response(query, store, memory, context, chunks):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.7,
        top_p=0.85,
        top_k=40,
        max_output_tokens=800,
        n=1,
        max_retries=5,
        timeout=0.30,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH : HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE ,
            HarmCategory.HARM_CATEGORY_HARASSMENT : HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE}
        )
    

    st_callback = StreamlitCallbackHandler(st.container())

    #define tool for financial queries
    if store is not None:
        vectorstoreinfo = VectorStoreInfo(name="financial_analysis",
                                    description="Comprehensive financial report analysis tool for banking and corporate finance",
                                    vectorstore=store)
        
        toolkit = VectorStoreToolkit(vectorstore_info=vectorstoreinfo,llm=llm)

        prefix = f"""You are a helpful financial analyst AI assistant. Your task is to answer questions based on the given context, which is derived from an uploaded document. If asked to explain, elaborate the terms in your final answer.
                Always use the information provided in the context to answer the query. This context represents the content of the uploaded document.

                When answering queries:
                1. Provide accurate and relevant information from the uploaded document.
                2. Use financial terminology appropriately.
                3. If asked for calculations or comparisons, double-check your math.
                4. If the information is not in the uploaded document, clearly state that.
                5. Offer concise but comprehensive answers, and ask if the user needs more details.
                6. If applicable, mention any important caveats or contexts for the financial data.
                7. While explaining terms, explain them in short way to minimize number of tokens.

                Always base your answers primarily on the information provided in the uploaded financial document. You may use your own knowledge to supplement or clarify information, but make it clear when you're doing so.

                Context:
                {context}

                Assistant: I understand you're asking about the uploaded document. Let me answer based on the information provided in the context, which represents the content of that document."""

        
        tools = summarize_tool(toolkit, llm, chunks)

    #define tool for regular conversation 
    else:
        tools = []
        prefix = """You are a helpful AI assistant. Your task is to answer questions and engage in general conversation.
                When answering queries:
                1. Provide accurate and relevant information.
                2. If you don't know something, clearly state that.
                3. Offer concise but comprehensive answers, and ask if the user needs more details.
                4. Be polite and engaging in your responses.
                5. Remember information provided by the user, such as their name or preferences.

                Assistant: Hello! How can I assist you today?"""

    #initialize the agent for LLM
    agent_executor = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                    verbose=True,
                    memory=memory,
                    agent_kwargs={"prefix": prefix})

    #handle exception of quota exceeded
    try:
        #get query answer
        response = agent_executor.run(query, callbacks=[st_callback])

        if response:
            print("TOTAL_CONV_MEMORY: ",memory.chat_memory)
            print("\n\n")
            return response

    except google_exceptions.ResourceExhausted as e:
        st.error("API quota exceeded. Please try again later")
        logging.error(f"ResourceExhausted error: {e}")

#======================================================================

#initialize states
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'store' not in st.session_state:
    st.session_state.store = None
if 'google_ef' not in st.session_state:
    st.session_state.google_ef = None
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=10, return_messages=True, memory_key="chat_history")
if "messages" not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': """Hello! 👋  Welcome to Your AI Financial Assistant! I'm here to help you analyze and understand your financial documents."""}]

#======================================================================

#Title
st.title("Chattergy")
# Query input and processing
query = st.chat_input("Enter your query")

#=======================================================================

#Get file from user
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False, label_visibility='collapsed')

# Process uploaded file
if uploaded_file is not None and st.session_state.store is None:
    with st.spinner(f"Processing {uploaded_file.name}..."):
        st.session_state.chunks = process_pdf(uploaded_file)
        st.session_state.store, st.session_state.google_ef = process_and_store_paragraphs(st.session_state.chunks)
    st.success("File processed successfully!")

#=======================================================================

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#=======================================================================

if query:
    #append query to the message session
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    #Get context from PDF
    if st.session_state.store is not None:
        context = get_context(st.session_state.store, st.session_state.google_ef, query)
    else:
        context = None
    
    #Get response from LLM
    response = get_llm_response(query, st.session_state.store, st.session_state.memory, context, st.session_state.chunks)
    
    #Write LLM response on APP
    with st.chat_message("assistant"):
        st.write(response)
    
    #ADD response to the messages session
    st.session_state.messages.append({"role": "assistant", "content": response})
    


#remove all the sessions to clear memory
#if st.button("Clear and Upload New File"):
#    st.session_state.store = None
#    st.session_state.google_ef = None
#    st.session_state.chunks = None
#    st.session_state.memory = ConversationBufferWindowMemory(k=10, return_messages=True, memory_key="chat_history")
#    st.session_state.messages = []
#    st.experimental_rerun()