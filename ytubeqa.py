import openai
import os
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import find_dotenv, load_dotenv
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import textwrap
import streamlit as st
from streamlit_chat import message
from googletrans import Translator

load_dotenv(find_dotenv())
# Get an OpenAI API Key before continuing
OPENAI_API_KEY : os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI()

embeddings= OpenAIEmbeddings()
#User input video
video_url= st.text_input('Please enter your Youtube link here!')
translator = Translator()
#creating a database
def creating_db(video_url):
    
    loader= YoutubeLoader.from_youtube_url(video_url, language=["en", "es", "fr", "pt-br", "pt"],
    translation="en")
    transcript= loader.load()     
    
    #to breakdown the enormous amount of tokens we will get from the transcript as we have a limited set we can input
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    #this is just a list with the bunch of splits from the above
    docs= text_splitter.split_documents(transcript)
    
    db= FAISS.from_documents(docs, embeddings) #embeddings are the vectors we convert the text over into
    
    return db

#creating another function to get response from querying the above database

db= creating_db(video_url)
retriever= db.as_retriever(k=4, filter=None)
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)
    
#chaining
chain= ConversationalRetrievalChain.from_llm(llm, verbose=False, retriever=retriever, chain_type="stuff", memory=memory) 

#setting up the title
st.title("Hello, I'm `Creed` your Youtube Assistant 👓  ")
st.subheader("Please ask questions about the video you entered: Supported Video Languages - EN, FR, ES, PT")

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        
        response = chain.run(user_query)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
