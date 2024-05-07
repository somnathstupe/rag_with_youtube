import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import os 



load_dotenv()

parser = StrOutputParser()

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_template(template)

embeddings = OpenAIEmbeddings()



def transcribe_audio(youtube_url):
    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
        texts = loader.load()
        with open("transcription.txt", "w") as file:
            file.write(str(texts))  # Write the entire transcript
    except Exception as e:
        st.error(f"Error during transcription: {e}")


loader = TextLoader("transcription.txt")
text_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)
vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings)



chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)


def chat(ask_question):
    output = chain.invoke(ask_question)
    return output


with st.sidebar:
    st.title("Youtube")
    youtube_url = st.text_input(label="Youtube URL")
    transcribe_audio(youtube_url)
    st.button(label="load URL")


def main():
    st.title("Chat with Youtube video")
    ask_question = st.text_input("Enter you question")

    if st.button("Ask Question to video"):
        output = chat(ask_question)
        st.write(f"Answer: {output}")



if __name__ == "__main__":
    main()
