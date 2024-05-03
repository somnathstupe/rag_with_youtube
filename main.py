import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")


def transcribe_audio(youtube_url):
    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
        result = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(result)
        return texts
    except Exception as e:
        st.error(f"Error during transcription: {e}")


def main():
    st.title("Extract you tube video information")
   
    youtube_url = st.text_input("Enter YouTube video URL:")

    if st.button("Download Audio and Transcribe"):
        docs = transcribe_audio(youtube_url)
        chain = load_summarize_chain(model, chain_type="map_reduce", verbose=True)
        output = chain.run(docs)

        with st.beta_container():
            st.header("Model Output")
            st.write(output)

   
if __name__ == "__main__":
    main()
