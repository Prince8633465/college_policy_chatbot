import os
import logging
from pathlib import Path
from typing import List

import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
PERSIST_DIR = "./storage"
MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF/resolve/main/llama-2-7b-chat.Q3_K_L.gguf"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Set the default embedding model to local
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

# Default system instruction
SYSTEM_INSTRUCTION = "You are a helpful assistant for Lambton College, focused on answering queries about its policies and information accurately and professionally."

def load_documents(directory: str) -> List:
    return SimpleDirectoryReader(directory).load_data()

@st.cache_resource
def initialize_llm() -> LlamaCPP:
    return LlamaCPP(
        model_url=MODEL_URL,
        model_path=None,
        temperature=0.1,
        max_new_tokens=1024,
        context_window=3900,
        generate_kwargs={"repeat_penalty": 1.2, "top_p": 0.95},
        model_kwargs={"n_gpu_layers": -1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

@st.cache_data
def create_or_load_index(
    persist_dir: str,
    _documents: List
) -> VectorStoreIndex:
    if os.path.exists(persist_dir):
        logger.info("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        logger.info("Index loaded successfully.")
    else:
        logger.info("Creating new index...")
        storage_context = StorageContext.from_defaults()
        node_parser = SimpleNodeParser.from_defaults()
        nodes = node_parser.get_nodes_from_documents(_documents)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
        )
        index.storage_context.persist(persist_dir=persist_dir)
        logger.info("New index created and persisted.")
    return index

@st.cache_resource
def create_chat_engine(
    _index: VectorStoreIndex,
    _llm: LlamaCPP,
) -> SimpleChatEngine:
    retriever = VectorIndexRetriever(
        index=_index,
        similarity_top_k=3,
    )

    return SimpleChatEngine.from_defaults(
        retriever=retriever,
        llm=_llm,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        streaming=True,
        system_prompt=SYSTEM_INSTRUCTION
    )

class StreamHandler:
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def main():
    st.set_page_config(page_title="Lambton College Policy Chatbot", page_icon="🏫", layout="wide")
    st.title("Lambton College Policy Chatbot")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_engine" not in st.session_state:
        documents = load_documents(DATA_DIR)
        llm = initialize_llm()
        index = create_or_load_index(PERSIST_DIR, documents)
        st.session_state.chat_engine = create_chat_engine(index, llm)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about Lambton College policies:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # Show loading message
            message_placeholder.markdown("Thinking...")

            stream_handler = StreamHandler(message_placeholder, initial_text="")
            try:
                response = st.session_state.chat_engine.stream_chat(prompt)
                full_response = ""
                for text in response.response_gen:
                    full_response += text
                    stream_handler.on_llm_new_token(text)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                message_placeholder.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
