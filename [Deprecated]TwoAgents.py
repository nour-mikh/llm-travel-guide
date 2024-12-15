import asyncio
import os
import requests
import json
import streamlit as st
from langchain.tools import tool
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langchain_chroma import Chroma  # Vector database
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd

# API keys
api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
search_api_key = os.getenv("CUSTOM_SEARCH_ENGINE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

# Tool definition for searching images
@tool
def search_image(query: str) -> list:
    """
    Search for images using Google Custom Search API. Only used if the user explicitly asks for an image.

    Args:
        query (str): The search query string for the image.
    Returns:
        list: URLs of the retrieved images.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": search_api_key,
        "cx": google_cse_id,
        "q": query,
        "searchType": "image",
        "num": 1
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        items = response.json().get("items", [])
        return [item["link"] for item in items if "link" in item]
    except Exception as e:
        print(f"Error: {e}")
        return []
    
@tool
def search_vector_db(query: str) -> str:
    """
    Search the vector database for documents similar to the query.
    Args:
        query (str): The search query string to find relevant documents
    Returns:
        str: A concatenated string of the top 5 most similar document contents found in the vector database
    """
    # Initialize embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Initialize/connect to vector database
    vector_store = Chroma(
        collection_name = "embeddings",
        embedding_function = embeddings,
        persist_directory = "./vector_db",
        collection_metadata = {"hnsw:space": "cosine"}  # Use cosine similarity
    )

    print("Searching the vector database for: ", query)
    result = vector_store.similarity_search(query=query, k=2)
    result_str = "\n".join([doc.page_content for doc in result])
    return result_str

@tool
def calculate_distance(city1: str, city2: str) -> str:
    """
    Calculate the distance between two cities using data from a CSV file.

    Args:
        city1 (str): The starting city.
        city2 (str): The destination city.

    Returns:
        str: The distance between the two cities or an error message if not found.
    """
    csv_path = "./pdf_files/distances.csv"

    try:
        df = pd.read_csv(csv_path, index_col=0)
        if city1 not in df.index:
            return f"{city1} not found in the data."
        if city2 not in df.columns:
            return f"{city2} not found in the data."
        distance = df.loc[city1, city2]
        return f"The driving distance between {city1} and {city2} is {distance} km."
    
    except FileNotFoundError:
        return "The distances CSV file could not be found."
    except Exception as e:
        return f"An error occurred while processing the CSV: {str(e)}"

# Main chat class for Georges
class GeminiChatGeorges:
    def __init__(self):
        self.api_key = api_key
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            api_key=self.api_key,
            temperature=0.0
        )
        def modify_messages(messages):
            messages.insert(0, HumanMessage(content="From now on, you are Georges. A friendly Lebanese tour guide."))
            return messages
        self.agent = create_react_agent(self.llm, tools=[search_vector_db, search_image, calculate_distance], messages_modifier=modify_messages)
        self.messages = []

    def send_message(self, message: str):
        self.messages.append(HumanMessage(content=message))
        history_length = len(self.messages)
        self.messages = self.agent.invoke({"messages": self.messages})["messages"]
        return self.messages[history_length:]

# Main chat class for Georgette
class GeminiChatGeorgette:
    def __init__(self):
        self.api_key = api_key
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            api_key=self.api_key,
            temperature=0.0
        )
        def modify_messages(messages):
            messages.insert(0, HumanMessage(content="From now on, you are Georgette. Georges' wife. You give suggestions about places in Lebanon."))
            return messages
        self.agent = create_react_agent(self.llm, tools=[search_vector_db, search_image, calculate_distance], messages_modifier=modify_messages)
        self.messages = []

    def send_message(self, message: str):
        self.messages.append(HumanMessage(content=message))
        history_length = len(self.messages)
        self.messages = self.agent.invoke({"messages": self.messages})["messages"]
        return self.messages[history_length:]

async def main():
    st.title("Georges & Georgette: Lebanese Tour Guides")
    st.write("Ask Georges or Georgette about places in Lebanon!")

    if "georges" not in st.session_state:
        st.session_state.georges = GeminiChatGeorges()
    if "georgette" not in st.session_state:
        st.session_state.georgette = GeminiChatGeorgette()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        initial_message = AIMessage(content="Hi there! I'm Georges, your friendly Lebanese tour guide. How can I assist you today?")
        st.session_state.messages.append(initial_message)

    image = None

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                if "Georgette" in message.content:
                    st.markdown(f"**Georgette:** {message.content}")
                else:
                    st.markdown(f"**Georges:** {message.content}")
        elif isinstance(message, ToolMessage):
            try:
                image = json.loads(message.content)[0]
                st.image(image)
            except:
                st.markdown(message.content)

    prompt = st.chat_input("Your message")
    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        if len(st.session_state.messages) % 2 == 0:
            responses = st.session_state.georgette.send_message(prompt)
        else:
            responses = st.session_state.georges.send_message(prompt)

        st.session_state.messages.extend(responses)

        for message in responses:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    if "Georgette" in message.content:
                        st.markdown(f"**Georgette:** {message.content}")
                    else:
                        st.markdown(f"**Georges:** {message.content}")
            elif isinstance(message, ToolMessage):
                try:
                    image = json.loads(message.content)[0]
                    st.image(image)
                except:
                    st.markdown(message.content)

if __name__ == "__main__":
    asyncio.run(main())
