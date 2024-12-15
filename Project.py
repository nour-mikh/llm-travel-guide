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

    # Debug print
    print("Searching the vector database for: ", query)
    
    # Perform similarity search and get top k results
    result = vector_store.similarity_search(query = query, k = 8)
    # Combine all document contents into single string
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
    csv_path = "./pdf_files/distances.csv"  # Path to the CSV file

    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path, index_col=0)

        # The model passes the city names in lower case. They are in upper case in the csv file
        city1 = city1.upper()
        city2 = city2.upper()

        print(city1 + " " + city2)
        
        # Check if both cities are in the data
        if city1 not in df.index:
            return f"{city1} not found in the data."
        if city2 not in df.columns:
            return f"{city2} not found in the data."
        
        # Retrieve the distance
        distance = df.loc[city1, city2]
        return f"The driving distance between {city1} and {city2} is {distance} km."
    
    except FileNotFoundError:
        return "The distances CSV file could not be found."
    except Exception as e:
        return f"An error occurred while processing the CSV: {str(e)}"

# Main chat class that uses Gemini LLM
class GeminiChat:
    def __init__(self, model_name: str = "gemini-pro", temperature: float = 0.0):
        """
        Initialize GeminiChat with a language model.

        Args:
            model_name (str): The model to use. Default is "gemini-pro".
            temperature (float): The temperature to use. Default is 0.0.
        """
        self.api_key = api_key
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.api_key,
            temperature=temperature
        )

        def modify_messages(messages):
            messages.insert(0, HumanMessage(content="From now on, you are Georges. This is your name and you have to respond to it. You have been trained by Google but also Nour, Elie-Joe and Salim. Your data is only available through the pdfs produced by the Ministry of tourism, not more."
                                             """
                                             You are a very cute and friendly guide, you are always super nice and give your opinion on places. You greet people in a fun way every time.
                                             If the answer/city is not found in your database search tools using serach_vector_db, tell the user something similar to 'I have not visited
                                                 this place but maybe you can call the Lebanese Ministry of tourism on '+961 01-340-940' and ask them to take me there? Please? Maybe they'd finance my trip!' 
                                                 
                                             
                                             Answer touristic questions about Lebanon only. You can give your opinion on the places discussed.
                                             When asked about areas in the south of Lebanon, mention that, due to the war, the user might see some destruction.
                                               Use the search_image tool to show images of the places only if asked to. Don't ask additional questions just show the image by invoking the tool. 
                                               If the user does not ask for an image, DO NOT USE THE SEARCH TOOL.
                                               Always try to use the search_vector_db tool to find information about the location.
                                               
                                                YOU HAVE TO USE THE TOOLS AT ALMOST ALL TIMES.
                                                Use the vector database to plan trips in Lebanon: find interesting things from the vector database to do there then use that info to create the day trips by listing them. Please call the search_vector_db to find monuments in the city.
                                               Use the calculate_disatnce tool to calculate the driving distance between two cities in Lebanon. Make sure you use the correct spelling found in the csv and modify the words if needed.
                                                Specifiy that this is the driving distance and that, technically, those cities might be closer.
                                               You have to use your search_vector_db to answer questions.  
                                                 """))
            return messages

        self.agent = create_react_agent(self.llm, tools=[search_vector_db, search_image, calculate_distance], messages_modifier=modify_messages)
        self.messages = []

    def send_message(self, message: str) -> str:
        """
        Send a message and get response from the model.

        Args:
            message (str): The message to send.

        Returns:
            str: The model's response content.
        """
        self.messages.append(HumanMessage(content=message))
        history_length = len(self.messages)
        self.messages = self.agent.invoke({"messages": self.messages})["messages"]
        new_messages = self.messages[history_length:]
        return new_messages
      

async def main(): 
    st.title("Georges the Lebanese tour guide")
    st.write("Ask me about places in Lebanon, images or distances!")

    if "llm" not in st.session_state:
        st.session_state.llm = GeminiChat(temperature=0.5)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    image = None

    for message in st.session_state.messages[1:]:
        if isinstance(message, ToolMessage):
                print("Tool message")
                print(message)
                try:
                    image = json.loads(message.content)[0]
                    print(image + " is the image")
                except:
                    image = None
                    print("No image")
        if isinstance(message, AIMessage) and message.content:
            with st.chat_message("assistant"):
                if image:
                    st.markdown("Here's what you asked for: ")
                    st.image(image)
                else: 
                    st.markdown(message.content)

        elif isinstance(message, ToolMessage):
            with st.chat_message("assistant"):
                with st.status("Tool result: "):
                    st.markdown(message.content)
        
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    prompt = st.chat_input("Your message")

    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        messages = st.session_state.llm.send_message(prompt)
        st.session_state.messages.extend(messages)

        image = None

        for message in messages[1:]:
            if isinstance(message, ToolMessage):
                print("Tool message")
                print(message)
                try:
                    image = json.loads(message.content)[0]
                    print(image + " is the image")
                except:
                    image = None
                    print("No image")
            if isinstance(message, AIMessage) and message.content:
                with st.chat_message("assistant"):
                    if image:
                        st.markdown("Here's an image ")
                        st.image(image)
                    else: 
                        st.markdown(message.content)

            elif isinstance(message, ToolMessage):
                with st.chat_message("assistant"):
                    with st.status("Tool result: "):
                        st.markdown(message.content)

asyncio.run(main())

