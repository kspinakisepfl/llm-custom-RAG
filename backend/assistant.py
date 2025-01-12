# Welcome to the exciting world of Retrieval-Augmented Generation (RAG) systems!
# In this exercise, you'll build a powerful RAG system step by step.
# Get ready to dive into embeddings, vector databases, and AI-powered search!

import os
from dotenv import load_dotenv
from typing import List, Tuple
import sqlite3
import numpy
print(numpy.__version__)
import faiss
import numpy as np
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function 
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools import tool
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from read_pdf import read_pdf

from flask import Flask, request, jsonify

# Let's start by setting up our environment and initializing our models
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

# Initialize SentenceTransformer and its underlying tokenizer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_db_loc():
    LOCAL = os.getenv("LOCAL", "True").lower() == "true"

    if LOCAL:
        db_path = "rag_database.sqlite"
    else:
        db_path = "rag_database_with_openai_embedding.sqlite"
    return db_path

def create_sqlite_tables(db_path: str) -> None:
    """
    Create SQLite tables for storing document chunks and their embeddings.

    This function sets up the foundation of our RAG system's database. It creates
    two tables: 'chunks' for storing text chunks and their metadata, and 'embeddings'
    for storing the vector representations of these chunks.

    Args:
        db_path (str): The file path where the SQLite database will be created or accessed.

    Returns:
        None

    Fun fact: SQLite is so reliable it's used in airplanes and smartphones!
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_content TEXT,
        source_document TEXT,
        start_page INTEGER
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id INTEGER,
        embedding BLOB,
        FOREIGN KEY (chunk_id) REFERENCES chunks (id)
    )
    ''')

    conn.commit()
    conn.close()

def chunk_document(pages: List[Document], source: str) -> List[Tuple[str, str, int]]:
    """
    Chunk the document pages, handling chunks that cross page boundaries.

    This function is like a master chef slicing a long document into bite-sized pieces.
    It ensures that each chunk is just the right size for our model to digest, while
    keeping track of where each chunk came from.

    Args:
        pages (List[Document]): List of Document objects, each representing a page.
        source (str): The source document name.

    Returns:
        List[Tuple[str, str, int]]: List of (chunk_text, source, start_page).

    """

    # initialization
    text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=200)
    result = [] # where we would like to accumulate the chunks

    # variables to keep track of chunking across pages
    previous_last_chunk = "" # stores any chunks that may have overflown from past pages
    chunk_start_page = 1 # use it to keep track of page number

    for page in pages:
        page_content = page.page_content
        page_number = page.metadata['page']
        ########################################################################
        # TODO: concatenate the current page content with the last chunk of previous page
        #       if the previous page was not exactly divisible by 500, then we 
        #       we wouldn't want to throw the leftover string away.
        #       Instead we treat it as if it's part of the next page.
        ########################################################################
        curr_chunk = previous_last_chunk + page_content
        
        ########################################################################
        # TODO: chunk this concatenated string
        # Hint: use text_splitter.split_text() method
        ########################################################################
        curr_chunk = text_splitter.split_text(curr_chunk) 
        if curr_chunk:
            previous_last_chunk = curr_chunk.pop(-1)
        if curr_chunk:
            prev_page_chunk = curr_chunk.pop(0)

        
        ########################################################################
        # TODO: add all the chunks but the last one to the result
        ########################################################################
        result.append((prev_page_chunk,source,chunk_start_page))
        chunk_start_page = page_number + 1

        curr_chunk = [(chunk,source,chunk_start_page) for chunk in curr_chunk]
        result+=(curr_chunk)



    # add the last chunk of the last page to the result
    if previous_last_chunk:
        result.append((previous_last_chunk, source, chunk_start_page))

    return result

def embed_chunks(chunks: List[str], local: bool = True) -> np.ndarray:
    """
    Embed a list of text chunks using either a local SentenceTransformer model or OpenAI's embedding model.

    This function is like a translator, converting our text chunks into a language
    that our AI models can understand - the language of vectors!

    Args:
        chunks (List[str]): The list of text chunks to be embedded.
        local (bool): If True, use the local SentenceTransformer model. If False, use OpenAI's model.

    Returns:
        np.ndarray: The embedding vectors for the chunks.

    Exercise: Try implementing the OpenAI embedding method. How does it compare to the local model?
    """
    if local:
        ########################################################################
        # TODO: Implement the local SentenceTransformer embedding method here
        # Hint: You'll need to use the model.encode() method, checkout its documentation!
        ########################################################################
        encoder_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")      # using paraphraseL3V2 as it is faster here for testing
        embeddings = encoder_model.encode(chunks)
        return embeddings

    else:
        ########################################################################
        # (Optional) TODO: Implement OpenAI embedding method here
        # Hint: You'll need to use the openai.Embedding.create() method, checkout its documentation!
        ########################################################################
        pass

def process_and_store_chunks(chunks: List[Tuple[str, str, int]], db_path: str, local: bool = True) -> None:
    """
    Process the input chunks, embed them, and store in the database.

    This function is like a librarian, taking our chunks of text, creating a special
    index for each (the embedding), and carefully storing both in our database.

    Args:
        chunks (List[Tuple[str, str, int]]): List of (chunk_text, source_document, start_page) tuples.
        db_path (str): Path to the SQLite database file.
        local (bool): Whether to use the local embedding model or OpenAI's.

    Returns:
        None

    Challenge: Can you modify this function to batch process chunks for better efficiency?
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for chunk_text, source_document, start_page in chunks:

        ########################################################################
        # TODO: Define the sql query to insert the chunk into the database.
        #       Ideally you should save information on 
        #       (1) the chunk's text, 
        #       (2) which document it comes from,
        #       (3) what page it starts at
        # Hint: the sqlite3 cursor usage is of the form 
        # cursor.execute("INSERT INTO table VALUES (?, ?, ?)", (var1, var2, var3))
        ########################################################################
        insert_chunk_sql_query = "INSERT INTO chunks (chunk_content, source_document, start_page) VALUES (?, ?, ?)"
        cursor.execute(
            insert_chunk_sql_query,
            (chunk_text, source_document, start_page) # TODO: pass the required variables to your SQL query here
        )
        chunk_id = cursor.lastrowid

        ########################################################################
        # TODO: Embed the chunk using the embed_chunks function
        ########################################################################
        embedding = embed_chunks(chunk_text,local)

        ########################################################################
        # TODO: Store the embedding in the database s.t. its unique ID is 
        #       the chunk_id you get from storing it in the database
        # Hint: You'll need to convert the embedding to bytes using the .tobytes() method
        ########################################################################
        insert_embed_sql_query = "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)"
        cursor.execute(
            insert_embed_sql_query,
            (chunk_id,embedding.tobytes()) # TODO: pass the required variables to your SQL query here
        )

    conn.commit()
    conn.close()

def create_faiss_index(db_path: str) -> faiss.Index:
    """
    Create a FAISS index from the stored embeddings in the database.

    This function is like building a high-tech library catalog. It takes all our
    stored embeddings and organizes them in a way that allows for super-fast searching!

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        faiss.Index: The created FAISS index.

    Fun fact: FAISS can handle billions of vectors, making it perfect for large-scale search systems!
    """
    # create conn and cursor to load the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    ############################################################################
    # TODO: retrieve embeddings from the database using the SELECT SQL query
    ############################################################################
    select_embed_sql_query = "SELECT embedding FROM embeddings"
    cursor.execute(select_embed_sql_query)
    embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]     # this undoes the byte-representation of the embeddings. 'row[0]' picks out the embedding from the tuple of format '(embedding,)'
    
    # close the database connection
    conn.close()
    
    ############################################################################
    # TODO: create the FAISS index using L2 distance
    # Hint: checkout the documentation on the faiss.IndexFlatL2 function
    ############################################################################
    dimension = len(embeddings[0]) # TODO: get the dimension of the embeddings
    index = faiss.IndexFlatL2(dimension) # TODO: create the L2 index
    
    ############################################################################
    # TODO: add the embeddings to the index
    # Hint: use the .add() method of the index
    ############################################################################
    index.add(np.array(embeddings).astype('float32'))

    return index

def process_pdf(file_path, db_path, local=True):

    # create a connection to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # check if document already exists in the database
    cursor.execute("SELECT id FROM chunks WHERE source_document = ?", (os.path.basename(file_path),))
    
    # close the connection
    conn.close()

    ############################################################################
    # TODO: read the pdf file, use the read_pdf function
    ############################################################################
    pages = read_pdf(file_path)     # returns list of 'Document's in Document langchain format
    source = os.path.basename(file_path)
    
    ############################################################################
    # TODO: get document chunks with the chunk_document functions
    ############################################################################
    chunks = chunk_document(pages, source)
    
    ############################################################################
    # TODO: process and store the chunks with the process_and_store_chunks function
    ############################################################################
    process_and_store_chunks(chunks,db_path,local)


def search_engine(query: str, faiss_index: faiss.Index, db_path: str, k: int = 5) -> List[Tuple[str, float, str, int]]:
    """
    Search for relevant chunks using the query and FAISS index.

    This function is the heart of our RAG system. It takes a question, finds the most
    relevant information in our database, and returns it. It's like having a super-smart
    research assistant at your fingertips!

    Args:
        query (str): The search query.
        faiss_index (faiss.Index): The FAISS index for similarity search.
        db_path (str): Path to the SQLite database file.
        k (int): Number of top results to return.

    Returns:
        List[Tuple[str, float, str, int]]: List of (chunk_content, similarity_score, source_document, start_page).

    Exercise: Can you modify this function to also return the actual similarity scores?
    """
    ############################################################################
    # Implement the search functionality
    # Hint: You'll need to 
    #       (1) embed the query
    #       (2) use faiss_index.search()
    #       (3) fetch corresponding chunks from the database
    # Note that here a query doesn't mean an SQL query but a user document
    # search query in the form of a NL string.
    ############################################################################
    # TODO: embed the query
    ############################################################################
    query_embedding = embed_chunks([query])
    
    ############################################################################
    # TODO: use faiss_index.search() to find the relevant documents
    ############################################################################
    distances, indices = faiss_index.search(query_embedding,k=k)

    # connect the database to get the results
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    ############################################################################
    # TODO: fetch the corresponding chunks from the database with an SQL query
    ############################################################################
    results = []
    for i, distance in zip(indices[0], distances[0]):
        select_chunk_sql_query = "SELECT * FROM chunks WHERE id = ?"
        cursor.execute(
            select_chunk_sql_query,
            (int(i),) # TODO: pass the required variables to your SQL query here
        )
        res = cursor.fetchone()
        if res is None:
            # Handle the case where no results are found
            chunkid, chunk_content, sourcedoc, startpage = None,None,None,None
        else:
            chunkid, chunk_content, sourcedoc, startpage = res
        results.append((chunk_content, distance, sourcedoc,startpage))
    conn.close()
    return results




# In the following code, you will implement the agent that uses the search engine to answer questions using langchain
# Some example and help can be found here: https://python.langchain.com/docs/how_to/agent_executor/

@tool
def search_tool(query: str) -> str:
    """
    Search for relevant information using the query.
    """
    db_path = get_db_loc()
    faiss_index = create_faiss_index(db_path)
    ############################################################################
    # TODO: Implement this function, you have to find a way to let the llm know 
    # which chunk comes from where so that we can add the sources in the end.
    # Hint: Use your search_engine function and return the formatted the results
    ############################################################################
    found_docs = search_engine(query=query,faiss_index=faiss_index,db_path=db_path)

    formatted_response = ["Content: "+str(doc[0])+"\nSource: "+str(doc[2])+"\nLocation: "+str(doc[3])+"\n\n" for doc in found_docs]
    # for response in formatted_response:
    #     print(response)
    return formatted_response


def parametrization(model:str,temp:float, top_p:float,max_toks:int):
    # create tools list containing search_tool as a single tool. Use Tool class from langchain
    tools = [Tool(name="Search", func=search_tool, description="Search for legal information about EPFL")]

    # load ChatOpenAI from LangChain
    llm = ChatOpenAI(temperature=temp, model=model, top_p=top_p, max_tokens=max_toks)

    ############################################################################
    # TODO: Create the prompt template in the file system_prompt.txt
    ############################################################################
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the system_prompt.txt file
    system_prompt_path = os.path.join(current_dir, 'system_prompt.txt')

    # Read the system prompt from the file
    with open(system_prompt_path, 'r') as file:
        system_prompt = file.read().strip()

    # Use ChatPromptTemplate.from_messages to create a prompt that instructs the AI
    # on how to use the search tool and format its responses
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Set up the memory
    
    ################################################################################
    # TODO: Create the agent
    ################################################################################
    # Use the RunnablePassthrough, prompt, llm, and OpenAIFunctionsAgentOutputParser
    # to create the agent you can find some infos here: 
    #   https://github.com/langchain-ai/langchain/discussions/18591

    functions = [convert_to_openai_function(t) for t in tools]
    llm_with_tools = llm.bind(functions=functions)

    agent = (
        {
            "input": lambda x: x["input"], # TODO: Implement the input format
            "chat_history": lambda x: x["chat_history"], # TODO: Implement the chat history format
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]) # TODO: Implement the agent scratchpad format
        }
        | prompt # TODO: Use the prompt
        | llm_with_tools # TODO: Use the language model with tools
        | OpenAIFunctionsAgentOutputParser() # TODO: Use the output parser
    )

    ################################################################################
    # TODO: Create the agent executor
    ################################################################################
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # TODO: Use the AgentExecutor to create the agent executor
    return agent_executor


def get_db_connection():
    db_path = get_db_loc()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return conn



import re

def replace_with_sources(response:str) -> None:
    db_path = get_db_loc()


    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    ids = re.findall(r"\[\[(\d+)\]\]", response)
    
    for id in ids:
        # 1. fetch the source and page from the database
        select_src_page_query = "SELECT source_document, start_page FROM chunks WHERE id = ?"
        cursor.execute(select_src_page_query, (id,))
        chunk_content = cursor.fetchone()
        print(chunk_content)
        ####################################################################
        # TODO: 2. replace the id with the source document 
        #          for the assistant response display
        ####################################################################
        response = response.replace(f"[[{id}]]","|Source: " + str(chunk_content[0]) + "," + "Page: " + str(chunk_content[1]))
        print(response)
    # print("Assistant:", response)
    conn.close()
    return response


# if __name__ == "__main__":
#     print("Welcome to your RAG system building adventure!")
#     LOCAL = os.getenv("LOCAL", "True").lower() == "true"

#     QUICK_DEMO = os.getenv("QUICK_DEMO", "False").lower() == "true"

#     # In your .env file, set LOCAL to False if you want to use the openai 
#     #   embedding model.
#     # Set QUICK_DEMO to False if you want to run the code on the entirety 
#     #   of the data, instead of a subset.
#     # Namely: add the following lines to your .env file
#     #   LOCAL=False
#     #   QUICK_DEMO=False

#     if LOCAL:
#         db_path = "rag_database.sqlite"
#     else:
#         db_path = "rag_database_with_openai_embedding.sqlite"

#     # Initialize the database and FAISS index
#     create_sqlite_tables(db_path)

#     # List all files in the data folder, to make sure you have the right path
#     data_folder = './data'
#     all_files = os.listdir(data_folder)

#     if QUICK_DEMO:
#         all_files = all_files[:2]

#     for file in all_files:
#         file_path = os.path.join(data_folder, file)
#         # check if file is a pdf
#         if file_path.endswith('.pdf'):
#             process_pdf(file_path, db_path,LOCAL)

#     # Create FAISS index
#     faiss_index = create_faiss_index(db_path)

#     # Run the conversation loop
#     run_agent_conversation()

