import streamlit as st
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI

from langchain.embeddings import OpenAIEmbeddings


# for tracing
load_dotenv()


st.title("Eskwelabs Chatbot")
st.write(st.session_state)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_DATA_PATH = 'embeddings_deployment_sentencetransformer'
COLLECTION_NAME = 'embeddings_deployment_sentencetransformer'


persistent_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = persistent_client.get_collection(COLLECTION_NAME)

vectordb = Chroma(client=persistent_client, collection_name=COLLECTION_NAME,embedding_function=embedding_function)

llm = ChatOllama(
    model = "llama3.1",
    temperature = 0.1,
    num_predict = 350,
    verbose=True
)

prompt = ChatPromptTemplate(
    messages=[
        MessagesPlaceholder(variable_name='chat_history'),
        ('system', "You're a helpful assistant who provides concise, complete answers without getting cut off mid-statement. Stick strictly to the user's questions, avoiding any unnecessary details."),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name="agent_scratchpad")       
                                                                    
    ]
)



def resume_retriever_tool(url):
    loader = PyPDFLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)

    chunks = splitter.split_documents(docs)

    vector_store = Chroma.from_texts(
        texts=[chunk.page_content for chunk in chunks],
        embedding=embedding_function,
        metadatas=[{'use_case': 'resume_info'}] * len(chunks)  # Ensure metadatas is a list with the same length as documents
    )



    resume_retriever = vector_store.as_retriever(
    search_kwargs={
                   "filter": {"use_case": {"$eq": "resume_info"}}
                   }      # filter according to knowledgebase
    )

    resume_retriever_tool = create_retriever_tool(
        retriever=resume_retriever, 
        name='resume_search', 
        description='''Use this tool to parse the user's resume for details such as name, contact information, educational background, skillset (soft and technical skills), and job experiences. This tool can be used together with either of the two tools, but not both at the same time.
        First, it can be used in conjunction with the eskwelabs_bootcamp_info_search tool for queries related to an applicant's qualifications. For example, it can help assess if the user's skills and educational background are sufficient for specific programs like the Data Science Fellowship (DSF) or Data Analytics Bootcamp (DAB).
        Second, it can be used with the bootcamp_vs_alternatives_search tool to determine whether a user's skills and qualifications make a bootcamp or an alternative learning path a better option for them.
        '''
    )

    return resume_retriever_tool


def create_db_retriever_tools(vectordb):
    retriever_eskwelabs = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
                   "score_threshold": 0.5, 
                   "filter": {"use_case": {"$eq": "eskwelabs_faqs"}}
                   }      # filter according to knowledgebase
    )

    eskwelabs_bootcamp_info_search_tool = create_retriever_tool(
        retriever=retriever_eskwelabs,
        name="eskwelabs_bootcamp_info_search",
        description='''Use this tool to retrieve comprehensive information about Eskwelabs, specifically its Data Analytics Bootcamp (DAB) and Data Science Fellowship (DSF). 
        This tool can answer queries about Eskwelabs' tuition fees, equipment requirements, program duration, curriculum, enrollment processes, scholarship offers, and other specific details. 
        **Avoid using this tool for questions unrelated to Eskwelabs, such as general educational comparisons (e.g., comparing bootcamps with other learning methods), unrelated topics, or information about public figures or current events.**
        This tool is particularly useful when specific details about Eskwelabs are needed.
        Additionally, it can be used in conjunction with the resume_search tool for queries that assess an applicant's readiness or suitability for Eskwelabs programs based on their resume skills.
        '''
    )

    retriever_bootcamp_vs_alternatives = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
                   "score_threshold":  0.5, 
                   "filter": {"use_case": {"$eq": "bootcamp_vs_selfstudy"}}
                   }      # filter according to knowledgebase
    )

    bootcamp_vs_alternatives_search_tool = create_retriever_tool(
        retriever=retriever_bootcamp_vs_alternatives,
        name="bootcamp_vs_alternatives_search",
        description='''Use this tool to retrieve information about the pros and cons of bootcamps compared to other learning methods, such as online courses and academic institutions. 
        **Avoid using this tool for questions unrelated to educational comparisons, such as questions about public figures, unrelated topics, or specific bootcamp programs like Eskwelabs.** 
        This tool is not intended for queries about specific bootcamp details.
        Additionally, this tool can be used in conjunction with the resume_search tool for queries assessing whether a user's skills and qualifications make a bootcamp or an alternative learning path a better option for them.
        '''
    )

    return eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool

with st.sidebar:
    resume = st.file_uploader("Upload File", type=['txt', 'docx', 'pdf'])

    if resume is not None:
        with open(resume.name, "wb") as f:
            f.write(resume.getbuffer())
        resume_tool = resume_retriever_tool(resume.name)
        eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool = create_db_retriever_tools(vectordb)
        tools = [resume_tool, eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool]

    else:
        eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool = create_db_retriever_tools(vectordb)
        tools = [eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool]


agent=create_tool_calling_agent(llm,tools,prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True, handle_parsing_errors=True)


def process_chat(agent_executor, user_input, chat_history):
    response = agent_executor.invoke(
        {'input': user_input,
         'chat_history': chat_history
         },
    )
    return response['output']

def format_message(message):            # turns human/ai message instance to string (to be used for metadatas)
    if isinstance(message, HumanMessage):
        return f"Human: {message.content}"
    elif isinstance(message, AIMessage):
        return f"AI: {message.content}"
    
def parse_message(formatted_message):           # turns string into human/ai message type instance (to be fed to chat history)
    if formatted_message.startswith("Human: "):
        content = formatted_message[len("Human: "):]
        return HumanMessage(content=content)
    elif formatted_message.startswith("AI: "):
        content = formatted_message[len("AI: "):]
        return AIMessage(content=content)
    else:
        raise ValueError("Unknown message format")
    



if "messages" not in st.session_state:
    st.session_state.messages = []


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "unique_id" not in st.session_state:
    st.session_state.unique_id = 0


if "chat_history_vector_store" not in st.session_state:
    st.session_state.chat_history_vector_store = None

if "fed_chat_history" not in st.session_state:
    st.session_state.fed_chat_history = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("Say something"):
    # Display user message in chat message container
    with st.chat_message("human"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.chat_history_vector_store:
        results = st.session_state.chat_history_vector_store.similarity_search(query=user_input,
                                                                k=4,
                                                                filter={'use_case':'chat_history'})

        sequenced_chat_history = [(parse_message(results.metadata['msg_element']), results.metadata['msg_placement']) for results in results]
        sequenced_chat_history.sort(key=lambda pair: pair[1])
        st.session_state.fed_chat_history = [message[0] for message in sequenced_chat_history]


    # chatbot response
    response = process_chat(agent_executor, user_input, st.session_state.fed_chat_history)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

    formatted_human_message = format_message(HumanMessage(content=user_input))
    formatted_ai_message = format_message(AIMessage(content=response))


    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


    # Add the last two messages (HumanMessage and AIMessage) to the vector store
    if st.session_state.chat_history_vector_store:
        st.session_state.chat_history_vector_store.add_texts(
            texts=[st.session_state.chat_history[-2].content, st.session_state.chat_history[-1].content], 
            ids=[str(st.session_state.unique_id), str(st.session_state.unique_id + 1)],
            metadatas=[
                {'msg_element': formatted_human_message, 'msg_placement': str(st.session_state.unique_id), 'use_case':'chat_history'},
                {'msg_element': formatted_ai_message, 'msg_placement': str(st.session_state.unique_id+1), 'use_case':'chat_history'}
            ],
            embedding=embedding_function
        )
        st.session_state.unique_id += 2
    else:
        # Initialize the vector store with the last two messages
        st.session_state.chat_history_vector_store = Chroma.from_texts(
            texts=[st.session_state.chat_history[-2].content, st.session_state.chat_history[-1].content], 
            ids=[str(st.session_state.unique_id), str(st.session_state.unique_id + 1)],
            metadatas=[
                {'msg_element': formatted_human_message, 'msg_placement': str(st.session_state.unique_id), 'use_case':'chat_history'},
                {'msg_element': formatted_ai_message, 'msg_placement': str(st.session_state.unique_id+1), 'use_case':'chat_history'}
            ],
            embedding=embedding_function
        )
        st.session_state.unique_id += 2
    
    st.session_state.chat_history = [] # after embedding convos to vector store, clear chat history before the end of the loop

