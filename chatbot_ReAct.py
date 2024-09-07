from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import create_tool_calling_agent, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages.utils import trim_messages
from langchain import hub
from langchain_core.prompts import PromptTemplate

load_dotenv()



prompt = hub.pull("hwchase17/react")
# print(prompt)
# print("...")


custom_react_prompt = PromptTemplate.from_template('''
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!

Question: {input}

Thought:{agent_scratchpad}''')


# print(custom_react_prompt)



embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_DATA_PATH = 'embeddings_practicev1'
COLLECTION_NAME = 'embeddings_practicev1'


persistent_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = persistent_client.get_collection(COLLECTION_NAME)

vectordb = Chroma(client=persistent_client, collection_name=COLLECTION_NAME,embedding_function=embedding_function)

llm = ChatOllama(
    model = "llama3.1",
    temperature = 0.5,
    num_predict = 256,
    verbose=True
    # other params ...
)

# llm = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0,
#     max_tokens=200,)

prompt = ChatPromptTemplate(
    messages=[
        MessagesPlaceholder(variable_name='chat_history'),
        ('system', 'You are a friendly assistant that answers questions on user inquiries.'),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name="agent_scratchpad")       # need exact variable name
                                                                    # The agent prompt must have an `agent_scratchpad` key
    ]
)


def resume_retriever_tool(url):
    loader = PyPDFLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)

    chunks = splitter.split_documents(docs)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=chunks, embedding = embedding_function)
    retriever = vector_store.as_retriever()
    resume_retriever_tool = create_retriever_tool(
        retriever=retriever, 
        name='resume_search', 
        description="Use this tool to parse the user's resume for details such as name, contact information, educational background, skillset (soft and technical skills), and job experiences."
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
        This tool can answer queries about Eskwelabs' tuition fees, equipment requirements, program duration, curriculum, enrollment processes, scholarship offers, and other specific details.'''
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
        **Avoid using this tool for questions unrelated to educational comparisons, such as questions about public figures, unrelated topics, or specific bootcamp programs like Eskwelabs.**'''
    )

    return eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool


resume_tool = resume_retriever_tool('C:/Users/alfon/Desktop/chatbot/rag/Alfonso Kan - Resume.pdf')
eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool = create_db_retriever_tools(vectordb)
tools = [resume_tool, eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool]


# agent=create_tool_calling_agent(llm,tools,prompt)
# agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)


agent = create_react_agent(llm, tools, custom_react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

response = agent_executor.invoke({'input': 'What are the skills on my resume and does these skills make me a good fit for Eskwelabs DSF Program?'})
print(response['output'])



# def process_chat(agent_executor, user_input, chat_history):
#     response = agent_executor.invoke(
#         {'input': user_input,
#          'chat_history': chat_history
#          },
#     )
#     return response['output']

# if __name__ == '__main__':
#     chat_history = []

#     while True:
#         user_input = input("You: ")

#         if user_input.lower() == 'exit':
#             break

#         if chat_history:
#             chat_history = trim_messages(
#                 chat_history,
#                 max_tokens=1000,
#                 strategy="last",
#                 token_counter=llm,
#                 include_system=True,
#             )

#         print(f"Trimmed chat history: {chat_history}")
#         response = process_chat(agent_executor, 
#                                 user_input, 
#                                 chat_history
#                                 )


#         chat_history.append(HumanMessage(content=user_input))
#         chat_history.append(AIMessage(content=response))

#         print("Assistant: ", response)

