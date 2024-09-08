# Eskwelabs Chatbot
This project focuses on the end-to-end development of a Q&A Chatbot tailored to answer bootcamp-related queries, specifically for Eskwelabs. The methodology covers the key steps, including knowledge base embedding,  Retrieval-Augmented Generation (RAG) chatbot development using LangChain, and deployment via Streamlit. 

Disclaimer: This guide demonstrates how to create your own chatbot using `Llama3.1`. However, `Llama3.1` struggles with making multiple tool calls simultaneously. In contrast, `GPT-Turbo-3.5` handles this task more efficiently. For improved results, `GPT-Turbo-3.5` is recommended over `Llama3.1`.

![image](https://github.com/user-attachments/assets/5c12ffc1-101f-46f1-bbd0-af179c133e75)

## Installation
1. Clone the repository
```bash
git clone https://github.com/alfonsokan/eskwelabs_chatbot.git
```
2. Install libraries
```bash
pip install -r requirements.txt
```
3. Install an open-source LLM using Ollama. Refer to the [Ollama documentation](https://github.com/ollama/ollama) and select an LLM.

<br/>

Then, run the following command in the command line (CMD):

```bash
ollama run llama3.1
```

4. For the code repository, open terminal and run the following command:
```python
streamlit run app.py
```

## Methodology
The flow chart below displays the 4-step approach to developing the chatbot.

![Askwelabs_Capstone-Project_Group-3](https://github.com/user-attachments/assets/72319f8f-0e94-4b05-8aa8-9fdadb78e640)

**1. Data Preparation**



For this step, the documents are embeddings and stored in the vector database called `embeddings_deployment_sentencetransformer` located in this repository. 

If interested, the code for embedding the documents can be viewed [here](https://colab.research.google.com/drive/1iyz_SkHv7TVDgKJBuRYb1iTtVfxDyGU0?usp=sharing).

<br/>

**2. Retriever Generation**
- There are two retriever tools, Eskwelabs Info Retriever and General Bootcamp Info Retriever, created from the embedded knowledge base.



- Another retriever is optionally used when a user submits their resume to the chatbot.
    
![image](https://github.com/user-attachments/assets/dc6b0a63-9e09-49dc-bc5b-1df98b1e3343)

<br/>

**3. Tool-calling Agent Creation**

Three parameters to instantiate a tool-calling agent:

- List of retriever tools
```python
    resume = st.file_uploader("Upload File", type=['txt', 'docx', 'pdf'])

    # if a resume is passed, include resume retriever as tool
    if resume is not None:
        with open(resume.name, "wb") as f:
            f.write(resume.getbuffer())
        resume_tool = resume_retriever_tool(resume.name)
        eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool = create_db_retriever_tools(vectordb)
        tools = [resume_tool, eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool]

    # if no resume is passed, do not include resume retriever as tool
    else:
        eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool = create_db_retriever_tools(vectordb)
        tools = [eskwelabs_bootcamp_info_search_tool, bootcamp_vs_alternatives_search_tool]
```

- LLM (Llama 3.1)
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "llama3.1",
    temperature = 0.1,
    num_predict = 350,
    verbose=True.
)
```
- Prompt passed to the chatbot
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate(
    messages=[
        MessagesPlaceholder(variable_name='chat_history'),
        ('system', "You're a helpful assistant who provides concise, complete answers without getting cut off mid-statement. Stick strictly to the user's questions, avoiding any unnecessary details."),
        ('human', '{input}'),
        MessagesPlaceholder(variable_name="agent_scratchpad")       
                                                                    
    ]
)
```

Afterwards, the tool-calling agent can now be instantiated:
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

agent=create_tool_calling_agent(llm,tools,prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True, handle_parsing_errors=True)
```
<br/>

**4. Response Generation**
- Pass the tool-calling agent, user input, as well as chat history to generate a response.
```python
def process_chat(agent_executor, user_input, chat_history):
    response = agent_executor.invoke(
        {'input': user_input,
         'chat_history': chat_history
         },
    )
    return response['output']
```
<br/>

## App Preview
The guide was meant to help users deploy their own Q&A chatbot locally using open-source LLMs. However, the chatbot can be deployed easier via Streamlit by getting access to an LLM through an API key. View the deployed project [here](https://askwelabscapstoneproject.streamlit.app/) using GPT-Turbo-3.5 as the LLM instead.


<br/>

## Recommendations
- Using a ReAct agent instead of a tool-calling agent
    -   Explore output quality using a ReAct agent instead of a tool-calling agent. Develop a ReAct prompt that enables the LLM to generate reasoning traces before taking action on a task.
- Explore different chunking strategies and embedding model
- Connect the chatbot to a third-party database (Redis-Upstash) to allow long-term storage of chat history

<br/>

## Appendix
### Chatbot's Selective History Retrieval Mechanism
- Past conversations’ user query and chatbot response are stored in a temporary vector store
- For each new user query, only the most relevant parts of the vector store are retrieved and passed as chat history to the chatbot.
- Importance: This feature reduces token consumption by efficiently retrieving only the relevant parts of the chat history, preventing unnecessary length and minimizing what is passed to the LLM.
- Code snippet of app with chat history implemented:
```python
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
```

<br/>

### Knowledge Base Embedding
The embedding of the knowledge base can be found [here.](https://colab.research.google.com/drive/1iyz_SkHv7TVDgKJBuRYb1iTtVfxDyGU0?usp=sharing)

<br/>

### LangSmith Tracing
LangSmith can be a useful tool for debugging the chatbot application. To trace its runs, do the following:
- Create a LangSmith account
- Retrieve API key
- Create `.env` file
```python
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY='ENTER_API_KEY_HERE'
LANGCHAIN_PROJECT='PROJECT_NAME'
```
- In the `app.py` file, make sure environment variables are loaded properly.
```python
from dotenv import load_dotenv
load_dotenv()
```
- LangSmith can be used to check if the LLM calls the appropriate tool/s for the prompt.
![image](https://github.com/user-attachments/assets/b848364f-512d-436f-9455-40d709112d7e)
![image](https://github.com/user-attachments/assets/2a8084b6-28a2-41e9-8ad9-f4b5d706e199)

