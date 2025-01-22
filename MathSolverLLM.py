import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain,LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from dotenv import load_dotenv
load_dotenv()
from langchain.callbacks import StreamlitCallbackHandler

## Streamlit app

st.title("Math Problem solver and data search assistant")

api_key=st.sidebar.text_input("Enter the GRoQ API key",type="password")

if not api_key:
    st.info("Please enter your Groq Api key to continue")
    st.stop()

llm=ChatGroq(model="Gemma2-9b-it",groq_api_key=api_key)

## initailizing the tools

wiki_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="A tool for searching the internet for the various information asked by the user"
)

## intialise the MAth tool

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="calculator",
    func=math_chain.run,
    description="A tool for answering math releted problems. Only mathamatical expression needs to be provided"
)

prompt="""  
You are an agent to solve the user's mathamatical questions.
logically arrive at the solution and describe it elaborately and 
display it point wise.
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=['question'],
    template=prompt
)

## combine all the tools to chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic based and reasoning questions"

)


## Initialize the agents
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True
)


## state 

if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"Hi, I'm a Math chatbot, I will be answering your math queries today"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## fn to generate response

def generate_response(user_question):
    response=assistant_agent.invoke({"input":user_question})
    return response

## lets start the interaction
question=st.chat_input(placeholder="Ask")

if question:
    with st.spinner("Model is generating your result......"):
        st.session_state.messages.append({"role":"user","content":question})
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.success(response)
else:
    st.warning("Please enter the question")     
