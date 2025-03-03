from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from constants import API_KEY
import os
import streamlit as st 
from langchain.memory import ConversationBufferMemory

os.environ["GOOGLE_API_KEY"] = 'AIzaSyCYVMVzaWO1owNnf60jEiz0qyd0WEG1hWI'

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',temperature=0.8)

first_prompt_template = PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"
)

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
desc_memory = ConversationBufferMemory(input_key='dob', memory_key='chat_history')

chain1 = LLMChain(llm=llm, prompt=first_prompt_template, verbose=True, memory=person_memory,output_key='person')

second_prompt_template = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

chain2 = LLMChain(llm=llm, prompt=second_prompt_template, verbose=True, output_key='dob', memory=dob_memory)

third_prompt_template = PromptTemplate(
    input_variables=['dob'],
    template="Mention any 5 major events happened around {dob} in the world."
)

chain3 = LLMChain(llm=llm, prompt=third_prompt_template, verbose=True, memory=desc_memory)

parent_chain = SequentialChain(chains=[chain1, chain2, chain3], input_variables=['name'], output_variables=['name','person','dob'],verbose=True)

input_text = st.text_input("Search the topic you want")

if input_text:
    st.write(parent_chain({'name':input_text}))
    with st.expander('Person Name'):
        st.info(person_memory.buffer)
    with st.expander('Major Events'):
        st.info(desc_memory.buffer)