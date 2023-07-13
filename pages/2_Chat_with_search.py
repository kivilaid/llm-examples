import streamlit as st
import streamlit.secrets as secrets

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

# Set Streamlit page configuration
st.set_page_config(page_title="LangChain - Chat with search")

# Retrieve the OpenAI API key from the cloud secret
openai_api_key = secrets["OPENAI_API_KEY"]

# Sidebar configuration
with st.sidebar:
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

# Main app title and description
st.title("🔎 LangChain - Chat with search")
st.markdown("""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
""")

# Initialize messages if not present in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display existing messages
for msg in st.session_state.messages:
    st.markdown(f"**{msg['role']}**: {msg['content']}")

# Get user input
prompt = st.text_input("User Input", placeholder="Who won the Women's U.S. Open in 2018?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Initialize LangChain agents
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    
    # Run the search agent
    with st.spinner("Searching..."):
        response = search_agent.run(st.session_state.messages)
    
    # Display the response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f"**assistant**: {response}")
