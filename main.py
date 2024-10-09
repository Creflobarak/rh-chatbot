import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PIL import Image
import io
import base64

# Set up the Streamlit app
st.set_page_config(page_title="Multimodal Chatbot", page_icon="🤖")
st.title("Multimodal Chatbot")

# Initialize Ollama with the llava model
ollama = Ollama(base_url="http://localhost:11434", model="llama3.1")

# Function to encode image to base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    encoded_image = encode_image(image)
    
    # Add image to chat history
    st.session_state.messages.append({"role": "user", "content": "Image uploaded"})
    with st.chat_message("user"):
        st.image(image, width=300)

# Generate response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if uploaded_file:
                # If image was uploaded, use it in the prompt
                prompt_template = PromptTemplate(
                    input_variables=["image", "prompt"],
                    template="[IMG]{image}[/IMG]\nHuman: {prompt}\nAssistant: "
                )
                chain = LLMChain(llm=ollama, prompt=prompt_template)
                response = chain.invoke({"image": encoded_image, "prompt": prompt})
            else:
                # Text-only prompt
                prompt_template = PromptTemplate(
                    input_variables=["prompt"],
                    template="Human: {prompt}\nAssistant: "
                )
                chain = LLMChain(llm=ollama, prompt=prompt_template)
                response = chain.invoke({"prompt": prompt})
            
            st.markdown(response['text'])
            st.session_state.messages.append({"role": "assistant", "content": response['text']})

# Add a clear button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()