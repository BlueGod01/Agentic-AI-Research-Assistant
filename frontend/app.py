import streamlit as st
import requests

st.set_page_config(page_title="Agentic Research Assistant", layout="wide")
st.title("AI Research Assistant")

backend_url = "http://localhost:8000"

# Sidebar for file upload
st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=['pdf', 'txt'])

if st.sidebar.button("Upload"):
    if uploaded_file:
        with st.spinner("Uploading and processing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                res = requests.post(f"{backend_url}/upload", files=files)
                res.raise_for_status()
                st.sidebar.success("Upload successful and embeddings populated!")
            except Exception as e:
                st.sidebar.error(f"Upload failed: {str(e)}")
    else:
        st.sidebar.warning("Please select a file.")

# Main window for querying the assistant
st.header("Ask the Assistant")
query = st.text_input("Enter your research query:")
generate_pdf = st.checkbox("Generate PDF Notes")

if st.button("Search"):
    if query:
        with st.spinner("Agent is researching..."):
            try:
                payload = {"query": query, "generate_pdf": generate_pdf}
                response = requests.post(f"{backend_url}/query", json=payload, stream=True)
                
                placeholder = st.empty()
                full_response = ""
                
                # Consume Server-Sent Events from FastAPI stream
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            content = decoded_line[6:]
                            full_response += content + "\n\n"
                            placeholder.markdown(full_response)
                            
            except Exception as e:
                st.error(f"Query failed: {str(e)}")
    else:
        st.warning("Please enter a query.")
