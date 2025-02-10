import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("API_KEY")  # Still needed for embeddings

class DeepSeekLLM:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "localhost:8501",
            "X-Title": "RAG PDF QA"
        }
        
    def invoke(self, prompt):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=self.headers,
            data=json.dumps({
                "model": "deepseek/deepseek-r1-distill-llama-70b:free",
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        return response.json()['choices'][0]['message']['content']

class RAGApplication:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = DeepSeekLLM()
        self.vector_store = None
        self.pdf_content = None

    def process_pdf(self, pdf_file):
        try:
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.getvalue())
            
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            self.pdf_content = " ".join([doc.page_content for doc in documents])
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            
            self.vector_store = FAISS.from_documents(
                documents=texts,
                embedding=self.embeddings
            )
            
            os.remove("temp.pdf")
            return len(texts)
        except Exception as e:
            st.error(f"PDF processing failed: {e}")
            return 0

    def get_answer(self, query: str) -> str:
        if not self.vector_store:
            return "Please upload a PDF document first."
        
        try:
            relevant_docs = self.vector_store.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            prompt = f"""Use the following context to answer the question.
            Context: {context}
            Question: {query}
            Answer: """
            
            return self.llm.invoke(prompt)
        except Exception as e:
            return f"Failed to generate answer: {e}"

    def generate_example_questions(self) -> list:
        if not self.pdf_content:
            return []
        
        try:
            prompt = f"Generate 5 key questions about this document:\n{self.pdf_content[:2000]}"
            response = self.llm.invoke(prompt)
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            return questions[:5]
        except Exception:
            return [
                "What is the main topic?",
                "Summarize key points",
                "What are the conclusions?",
                "Who is the target audience?",
                "What are the recommendations?"
            ]

def main():
    st.set_page_config(page_title="RAG PDF Q&A", layout="wide")

    st.markdown("""
        <div style='position: absolute; top: 0; left: 0; padding: 10px; color: rgba(0,0,0,0.3); font-size: 16px;'>
            by Teju SMP
        </div>
    """, unsafe_allow_html=True)

    st.title("PDF Question Answering System")
    st.subheader("Built Using RAG and DeepSeek AI")

    if 'rag_app' not in st.session_state:
        st.session_state.rag_app = RAGApplication()

    pdf_file = st.file_uploader("Upload your PDF", type=['pdf'])
    if pdf_file and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            num_chunks = st.session_state.rag_app.process_pdf(pdf_file)
            if num_chunks > 0:
                st.success(f"PDF processed successfully! Created {num_chunks} text chunks.")

    query = st.text_input("Ask a question about your PDF:")
    if query:
        with st.spinner("Getting answer..."):
            answer = st.session_state.rag_app.get_answer(query)
            st.write("Answer:", answer)

    st.markdown("---")
    st.subheader("Suggested Questions")
    if st.session_state.rag_app.pdf_content:
        questions = st.session_state.rag_app.generate_example_questions()
        for i, question in enumerate(questions, 1):
            st.write(f"{i}. {question}")
    else:
        st.info("Upload a PDF to see suggested questions")

if __name__ == "__main__":
    main()