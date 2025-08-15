import gradio as gr
from config import SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.llm.model_manager import LLMModelManager
from src.vector_store.vector_store import VectorStore
from dotenv import load_dotenv

class ChatBot:
    def __init__(self):
        self.rag_chain = self._setup_rag_chain()

    def _setup_rag_chain(self):
        llm = LLMModelManager().get_chat_model()
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        retriever = VectorStore().load().as_retriever()

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    def chat(self, message, history):
        response = self.rag_chain.invoke(str(message))
        return response

def main():
    """Initialize and launch the Gradio chat interface."""
    load_dotenv()
    
    chatbot = ChatBot()
    app = gr.ChatInterface(
        fn=chatbot.chat,
        type="messages",
    )
    app.launch()

if __name__ == "__main__":
    main()
