import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import SYSTEM_PROMPT
from src.llm.model_manager import LLMModelManager
from src.vector_store.vector_store import VectorStore


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

        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def chat(self, message, history):
        return self.rag_chain.invoke(str(message))

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
