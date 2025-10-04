import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from config import SYSTEM_PROMPT
from src.llm.model_manager import LLMModelManager
from src.stores.vector_store import VectorStore
from src.stores.graph_store import TechGraphStore



class ChatBot:
    def __init__(self):
        self.rag_chain = self._setup_graphrag_chain()

    def _setup_graphrag_chain(self):
        llm = LLMModelManager().get_chat_model()
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        graph_store = TechGraphStore()
        graph_store.load()
        retriever = graph_store.as_retriever()

        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

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

    def _preprocess_query(self, query: str) -> str:
        """Normalize common terms in user queries."""
        query = query.lower()
        
        # Normalize ring terms
        ring_mappings = {
            'assess': 'Assess', 'assessment': 'Assess', 'assessing': 'Assess',
            'trial': 'Trial', 'trying': 'Trial', 'experiment': 'Trial',
            'adopt': 'Adopt', 'adoption': 'Adopt', 'adopted': 'Adopt',
            'hold': 'Hold', 'holding': 'Hold', 'avoid': 'Hold'
        }
        
        # Normalize quadrant terms
        quadrant_mappings = {
            'technique': 'Techniques', 'platform': 'Platforms',
            'tool': 'Tools', 'languages': 'Languages and \nFrameworks',
            'frameworks': 'Languages and \nFrameworks'
        }
        
        for term, replacement in {**ring_mappings, **quadrant_mappings}.items():
            query = query.replace(term, replacement)
        
        return query

    def chat(self, message, history):
        processed_message = self._preprocess_query(str(message))
        return self.rag_chain.invoke(processed_message)

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
