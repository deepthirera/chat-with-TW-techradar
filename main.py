import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from config import SYSTEM_PROMPT, GRAPH_SYSTEM_PROMPT
from src.llm.model_manager import LLMModelManager
from src.stores.vector_store import VectorStore
from src.stores.graph_store import TechGraphStore
from src.utils.logger import logger


class ChatBot:
    def __init__(self):
        self.rag_chain = self._setup_graphrag_chain()

    def _setup_graphrag_chain(self):
        graph_store = TechGraphStore()
        graph_store.load()
        llm = LLMModelManager().get_chat_model()
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", GRAPH_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        router_chain = router_prompt | llm | StrOutputParser()

        semantic_chain = (
            graph_store.semantic_retriever()
            | RunnableLambda(lambda response: response.get("result", "No answer found"))
        )

        cypher_chain = (
            graph_store.cypher_retriever()
            | RunnableLambda(lambda response: response.get("result", "No answer found"))
        )

        def route_and_search(user_question):
            search_type = router_chain.invoke({"question": user_question}).strip().lower()
            logger.info(f"The route chosen is, {search_type}")
            if search_type == "graph_semantic_search":
                return semantic_chain.invoke(user_question)
            elif search_type == "graph_cypher_search":
                return cypher_chain.invoke({"query": user_question})
            else:
                return "No answer found"
        return RunnableLambda(route_and_search)

    def _setup_graph_search_chain(self):
        graph_store = TechGraphStore()
        # graph_store.load()
        retriever_chain = graph_store.semantic_retriever()
        input_transformer = RunnableLambda(lambda user_question: user_question)
        result_extractor = RunnableLambda(lambda response: response.get("result", "No answer found"))
        return input_transformer | retriever_chain | result_extractor

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
