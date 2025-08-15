from config import SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.llm.model_manager import LLMModelManager
from src.vector_store.vector_store import VectorStore
from dotenv import load_dotenv

def main():
    load_dotenv()
    llm = LLMModelManager().get_chat_model()
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    print("\nEmbed Response:")
    retriever = VectorStore().load().as_retriever()

    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    # result = rag_chain.invoke("What is the volume number of the latest Radar? List down the trial topics")
    result = rag_chain.invoke("What is the on the radar that is not about genAI and LLMs for a developer to learn?")

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(result)

if __name__ == "__main__":
    main()
