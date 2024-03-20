from operator import itemgetter
from typing import Sequence

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable.passthrough import RunnableAssign


# After the retriever fetches documents, this
# function formats them in a string to present for the LLM
def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = (
            f"<document index='{i}'>\n"
            f"<source>{doc.metadata.get('source')}</source>\n"
            f"<doc_content>{doc.page_content}</doc_content>\n"
            "</document>"
        )
        formatted_docs.append(doc_string)
    formatted_str = "\n".join(formatted_docs)
    return f"<documents>\n{formatted_str}\n</documents>"


def get_chain(retriever, model="gpt-3.5-turbo-1106", temperature=0.7):
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "Here are some documents from Supervisely Developer docs to expand your knowledge about the topic:"
    #             "\n{context}\n"
    #             "Carefully respond to the user query using the provided documents."
    #             "If provided documents are not relevant, mention it to the user and answer based on your knowledge.",
    #         ),
    #         ("human", "{question}"),
    #     ]
    # )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a specialized assistant focused on helping users to understand documentation. You are adept at providing coding assistance and answering questions related to the docs. You will be provided with parts of the documentation. You can incorporate the texts from documentation into your responses, offering further explanations. You handle multiple documents adeptly, synthesizing information to deliver comprehensive, accurate answers. You maintain a strictly professional and technical tone, ensuring responses are clear and precise, making it suitable for both beginners and seasoned programmers. Avoid misinformation. In situations involving ambiguous or partial questions, you make logical assumptions to offer useful guidance. The primary focus is to be a reliable resource of the information in documentation, ensuring users receive expert advice and code examples tailored to their needs.
                Here are texts from documentation:
                {context}
                Answer questions based on the provided texts. If provided texts are not relevant, mention it to the user and answer based on your knowledge.
                """
            ),
            (
                "human",
                """The question is:
                {question}"""
            ),
        ]
    )
    llm = ChatOpenAI(temperature=temperature, model=model)

    response_generator = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )

    # This is the final response chain.
    # It fetches the "question" key from the input dict,
    # passes it to the retriever, then formats as a string.

    chain = (
        RunnableAssign(
            {
                "context": (itemgetter("question") | retriever | format_docs).with_config(
                    run_name="FormatDocs"
                )
            }
        )
        # The "RunnableAssign" above returns a dict with keys
        # question (from the original input) and
        # context: the string-formatted docs.
        # This is passed to the response_generator above
        | response_generator
    )
    return chain


def retrieve_docs(retriever, prompt):
    docs = retriever.invoke(prompt)
    return docs


def generate_response(chain, prompt):
    response = chain.invoke({"question": prompt})
    return response


def generate_response_stream(chain, prompt):
    response_stream = chain.stream({"question": prompt})
    return response_stream