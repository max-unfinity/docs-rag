from operator import itemgetter
from typing import Sequence

from langchain.chat_models import ChatOpenAI
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
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Here are some documents from Supervisely Developer docs to expand your knowledge about the topic:"
                "\n{context}\n"
                "Carefully respond to the user query using the provided documents. It there is no relevant information, mention it to the user and try to answer based on your knowledge.",
            ),
            ("human", "{question}"),
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


def run_chain(chain, retriever, prompt):
    response = chain.invoke({"question": prompt})
    docs = retriever.invoke(prompt)
    return response, docs