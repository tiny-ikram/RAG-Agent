# rag_agent.py
import os
import uuid
import base64
from base64 import b64decode
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from unstructured.partition.pdf import partition_pdf

# ----------- CONFIG -----------
PERSIST_DIR = "./data/chroma_store"
id_key = "doc_id"

# ----------- GLOBALS -----------
vectorstore = None
store = None
retriever = None
chain_with_sources = None

# ----------- HELPERS -----------
def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def parse_docs(docs):
    """Split base64-encoded images and texts."""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and images.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]
    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })

    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

def safe_add_documents(retriever, summaries, originals):
    if not summaries or not originals:
        return
    doc_ids = [str(uuid.uuid4()) for _ in originals]
    docs = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(summaries)
        if summary and summary.strip()
    ]
    if docs:
        retriever.vectorstore.add_documents(docs)
        retriever.docstore.mset(list(zip(doc_ids, originals)))
        retriever.vectorstore.persist()

# ----------- INIT -----------
def init_retriever():
    global vectorstore, store, retriever, chain_with_sources
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory=PERSIST_DIR
    )
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
            | StrOutputParser()
        )
    )

# ----------- INDEXING -----------
def index_pdf(file_path):
    # Extract chunks
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000
    )

    # Separate types
    texts, tables = [], []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        elif "CompositeElement" in str(type(chunk)):
            texts.append(chunk)

    images = get_images_base64(chunks)

    # Summarize text
    text_summaries = summarize_batch([t.text for t in texts])
    table_summaries = summarize_batch([t.metadata.text_as_html for t in tables])
    image_summaries = summarize_images(images)

    # Add to store
    safe_add_documents(retriever, text_summaries, texts)
    safe_add_documents(retriever, table_summaries, tables)
    safe_add_documents(retriever, image_summaries, images)

def summarize_batch(elements):
    if not elements:
        return []
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant tasked with summarizing tables and text.
    Respond only with the summary, no extra comment.
    Element: {element}
    """)
    chain = {"element": lambda x: x} | prompt | ChatGroq(temperature=0.5, model="llama-3.1-8b-instant") | StrOutputParser()
    return chain.batch(elements, {"max_concurrency": 3})

def summarize_images(images_b64):
    if not images_b64:
        return []
    prompt_template = """Describe the image in detail."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
            ],
        )
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-1.5-flash") | StrOutputParser()
    formatted_images = [{"image": img} for img in images_b64]
    return chain.batch(formatted_images)

# ----------- QUERYING -----------
def query_agent(question):
    return chain_with_sources.invoke(question)
