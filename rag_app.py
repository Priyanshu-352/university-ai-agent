
import os
import time
import random
import logging
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


st.set_page_config(
    page_title="Disease Handbook RAG",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "my_rag"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "Qwen/Qwen2.5-7B-Instruct"

MAX_RETRIES = 5
MAX_BACKOFF = 30



def get_secret(key):

    value = os.getenv(key)

    if value:
        return value

    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return None


GROQ_API_KEY = get_secret("HF_TOKEN")

QDRANT_URL = get_secret("QDRANT_URL")

QDRANT_API_KEY = get_secret("QDRANT_API_KEY")


# ==========================================
# 3. VALIDATE CONFIGURATION
# ==========================================

missing_keys = []

if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")

if not QDRANT_URL:
    missing_keys.append("QDRANT_URL")

if not QDRANT_API_KEY:
    missing_keys.append("QDRANT_API_KEY")


if missing_keys:

    st.error(
        "Missing configuration: "
        + ", ".join(missing_keys)
    )

    st.stop()



def is_retryable_error(exc):

    status_code = getattr(exc, "status_code", None)

    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)

    if status_code in (429, 500, 502, 503, 504):
        return True

    msg = str(exc).lower()

    return any(keyword in msg for keyword in (
        "rate limit",
        "too many requests",
        "resource exhausted",
        "temporarily unavailable",
        "connection error",
        "connection reset",
        "timed out",
        "timeout",
        "try again later"
    ))


def call_with_retry(
    fn,
    *args,
    max_retries=MAX_RETRIES,
    what="request",
    **kwargs
):

    for attempt in range(max_retries + 1):

        try:

            return fn(*args, **kwargs)

        except Exception as exc:

            if not is_retryable_error(exc):
                raise

            if attempt == max_retries:
                raise

            delay = min(
                2 ** attempt,
                MAX_BACKOFF
            ) + random.uniform(0, 1)

            logger.warning(
                "%s failed: %s. Retrying in %.2f seconds.",
                what,
                exc,
                delay
            )

            time.sleep(delay)


@st.cache_resource(show_spinner="Loading embedding model...")

def load_embedding_model():

    model = HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        }
    )

    return model


@st.cache_resource(show_spinner="Connecting to Qdrant...")

def load_qdrant(url, api_key):

    embedding_model = load_embedding_model()

    qdrant_client = QdrantClient(
        url=url,
        api_key=api_key
    )

    # Check whether the collection exists

    if not qdrant_client.collection_exists(
        COLLECTION_NAME
    ):

        raise ValueError(
            f"Collection '{COLLECTION_NAME}' does not exist. "
            "Run your PDF indexing script first."
        )

    # Connect to the existing collection

    vector_store = QdrantVectorStore(
        client=qdrant_client,

        collection_name=COLLECTION_NAME,

        embedding=embedding_model
    )

    return vector_store



from huggingface_hub import InferenceClient

@st.cache_resource
def load_hf_client(api_key):

    return InferenceClient(
        provider="auto",
        api_key=api_key,
        timeout=60
    )


try:

    qdrant = load_qdrant(
        QDRANT_URL,
        QDRANT_API_KEY
    )

    client = load_hf_client(
        GROQ_API_KEY
    )

except Exception as exc:

    logger.exception("Initialization failed")

    st.error(f"Initialization error: {exc}")

    st.stop()

def generate_answer(
    client,
    system_prompt,
    user_query,
    chat_history
):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Include previous conversation
    messages.extend(chat_history[-6:])

    messages.append({
        "role": "user",
        "content": user_query
    })

    response = client.chat.completions.create(

        model="Qwen/Qwen2.5-7B-Instruct",

        messages=messages,

        temperature=0.2,

        max_tokens=1200
    )

    return response.choices[0].message.content

def retrieve_documents(query):

    results = call_with_retry(

        qdrant.max_marginal_relevance_search,

        query,

        k=5,

        fetch_k=20,

        lambda_mult=0.7,

        what="Qdrant retrieval"
    )

    return results



#PREPARE CONTEXT AND CITATIONS


def prepare_context(results):

    context_parts = []

    citations = []

    seen = set()

    for index, doc in enumerate(results, 1):

        content = doc.page_content.strip()

        if not content or content in seen:
            continue

        seen.add(content)

        source = os.path.basename(
            doc.metadata.get("source", "Unknown source")
        )

        # PyMuPDFLoader page metadata is zero-indexed

        page = doc.metadata.get("page")

        if isinstance(page, int):
            page_number = page + 1
        else:
            page_number = "Unknown"

        context_parts.append(
            f"""
            [Document {index}]
            Source: {source}
            Page: {page_number}

            Content:
            {content}
            """
        )

        citations.append({
            "source": source,
            "page": page_number
        })

    context = "\n\n".join(context_parts)

    # Limit context size to control token consumption

    context = context[:12000]

    return context, citations


# ==========================================
# 11. SYSTEM PROMPT
# ==========================================

def create_system_prompt(context):

    return f"""
You are a reliable AI research assistant
specializing in information retrieval from documents.

Your task is to answer questions using ONLY
the provided document context.

DOCUMENT CONTEXT:

{context}

INSTRUCTIONS:

1. Answer using the information available
   in the provided context.

2. Do not invent medical facts or unsupported
   information.

3. If the answer is unavailable, respond:
   "I couldn't find specific information
   about this in the documents."

4. Synthesize relevant information from
   multiple retrieved document chunks.

5. Provide clear and structured answers.

6. Include source names and page numbers
   when available.

7. If the context contains conflicting
   information, mention the inconsistency.

8. Do not present generated information
   as a medical diagnosis or treatment recommendation.

9. When making a factual statement from the context,
   cite its source using [Source 1], [Source 2], etc.
10. Only cite sources that actually support the statement.

Use a professional and academic tone.
"""


st.title("🤖 Disease Handbook RAG Assistant")

st.caption(
    "Ask questions about your uploaded disease handbook "
    "using semantic search and AI-powered answers."
)

with st.sidebar:

    st.header("System Information")

    st.success("Qdrant connected")

    st.success("MiniLM embeddings loaded")

    st.success("Groq client initialized")

    st.write("**Embedding model:**")

    st.caption(EMBEDDING_MODEL)

    st.write("**Collection:**")

    st.caption(COLLECTION_NAME)

    if st.button("Clear conversation"):

        st.session_state.messages = []

        st.rerun()



if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

if user_query := st.chat_input(
    "Ask a question about your documents..."
):

    # Display user message

    st.session_state.messages.append({

        "role": "user",

        "content": user_query

    })

    with st.chat_message("user"):

        st.markdown(user_query)


    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        try:

            # Step 1: Retrieve relevant documents

            with st.spinner("Searching documents..."):

                results = retrieve_documents(
                    user_query
                )

                if not results:

                    answer = (
                        "I couldn't find specific information "
                        "about this in the documents."
                    )

                    message_placeholder.markdown(answer)

                    st.session_state.messages.append({

                        "role": "assistant",

                        "content": answer

                    })

                    st.stop()


                # Step 2: Prepare context

                context, citations = prepare_context(
                    results
                )


            # Step 3: Generate answer

            with st.spinner("Generating answer..."):

                system_prompt = create_system_prompt(
                    context
                )

                chat_history = [
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }

                    for message in st.session_state.messages[:-1]
                    if message["role"] in ("user", "assistant")
                ]

                answer = call_with_retry(

                    generate_answer,

                    client,

                    system_prompt,

                    user_query,

                    chat_history,

                    what="Groq generation"
                )


            # Step 4: Display answer

            message_placeholder.markdown(answer)


            # Step 5: Display citations

            if citations:

                with st.expander("📚 Retrieved sources"):

                    for citation in citations:

                        st.write(
                            f"- {citation['source']} "
                            f"— Page {citation['page']}"
                        )


            # Step 6: Save assistant response

            st.session_state.messages.append({

                "role": "assistant",

                "content": answer

            })


        except Exception as exc:

            logger.exception("Query processing failed")

            st.error(
                f"An error occurred: {exc}"
            )
