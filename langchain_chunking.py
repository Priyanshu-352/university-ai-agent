
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

MAX_RETRIES = 6
MAX_BACKOFF = 60
BATCH_SIZE = 20

file_path = "./disease-handbook-complete.pdf"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "my_rag"

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Missing Qdrant credentials in .env")



def call_with_retry(fn, *args, max_retries=MAX_RETRIES,
                    what="request", **kwargs):

    attempt = 0

    while True:
        try:
            return fn(*args, **kwargs)

        except Exception as exc:

            if attempt >= max_retries:
                raise

            delay = min(2 ** attempt, MAX_BACKOFF)

            print(
                f"Error during {what}: {exc}\n"
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)
            attempt += 1



from langchain_community.document_loaders import PyMuPDFLoader

if not os.path.exists(file_path):
    raise FileNotFoundError(f"PDF not found: {file_path}")

loader = PyMuPDFLoader(
    file_path="disease-handbook-complete.pdf"
)

docs = loader.load()

print(f"Total PDF pages loaded: {len(docs)}")

print("\nFirst page preview:\n")
print(docs[0].page_content[:500])


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False
)

texts = text_splitter.split_documents(docs)

print(f"\nTotal chunks generated: {len(texts)}")

##Embedding
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("\nEmbedding model loaded successfully!")




from langchain_qdrant import QdrantVectorStore

qdrant = None

total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, len(texts), BATCH_SIZE):

    batch = texts[i:i + BATCH_SIZE]

    batch_number = (i // BATCH_SIZE) + 1

    print(
        f"\nProcessing batch {batch_number}/{total_batches}"
        f" ({len(batch)} chunks)..."
    )

    if qdrant is None:

        qdrant = call_with_retry(
            QdrantVectorStore.from_documents,

            documents=batch,
            embedding=embeddings,

            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,

            collection_name=COLLECTION_NAME,

            force_recreate=False,

            what="Qdrant indexing"
        )

    else:

        call_with_retry(
            qdrant.add_documents,
            documents=batch,
            what="Qdrant batch insertion"
        )

    print(f"Batch {batch_number} completed!")

print("\nAll documents successfully embedded and stored!")

print(f"Collection: {COLLECTION_NAME}")
print(f"Total chunks indexed: {len(texts)}")
