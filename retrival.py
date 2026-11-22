from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)



qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="my_rag",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)



query = "What are the symptoms of diabetes?"


results = qdrant.similarity_search(
    query,
    k=5 
)


for i, doc in enumerate(results, 1):

    print(f"\n----- Result {i} -----")

    print("Content:")
    print(doc.page_content)

    print("\nSource:")
    print(doc.metadata.get("source"))

    print("\nPage:")
    print(doc.metadata.get("page", -1) + 1)