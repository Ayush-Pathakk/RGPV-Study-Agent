from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

batch = next(index.list())

# Convert ListItem -> string id
ids = [item.id for item in batch.vectors[:5]]

print(ids)

fetched = index.fetch(ids=ids)

print(fetched)
print("Returned vectors:", len(fetched.vectors))

if fetched.vectors:
    first = next(iter(fetched.vectors.values()))
    print(first.metadata)