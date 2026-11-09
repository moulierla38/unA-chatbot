
import chromadb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "vector_db"

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_collection(
    name="defender_vehicle_specs"
)

# Search the vehicle database
results = collection.query(
    query_texts=["Which Defender has the highest horsepower?"],
    n_results=3
)

print("\nDefender search results:\n")

for document in results["documents"][0]:
    print(document)
    print("-" * 40)
