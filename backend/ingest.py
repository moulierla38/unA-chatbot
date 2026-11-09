
import pandas as pd
import chromadb
from pathlib import Path

# Find project folders
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "data" / "documents" / "land_rover_defender_complete_dataset.csv"
DB_PATH = BASE_DIR / "vector_db"

# Read the Defender dataset
df = pd.read_csv(CSV_PATH)

print("Dataset loaded successfully!")
print("Number of vehicles:", len(df))

# Connect to ChromaDB
client = chromadb.PersistentClient(path=str(DB_PATH))

# Create a collection for vehicle specifications
collection = client.get_or_create_collection(
    name="defender_vehicle_specs"
)

documents = []
ids = []
metadatas = []

# Convert each vehicle record into searchable text
for index, row in df.iterrows():

    vehicle_text = "\n".join(
        f"{column.replace('_', ' ')}: {value}"
        for column, value in row.items()
        if pd.notna(value)
    )

    documents.append(vehicle_text)
    ids.append(f"defender_vehicle_{index + 1}")

    metadatas.append({
        "make": str(row["Make"]),
        "model": str(row["Model"]),
        "trim": str(row["Trim"]),
        "year": int(row["Year"])
    })

# Store vehicle information in ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print("Defender vehicle data stored successfully!")
print("Total records in ChromaDB:", collection.count())
