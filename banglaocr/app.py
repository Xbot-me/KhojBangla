import streamlit as st
import sqlite3
import os
from PIL import Image
import sys
import chromadb
from sentence_transformers import SentenceTransformer

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from storage import get_review_queue, update_review_status, get_page_text, connect

st.set_page_config(layout="wide", page_title="Bangla Historical Newspaper OCR")

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr.db")

if not os.path.exists(db_path):
    st.error(f"Database not found at {db_path}. Run the pipeline first.")
    st.stop()

# Initialize Embedding Model and ChromaDB
@st.cache_resource
def load_search_engine():
    # Use a multilingual model for Bengali
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="newspaper_archive")
    return model, collection

model, collection = load_search_engine()

# --- Sync Verified Texts to ChromaDB ---
# In a real app, this would happen in the pipeline or after human review automatically
def sync_to_vector_db():
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT orr.id, orr.corrected_text, orr.text, p.source_path, lc.page_id 
               FROM ocr_results orr
               JOIN line_crops lc ON lc.id = orr.line_crop_id
               JOIN pages p ON p.id = lc.page_id
               WHERE orr.status = 'verified'"""
        ).fetchall()
        
    for row in rows:
        result_id = str(row[0])
        text = row[1] if row[1] else row[2]
        source_path = row[3]
        page_id = row[4]
        
        # Check if already embedded
        res = collection.get(ids=[result_id])
        if not res['ids']:
            embedding = model.encode(text).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"source_path": source_path, "page_id": page_id}],
                ids=[result_id]
            )

tab1, tab2 = st.tabs(["Review Queue", "Search Archive"])

with tab1:
    st.title("Human Verification Queue")
    # Get pending reviews
    queue = get_review_queue(db_path)

    if not queue:
        st.success("No items in the review queue! You're all caught up.")
    else:
        st.sidebar.header(f"Pending Reviews: {len(queue)}")
        item = queue[0]
        st.subheader(f"Review Item ID: {item['result_id']} (Page {item['page_id']})")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Original Crop")
            try:
                img_path = item['source_path']
                if not os.path.isabs(img_path):
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    img_path = os.path.normpath(os.path.join(project_root, img_path))
                
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    bbox = item['bbox']
                    parsed_bbox = []
                    for b in bbox:
                        if isinstance(b, bytes):
                            parsed_bbox.append(int.from_bytes(b, byteorder='little'))
                        else:
                            parsed_bbox.append(int(b))
                    x, y, w, h = parsed_bbox
                    crop = img.crop((x, y, x + w, y + h))
                    st.image(crop, use_container_width=True, caption=f"Confidence: {item['confidence']:.2f}")
                else:
                    st.error(f"Image not found: {img_path}")
            except Exception as e:
                st.error(f"Error loading image: {e}")

        with col2:
            st.markdown("### OCR & LLM Correction")
            default_text = item['corrected_text'] if item['corrected_text'] else item['text']
            corrected_text = st.text_area(
                "Edit text if necessary:",
                value=default_text,
                height=200
            )
            if st.button("Save & Verify", type="primary"):
                update_review_status(db_path, item['result_id'], corrected_text, status='verified')
                sync_to_vector_db() # Sync immediately upon verification
                st.success("Saved! Loading next item...")
                st.rerun()

with tab2:
    st.title("Search Historical Archive")
    
    # Sync button for manual sync
    if st.button("Sync Verified Data to Search Engine"):
        with st.spinner("Syncing..."):
            sync_to_vector_db()
        st.success("Sync complete!")
        
    query = st.text_input("Enter search query (in Bengali or English):")
    
    if query:
        with st.spinner("Searching..."):
            query_embedding = model.encode(query).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10
            )
            
            if results['ids'] and results['ids'][0]:
                st.write(f"Found {len(results['ids'][0])} results:")
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    doc_text = results['documents'][0][i]
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    st.markdown(f"**Result {i+1}** (Distance: {distance:.4f})")
                    st.markdown(f"> {doc_text}")
                    st.caption(f"Source: {metadata['source_path']} | Page: {metadata['page_id']}")
                    st.divider()
            else:
                st.warning("No results found.")
