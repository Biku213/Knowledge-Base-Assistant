import os
import streamlit as st
import datetime
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from typing import List, Dict, Tuple, Optional
from PIL import Image
import json
import base64
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Knowledge Base Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main {
        padding: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e8f0fe;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .image-result {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize directories
DOCUMENTS_DIR = Path("documents")
DOCUMENTS_DIR.mkdir(exist_ok=True)

CHROMA_DB_PATH = Path("chroma_db")
CHROMA_DB_PATH.mkdir(exist_ok=True)

IMAGE_DATA_DIR = Path("image_data")
IMAGE_DATA_DIR.mkdir(exist_ok=True)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

# Define collection name
COLLECTION_NAME = "knowledge_base_collection"

# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize or get the Collection
if COLLECTION_NAME not in [col.name for col in chroma_client.list_collections()]:
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
else:
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

# Initialize Session State
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ""
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = "llama-3.1-70b-versatile"
if 'system_prompt' not in st.session_state:
    st.session_state['system_prompt'] = """You are a knowledgeable assistant with access to a document database. 
    Use the provided context to answer questions accurately. If the context doesn't contain relevant information, 
    acknowledge this and provide a general response based on your knowledge."""

def initialize_groq_client() -> Optional[Groq]:
    """Initialize Groq client with API key"""
    api_key = st.session_state.get('api_key', "")
    if not api_key:
        st.warning("Please enter your Groq API Key in the Settings tab.")
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return None

def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file with versioning"""
    filename = uploaded_file.name
    name, ext = os.path.splitext(filename)
    today = datetime.datetime.now().strftime("%Y%m%d")
    version = 1
    while True:
        new_filename = f"{name}_{today}_v{version}{ext}"
        file_path = DOCUMENTS_DIR / new_filename
        if not file_path.exists():
            break
        version += 1
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def save_image_data(image_path: Path, img: Image) -> str:
    """Save image data and return reference path"""
    image_id = f"{image_path.stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    data_path = IMAGE_DATA_DIR / f"{image_id}.png"
    img.save(data_path)
    return str(data_path)

def process_file_content(file_path: Path) -> List[Dict]:
    """Process file content and return chunks with metadata"""
    chunks = []
    try:
        if file_path.suffix.lower() in ['.txt', '.csv', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                text_chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        "id": f"{file_path.stem}_chunk_{i}",
                        "content": chunk,
                        "metadata": {
                            "source": file_path.name,
                            "chunk": i,
                            "type": "text"
                        }
                    })
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content = json.dumps(data, indent=2)
                chunks.append({
                    "id": f"{file_path.stem}_json",
                    "content": content,
                    "metadata": {
                        "source": file_path.name,
                        "type": "json"
                    }
                })
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            img = Image.open(file_path)
            image_data_path = save_image_data(file_path, img)
            description = f"An image showing {file_path.stem.replace('_', ' ')}"
            chunks.append({
                "id": f"{file_path.stem}_image",
                "content": description,
                "metadata": {
                    "source": file_path.name,
                    "type": "image",
                    "image_data_path": image_data_path,
                    "description": description
                }
            })
    except Exception as e:
        st.error(f"Error processing {file_path.name}: {str(e)}")
        return []
    return chunks

def add_to_vector_store(chunks: List[Dict]):
    """Add processed chunks to ChromaDB"""
    if not chunks:
        return
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks]
    )

def vector_search(query: str, n_results: int = 5) -> tuple:
    """Search the vector database for relevant content"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    return results['documents'][0], results['metadatas'][0], results['distances'][0]

def check_image_relevance(query: str, metadata: Dict) -> float:
    """Check image relevance using CLIP embeddings"""
    try:
        if metadata.get('type') == 'image':
            inputs = processor(text=[query], return_tensors="pt", padding=True)
            query_embedding = clip_model.get_text_features(**inputs).detach().numpy()
            
            image_description = metadata.get('description', '') or f"An image showing {Path(metadata['source']).stem}"
            inputs = processor(text=[image_description], return_tensors="pt", padding=True)
            image_embedding = clip_model.get_text_features(**inputs).detach().numpy()
            
            similarity = cosine_similarity(query_embedding, image_embedding)[0][0]
            return float(similarity)
    except Exception as e:
        st.error(f"Error calculating image relevance: {e}")
    return 0.0

def get_image_data(metadata: Dict) -> Optional[Dict]:
    """Get image data from stored file"""
    try:
        if 'image_data_path' in metadata:
            image_path = metadata['image_data_path']
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format=img.format or 'PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    img_base64 = base64.b64encode(img_byte_arr).decode()
                    
                    return {
                        "filename": Path(image_path).name,
                        "size": f"{img.size[0]}x{img.size[1]}",
                        "format": img.format or 'PNG',
                        "mode": img.mode,
                        "base64_data": img_base64,
                        "description": metadata.get('description', '')
                    }
    except Exception as e:
        st.error(f"Error retrieving image data: {e}")
    return None

def display_image_from_metadata(metadata: Dict) -> bool:
    """Display an image from its metadata"""
    try:
        image_data = get_image_data(metadata)
        if image_data and 'base64_data' in image_data:
            img_data = base64.b64decode(image_data['base64_data'])
            img = Image.open(BytesIO(img_data))
            st.image(img, caption=image_data.get('description', ''), use_column_width=True)
            return True
    except Exception as e:
        st.error(f"Error displaying image: {e}")
    return False

def image_search_tab():
    """Image search interface showing only the most relevant image"""
    st.header("🖼️ Image Search")
    
    with st.container():
        query = st.text_input("🔍 Search for images:", 
                            placeholder="Describe what you're looking for...",
                            help="Enter keywords or descriptions to find relevant images")

    if query:
        try:
            # Get relevant documents and find the most relevant image
            docs, metadatas, distances = vector_search(query)
            best_image = None
            best_relevance = 0
            
            for metadata in metadatas:
                if metadata['type'] == 'image':
                    relevance = check_image_relevance(query, metadata)
                    if relevance > best_relevance:
                        best_relevance = relevance
                        best_image = metadata
            
            # Only show the image if it's relevant enough
            if best_image and best_relevance > 0.5:
                st.markdown("### 📸 Most Relevant Image")
                st.markdown(f"**Relevance Score: {best_relevance:.2f}**")
                with st.container():
                    display_image_from_metadata(best_image)
                    if 'description' in best_image:
                        st.caption(best_image['description'])
            else:
                st.info("No relevant images found. Try a different search query.")
                
        except Exception as e:
            st.error(f"Error during image search: {str(e)}")

def handle_chat(query: str):
    """Handle chat interactions with text content only"""
    if not query.strip():
        return

    groq_client = initialize_groq_client()
    if not groq_client:
        return

    try:
        # Get relevant context from documents
        docs, metadatas, distances = vector_search(query)
        
        # Filter for text content only
        text_context = []
        text_sources = []
        
        for doc, metadata, distance in zip(docs, metadatas, distances):
            if metadata['type'] != 'image':
                text_context.append(doc)
                text_sources.append({
                    'source': metadata['source'],
                    'type': metadata['type'],
                    'content': doc
                })
        
        # Prepare context for the LLM
        context = "\n\nRelevant information from documents:\n" + "\n".join(text_context) if text_context else ""

        # Get response from LLM
        messages = [
            {"role": "system", "content": st.session_state['system_prompt']},
            {"role": "user", "content": f"{query}\n\n{context}"}
        ]
        
        response = groq_client.chat.completions.create(
            messages=messages,
            model=st.session_state['model_name'],
            temperature=0.7,
            max_tokens=1000,
        )
        answer = response.choices[0].message.content

        # Display chat messages
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            st.markdown(answer)
            if text_sources:
                with st.expander("📚 Sources"):
                    for source in text_sources:
                        st.markdown(f"**{source['source']}**")
                        st.text(source['content'][:200] + "..." if len(source['content']) > 200 else source['content'])

        # Update chat history
        st.session_state['chat_history'].append({
            "role": "user",
            "content": query
        })
        st.session_state['chat_history'].append({
            "role": "assistant",
            "content": answer,
            "sources": text_sources
        })

    except Exception as e:
        st.error(f"Error in chat: {str(e)}")

def chat_tab():
    """Text-based chat interface"""
    st.header("💬 Knowledge Base Chat")

    # Display chat history
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['source']}**")
                        st.text(source['content'][:200] + "..." if len(source['content']) > 200 else source['content'])

    # Chat input
    if query := st.chat_input("💭 Ask me anything about your documents..."):
        handle_chat(query)

def documents_tab():
    """Documents management interface"""
    st.header("📚 Document Management")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["txt", "csv", "json", "png", "jpg", "jpeg", "md"],
            accept_multiple_files=True,
            help="Upload your documents to build the knowledge base"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                existing_files = [f.stem.split('_')[0] for f in DOCUMENTS_DIR.glob("*.*")]
                base_name = os.path.splitext(uploaded_file.name)[0]
                
                if base_name not in existing_files:
                    file_path = save_uploaded_file(uploaded_file)
                    chunks = process_file_content(file_path)
                    add_to_vector_store(chunks)
                    st.success(f"✅ Successfully processed {uploaded_file.name}")
                else:
                    st.info(f"📝 {uploaded_file.name} already exists in the knowledge base")

    with col2:
        if st.button("🗑️ Clear All Documents", type="secondary"):
            try:
                # Delete collection
                chroma_client.delete_collection(name=COLLECTION_NAME)
                # Create new collection
                embedding_function = embedding_functions.DefaultEmbeddingFunction()
                collection = chroma_client.create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                # Delete files
                for file in DOCUMENTS_DIR.glob("*"):
                    file.unlink()
                # Delete image data
                for file in IMAGE_DATA_DIR.glob("*"):
                    file.unlink()
                st.success("✅ All documents cleared successfully")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing documents: {e}")

    # Document preview
    st.subheader("📄 Uploaded Documents")
    documents = sorted(DOCUMENTS_DIR.glob("*.*"))
    if documents:
        for doc in documents:
            with st.expander(f"📄 {doc.name}"):
                if doc.suffix.lower() in ['.txt', '.csv', '.json', '.md']:
                    with open(doc, 'r', encoding='utf-8') as f:
                        st.code(f.read(), language='plain')
                elif doc.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    st.image(str(doc), use_column_width=True)
    else:
        st.info("No documents uploaded yet. Upload some documents to get started!")

def settings_tab():
    """Settings interface"""
    st.header("⚙️ Settings")

    with st.form("settings_form"):
        st.subheader("🤖 Model Configuration")
        
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state['api_key'],
            help="Enter your Groq API key to enable the chatbot"
        )
        
        # Updated model selection with correct Groq model names
        model_name = st.selectbox(
            "Model",
            [
                "llama-3.1-70b-versatile",           
                # "mixtral-8x7b-chat",         # Mixtral model
                # "gemma-7b-it",               # Gemma model
               
            ],
            index=0,
            help="Select the model to use for generating responses"
        )
        
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state['system_prompt'],
            height=100,
            help="Customize the AI assistant's behavior"
        )

        if st.form_submit_button("💾 Save Settings"):
            st.session_state['api_key'] = api_key
            st.session_state['model_name'] = model_name
            st.session_state['system_prompt'] = system_prompt
            st.success("✅ Settings saved successfully")

    if st.button("🗑️ Clear Chat History"):
        st.session_state['chat_history'] = []
        st.success("✅ Chat history cleared")
        st.rerun()

def main():
    """Main application"""
    st.title("🤖 Knowledge Base Assistant")
    
    # Tab-based navigation
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🖼️ Image Search", "📚 Documents", "⚙️ Settings"])
    
    with tab1:
        chat_tab()
    
    with tab2:
        image_search_tab()
    
    with tab3:
        documents_tab()
    
    with tab4:
        settings_tab()

if __name__ == "__main__":
    main()
