# Knowledge Base Assistant

## Project Purpose

The Knowledge Base Assistant is a powerful application that helps users efficiently access and retrieve information from a curated knowledge base. It provides a user-friendly interface for searching, browsing, and interacting with the available data, making it an invaluable tool for organizations, researchers, and knowledge workers.

The key features of the Knowledge Base Assistant include:

1. **Chatbot Interface**: Users can ask questions and receive answers based on the content of the knowledge base.
2. **Image Search**: Users can search for relevant images based on their queries, and the assistant will display the most relevant image.
3. **Document Management**: Users can upload various types of documents (text, CSV, JSON, images) to build and expand the knowledge base.
4. **Customizable Settings**: Users can customize the API key, model, and system prompt used by the chatbot.

## How it Works

The Knowledge Base Assistant uses ChromaDB, a vector database, to store and retrieve the content from the uploaded documents. Here's a high-level overview of the application's workflow:

1. **Document Ingestion**: When users upload documents to the application, the content is processed and split into smaller chunks (1000 characters or less). These chunks are then stored in the ChromaDB vector store along with their metadata (e.g., source file name, chunk index, content type).

2. **Query Processing**: When a user asks a question or performs a search, the application generates embeddings for the query using a pre-trained language model (CLIP in this case). These embeddings are then used to perform a nearest-neighbor search in the ChromaDB vector store to find the most relevant chunks of text or image data.

3. **Response Generation**: The relevant text content is then used to generate a response to the user's query using a large language model (LLM) provided by the Groq API. For image searches, the application calculates the relevance of each image based on the CLIP embeddings and displays the most relevant one.

4. **Personalization and Customization**: Users can customize the application's settings, such as the Groq API key, the language model used, and the system prompt that defines the assistant's behavior.

## Installation and Setup

To set up the Knowledge Base Assistant application, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/Biku213/Knowledge-Base-Assistant
   cd Knowledge-Base-Assistant
   ```

2. Create a virtual environment (if not already created) and activate it:

- For Linux/Mac:

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

- For Windows:

   ```
   python3 -m venv venv
   venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Obtain a Groq API key and update the `st.session_state['api_key']` value in the `settings_tab()` function.

5. Run the Streamlit application:

   ```
   streamlit run app.py
   ```

6. The application will open in your default web browser. You can now start uploading documents, customizing settings, and using the chat and image search functionality.

## Integration with ChromaDB

The Knowledge Base Assistant uses ChromaDB, a vector database, to store and retrieve the content from the uploaded documents. Here's how ChromaDB is integrated into the application:

1. **Document Indexing**: When a user uploads a document, the content is processed and split into smaller chunks (1000 characters or less). These chunks are then stored in the ChromaDB vector store along with their metadata (e.g., source file name, chunk index, content type).

2. **Vector Search**: When a user asks a question or performs a search, the application generates embeddings for the query using a pre-trained language model (CLIP in this case). These embeddings are then used to perform a nearest-neighbor search in the ChromaDB vector store to find the most relevant chunks of text or image data.

3. **Optimized Performance**: The application leverages the `hnsw:space` metadata in ChromaDB to use the Hierarchical Navigable Small World (HNSW) graph-based index, which provides a significant speed boost for nearest-neighbor searches. Additionally, the application implements a chunking strategy to reduce the overall memory usage of the vector store.

4. **Persistence**: The ChromaDB vector store is persisted to disk, allowing the application to maintain the knowledge base even after restarts or deployments.

## Screenshots

Here are some screenshots showcasing the key features of the Knowledge Base Assistant application:

![Chat Interface](https://github.com/user-attachments/assets/744764dc-5f65-4a69-b281-fcc7e92be3c8)

![Image Search](https://github.com/user-attachments/assets/14e9e01b-2e64-45cc-a6f9-645042b862fe)

![Document Management](https://github.com/user-attachments/assets/0a4af41b-837e-411b-992f-9cdf22e86be1)

![Settings Config](https://github.com/user-attachments/assets/03332a39-42c7-4900-ab82-86eb229af19e)

## Demo Video

https://github.com/user-attachments/assets/d4d904d3-bce9-4ce0-ab1c-391acf383d71

## Future Improvements

- Implement support for additional file types (e.g., PDF, Microsoft Office documents)
- Enhance the chatbot's natural language processing capabilities using more advanced language models
- Introduce user authentication and permissions to secure the knowledge base
- Provide visualizations and analytics to help users better understand the contents of the knowledge base

## Conclusion

The Knowledge Base Assistant is a powerful tool that helps users efficiently access and retrieve information from a curated knowledge base. By leveraging the capabilities of ChromaDB, Groq, and various language models, the application delivers fast and accurate responses, making it a valuable asset for organizations, researchers, and knowledge workers. With its user-friendly interface, robust performance, and continuous improvements, the Knowledge Base Assistant is poised to become an indispensable tool for knowledge management and information retrieval.
