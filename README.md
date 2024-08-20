# Advanced-RAG-Pipeline

The Advanced RAG Pipeline is a cutting-edge tool designed to support researchers by efficiently retrieving and generating precise information from various sources. Leveraging modern libraries like Langchain and Hugging Face Transformers, it enhances research workflows with high accuracy and scalability.

### Workflow

1. **PDF and Document Handling:**
   - The pipeline begins by loading and processing documents from different formats, such as PDFs, DOCX, and TXT files. This is done using `PyPDFLoader`, `Docx2txtLoader`, and `TextLoader` from the LangChain library.
   - PDF files are converted to images using the `pdf2image` library for further processing.

2. **Table Detection and Extraction:**
   - The `TableTransformerForObjectDetection` model from the `transformers` library detects tables in the images converted from PDF pages.
   - The `PaddleOCR` and `PPStructure` libraries are employed to extract text and table structures from the images.

3. **Table and Header Processing:**
   - Detected tables and headers are processed and stored using a custom `CSVHashMap` class. Headers are extracted using custom logic that adjusts bounding boxes to crop relevant parts of the images.

4. **Web Scraping and Context Retrieval:**
   - For context retrieval, web pages are scraped using `requests` and `BeautifulSoup` if the information is not found in local documents.
   - DuckDuckGo search is used to find web contexts, and the most similar context is identified using cosine distance calculations on embeddings.

5. **Embedding and Vector Database:**
   - Text embeddings are generated using the `HuggingFaceEmbeddings` class with the `sentence-transformers/all-MiniLM-L6-v2` model.
   - A Chroma vector store is used to store and retrieve document chunks efficiently.

6. **Query Handling and Response Generation:**
   - Queries are improved by removing common question phrases and summarized using a pre-trained language model.
   - A `ChatModel` class handles the interaction between the user query, the retrieval process, and the language model.
   - The language model, `meta-llama/Meta-Llama-3-8B-Instruct`, is used to generate responses based on retrieved contexts and user input.

7. **Data Chunking and Vector Store Creation:**
   - Documents are split into manageable chunks using the `CharacterTextSplitter` for efficient processing and retrieval.

### Libraries and Dependencies

1. **Image and PDF Processing:**
   - `pdf2image`: Converts PDF pages into images.
   - `PIL` (Pillow): Provides image handling and processing capabilities.
   - `cv2` (OpenCV): Utilized for image manipulation and table extraction.

2. **Natural Language Processing:**
   - `transformers`: Used for table detection with the `TableTransformerForObjectDetection` and language models.
   - `HuggingFaceEmbeddings`: Generates sentence embeddings for similarity comparison.
   - `langchain`: Facilitates document loading and chunking.
   - `langchain_community`: Provides community-contributed implementations for embeddings and vector stores.

3. **Data Handling:**
   - `pandas`: Provides DataFrame structures for handling tabular data.
   - `openpyxl`: Used for Excel file manipulation and saving images.
   - `numpy`: Supports numerical calculations and vector operations.

4. **Web Scraping and Search:**
   - `requests`: Handles HTTP requests to fetch web pages.
   - `BeautifulSoup`: Parses HTML content to extract relevant data.
   - `duckduckgo_search`: Conducts web searches to retrieve contextual information.

5. **OCR and Table Extraction:**
   - `PaddleOCR` and `PPStructure`: Extract text and table structures from images.

6. **General Utilities:**
   - `os`: Handles file and directory operations.
   - `torch`: Supports PyTorch operations and tensor handling for models.

7. **Environment Settings:**
   - `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'`: Suppresses KMP duplicate library warning.
   - `os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'`: Disables Hugging Face cache symlink warnings.

### Files

1. **`utils_final.py`:**
   - Contains utility functions and classes used throughout the pipeline.

2. **`final_app.py`:**
   - Contains the frontend code to interact with the pipeline.

### Usage Instructions


1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Credit Risk Analysis using ML algorithms.git
   cd Credit Risk Analysis using ML algorithms
   ```

2. **Configuration:**
   - Set up environment variables as needed in the `.env` file.
   - Configure API keys and other credentials required by the libraries.

3. **Document Upload:**
   - Before running the pipeline, Create a folder inside your virtual environment and then upload your document in the given folder. 

4. **Running the Pipeline:**
   - Execute the main script: `python final_app.py`
   - Provide necessary input files and configuration settings as specified in the documentation.

### Contributing

Contributions to the Advanced RAG Pipeline project are welcome! Please follow these guidelines:
   - Fork the repository and create a feature branch.
   - Commit your changes with clear messages.
   - Open a pull request and provide a description of your changes.

### License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

### Acknowledgements
- Dr. Sukanta Halder, Assistant Professor, Department of Electrical Engineering, I.I.T (I.S.M), Dhanbaad.
- Mr. Sudheer Kumar, Research Scholar, S.V.N.I.T, Surat.

### Test Results

We have rigorously tested the Advanced RAG Pipeline to ensure its accuracy and reliability. Our testing involved multiple questions, and the model correctly answered with the accuracy of 94.3% in, demonstrating high performance.


For detailed test results, including the questions and answers, please refer to the [Chatchain Test Results PDF](docs/ChatChain.pdf).



...

### Contact
### Contact

For any questions or issues, please contact tanmaychaturvedimgs@gmail.com or open an issue on the GitHub repository.
