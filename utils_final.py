import os
from pdf2image import convert_from_path
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image
import pandas as pd
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from duckduckgo_search import DDGS
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
import cv2
from paddleocr import PPStructure
from paddleocr import PaddleOCR
from openpyxl import load_workbook, Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl import load_workbook, Workbook
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Suppress HF cache symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load the pre-trained table detection model
image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")



def cosine_distance(vector1, vector2):
    return 1 - (np.dot(vector1, vector2)) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def find_contexts_from_web(msg):
    results = DDGS().text(keywords=msg,
                          region='wt-wt',
                          safesearch='off',
                          timelimit='1d',
                          max_results=10)
    body_entries = [entry['body'] for entry in results]
    return body_entries

def find_most_similar_context_from_web(msg, body_entries):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    embedded_query = embeddings.embed_query(msg)
    embedded_contexts = [embeddings.embed_query(entry) for entry in body_entries]
    scores = [cosine_distance(embedded_query, entry) for entry in embedded_contexts]
    minIndex = scores.index(min(scores))
    return body_entries[minIndex]

def fetch_and_process(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f'Error: Unable to fetch data from the provided URL. Exception: {e}'
    
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find(id='mw-content-text')
    
    if content:
        paragraphs = content.find_all('p')
        detailed_info = '\n'.join([p.text.strip() for p in paragraphs])
        return detailed_info
    else:
        return 'Error: Unable to locate the main content on the page.'


def update_box_header(box, image_path):
    xmin,ymin,xmax,ymax = box
    
    image = Image.open(image_path).convert("RGB")
    page_width = image.size[0]
    page_height = image.size[1]   
    if(xmax < page_width / 2 and xmin < page_width / 2):
        new_xmax = page_width / 2
        new_xmin = 0
        new_ymax = ymin
        new_ymin = ymin - 380 #(this number is temporary, use a proper formula for this.)
        # Update the bounding box with new ymin and ymax
        new_box = [new_xmin, new_ymin, new_xmax, new_ymax]
        new_box = [round(coord, 2) for coord in new_box]  # Round coordinates for consistency
    elif(xmin > page_width/2 and xmax<page_width):
        new_xmax = page_width
        new_xmin = page_width/2
        new_ymax = ymin
        new_ymin = ymin - 380
        # Update the bounding box with new ymin and ymax
        new_box = [new_xmin, new_ymin, new_xmax, new_ymax]
        new_box = [round(coord, 2) for coord in new_box]  # Round coordinates for consistency
    elif(xmin < page_width/2 and xmax > page_width/2):
        new_xmax = page_width
        new_xmin = 0
        new_ymax = ymin
        new_ymin = ymin - 380
        # Update the bounding box with new ymin and ymax
        new_box = [new_xmin, new_ymin, new_xmax, new_ymax]
        new_box = [round(coord, 2) for coord in new_box]  # Round coordinates for consistency
        
    return new_box

def create_images():
    for file in os.listdir("dataset_llm"):
        print(f"Processing file: {file}")
        if file.endswith(".pdf"):
            pdf_path = os.path.join("dataset_llm", file)
            print(f"Loading PDF: {pdf_path}")

            # Convert the PDF to images (one image per page) at 500 DPI
            images = convert_from_path(pdf_path, dpi=500, poppler_path=r'C:\Program Files (x86)\poppler-23.11.0\Library\bin')

            # Ensure the directory to store the converted images exists
            os.makedirs("pages", exist_ok = True)
    numberOfPages = len(images)
    i, j, k = 0, 0, 0
    for l in range(numberOfPages):
        if j % 10 == 0 and j != 0:
            k += 1
            j = 0
        if i % 10 == 0 and i != 0:
            j += 1
            i = 0
        images[l].save('pages/' + str(k) + str(j) + str(i) + '.jpg', 'JPEG')
        i += 1

def detect_table_and_header_images():
    folder_path = "pages"
    number_of_pages = len(os.listdir(folder_path))

    # Initialize the image processor and model once outside the loop
    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    box_list = []

    for i, file in zip(range(number_of_pages), os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)
        image = Image.open(file_path).convert("RGB")

        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        processed_folder_path = "processed_images_of_tables"
        if not os.path.exists(processed_folder_path):
            os.makedirs(processed_folder_path)

        processed_folder_path_headers = "processed_images_of_headers"
        if not os.path.exists(processed_folder_path_headers):
            os.makedirs(processed_folder_path_headers)

        flag = None

        numberOfBoxes = len(results["boxes"])

        for j, box in zip(range(numberOfBoxes), sorted(results["boxes"], key = lambda box : (box[0], box[1]))):
            print(file)
            print(results["boxes"])
            box = [round(i, 2) for i in box.tolist()]
            if box not in box_list:
                box_list.append(box)
            print(f"Score: {results['scores'][j]}")
            print(f"Label: {results['labels'][j]}")
            print(f"Box: {box}")
            if box != None:
                flag = True
            
            if flag == None:
                continue
            header_box = update_box_header(box, file_path)
            table_image = image.crop(box)
            box_image = image.crop(header_box)
            table_image.save(f'processed_images_of_tables/processed_image_{i + 1}_{j + 1}.jpg', 'JPEG')
            box_image.save(f'processed_images_of_headers/processed_image_of_header_{i + 1}_{j + 1}.jpg', 'JPEG')
            j += 1
        i += 1  

def extract_headers():
    headers_list = []
    folder_path = 'processed_images_of_headers'
    # Use PaddleOCR to extract text from the table image
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    print("Initialized PaddleOCR.")
    
    for header in os.listdir(folder_path):
        file_path = os.path.join(folder_path, header)
        header_text = ocr.ocr(file_path, cls=True)
        extracted_text = ' '.join([line[1][0] for line in header_text[0]])
        headers_list.append(extracted_text)

    return headers_list


def extract_tables_from_images():
    tables_list = []
    folder_path = 'processed_images_of_tables'
    # Initialize PPStructure for table extraction with recovery and OCR results
    table_engine = PPStructure(recovery=True, return_ocr_result_in_table=True)
    
    df = None
    
    for image in os.listdir(folder_path):
        # Process images in a loop
        for n in range(1, 5):
            print('image', n)
            img_path = os.path.join(folder_path, image)
            img = cv2.imread(img_path)
            result = table_engine(img)

            # Create an mage object for openpyxl
            xlimg = XLImage(img_path)
            i = 1
            for line in result:
                # Remove the 'img' key from the result
                line.pop('img')
                # Check if the line is a table
                if line.get("type") == "table":
                    # Extract HTML table and convert to DataFrame
                    html_table = line.get("res").get("html")
                    html_data = pd.read_html(html_table)
                    df = pd.DataFrame(html_data[0])
                    csv_string = df.to_string(index=False)
        tables_list.append(csv_string)
    
    return tables_list

def update_tables_list(headers_list, tables_list):
    for i in range(len(tables_list)):
        tables_list[i] = headers_list[i] + "\n\n" + tables_list[i]

    return tables_list

# Load data from PDF or URLs
#pass an empty list to load all the data into.
def load_data(document):
    print("Starting to load data...")
    str = ''
    if len(os.listdir("URLs")) != 0:
        folder_path = 'URLs'
        file_name = 'URLs.txt'
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            urls = file.readlines()
            for url in urls:
                url = url.strip()
                if url: 
                    print(f"Processing URL: {url}")
                    document.append(Document(page_content = fetch_and_process(url)))              
                            
    for file in os.listdir("dataset_llm"):
        print(f"Processing file: {file}")
        if file.endswith(".pdf"):
            pdf_path = "./dataset_llm/" + file
            print(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            document.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = "./dataset_llm/" + file
            print(f"Loading DOCX/DOC: {doc_path}")
            loader = Docx2txtLoader(doc_path)
            document.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = "./dataset_llm/" + file
            print(f"Loading TXT: {text_path}")
            loader = TextLoader(text_path)
            document.extend(loader.load())
    
    print("Processing tables from PDFs")
    create_images()
    detect_table_and_header_images()
    headers_list = extract_headers()
    tables_list = extract_tables_from_images()
    updated_tables_list = update_tables_list(headers_list = headers_list, tables_list = tables_list)

    print("Finished loading tables.")
    print("Finished loading data.")
    return document, updated_tables_list       

# Create document chunks
def dataChunking(document):
    document_chunks = []
    document_splitter = CharacterTextSplitter(separator='\n', chunk_size=600, chunk_overlap=480)

    for doc in document:
        chunks = document_splitter.split_documents([doc])
        document_chunks.extend(chunks)

    return document_chunks

def create_and_persist_vectordb(document_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectordb = Chroma.from_documents(document_chunks, embedding=embeddings, persist_directory='./data')
    return vectordb

class ChatChain:
    def __init__(self, chain):
        self.chain = chain

    def __str__(self):
        return '\n'.join(tuple(map(str, self.chain)))

    def __repr__(self):
        return '\n'.join(tuple(map(repr, self.chain)))

    def generate_prompt(self):
        return f'{self.__repr__()} \n <|assistant|> \n'

class ChatModel:
    def __init__(self, model, vectordb = None, client = None, chatChain = None):
        self.client = client
        self.model = model
        self.vectordb = vectordb
        self.chatChain = chatChain

    def tokenize(self, role, msg):
        self.role = role
        self.msg = msg
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        template = self.tokenizer.apply_chat_template([{"role" : f"{self.role}", "content" : f"{self.msg}"}], tokenize = False)
        return template

    def improveQuery(self, msg):
        wh_list = ["Who are", "Who is", "Who do", "Who did", "Who made", "Who are the", "Who is the", 
           "Who invented", "Who discovered", "Who invented the", "Who discovered the", "Who is the",
           "Who made the", "Who are the", "Who was", "Who was the", "What is", "What are", "What is the",
           "What are the", "What's the", "What do", "What does", "What do the",  "What does the",
           "When is", "When is the", "When was", "When was the", "When did", "When did the",
           "When does", "When does the", "When do", "When do the", "When should", "When should the",
           "When will", "When will the", "Where is", "When is the", "When are", "Where does", "When does the",
           "Where are", "When did", "When did the", "Where were", "Where can", "Where will", "Why are",
           "Why is", "Why is the", "Why do", "Why does the", "Why does", "Which do", "Which of", "Which are", 
           "Which is", "Which type", "How are", "How is", "How did", "How many", "How does", "How do",
           "How can","How can we", "How", "Which", "Why", "Where", "When", "What", "Who", "How can we represent",
           "Tell me something about", 
           "Can you describe", "Explain in detail", "Explain breifly", "Explain", "Can you tell me something about",
           "Can you explain what is", "Can you tell", "Can you provide", "Can", "Please", "Write an", 
           "Write an overview on", "Write about", "Summarize", "Summarize the", "Provide some information about",
           "Provide some information on"]
        improvedQuery = msg 
        for sString in wh_list :
            if sString in msg:
                improvedQuery = msg.replace(sString, '')
                break
        if '?' in msg:
            improvedQuery = improvedQuery.replace('?', '')
        if '.' in msg:
            improvedQuery = improvedQuery.replace('.', '')
        return improvedQuery

    def extractiveSummarize(self, improvedQuery, msg):
        prompt = self.tokenize("system", "You are an expert summarizer which writes a summary of the text given by the user .Return your response which covers the key points of the text and use related words to write an extractive summary.")
        queryChain = ChatChain([prompt])
        contexted_query = "Summarize the following text : \n" + msg
        humanMessage = self.tokenize("user", contexted_query)
        queryChain.chain.append(humanMessage)

        summarizedQuery = ''
        for token in self.client.text_generation(prompt = queryChain.generate_prompt(), model = self.model, max_new_tokens = 256, stream = False):
            if token == '':
                break
            summarizedQuery += token
        aiMessage = self.tokenize("assistant", summarizedQuery)
        queryChain.chain.append(aiMessage)
        return summarizedQuery

    def invoke(self, msg, stream=False):
        # Get the improved query
        improvedQuery = self.improveQuery(msg)
        summarizedQuery = self.extractiveSummarize(improvedQuery, msg)
        
        context = ''

        if self.vectordb is None:
            # Perform the search when no relevant document is found
            body_entries = find_contexts_from_web(summarizedQuery)
            context += find_most_similar_context_from_web(summarizedQuery, body_entries)
        
        else:
            similar_contexts = self.vectordb.similarity_search_with_score(summarizedQuery)
            context += similar_contexts[0][0].page_content + " " + similar_contexts[1][0].page_content
            
        # Append human message with context
        contexted_msg = f"{msg}\nUse the context given below to answer the question:\n{context}"
        human_message = self.tokenize("user", contexted_msg)
        self.chatChain.chain.append(human_message)

        output = ''
        for token in self.client.text_generation(prompt=self.chatChain.generate_prompt(), model=self.model, max_new_tokens=512, stream=stream):
            if token == '':
                break
            output += token
        ai_message = self.tokenize("assistant", output)
        self.chatChain.chain.append(ai_message)
        return output