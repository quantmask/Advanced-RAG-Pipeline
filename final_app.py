from flask import Flask, render_template, request, jsonify, redirect, url_for
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient, login
from utils_final import load_data,dataChunking,create_and_persist_vectordb,ChatChain,ChatModel
from langchain.docstore.document import Document

app = Flask(__name__)

# Initialize and login to Hugging Face
login(token='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')
client = InferenceClient()

# Load documents and create vector database
data = []
documents, tables_list = load_data(data)
if documents != []:
    document_chunks = dataChunking(documents)
    for i in range(len(tables_list)):
        document_chunks.append(Document(page_content = tables_list[i]))
    vectordb = create_and_persist_vectordb(document_chunks)
else:
    vectordb = None

model = 'meta-llama/Meta-Llama-3-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model)
system_msg = tokenizer.apply_chat_template([{"role" : "system", "content" : "You are an AI Assistant that gives correct answers with respect to the given context"}], tokenize = False)
chat_chain = ChatChain([system_msg])
chat_model = ChatModel(model = model, client = client, vectordb = vectordb,chatChain = chat_chain)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        # Process the login information here
        # For now, you can just print the details or redirect to a welcome page
        print(f"Name: {name}, Email: {email}, Phone: {phone}")
        return redirect(url_for('home'))  # Redirect to home page after login
    return render_template('login.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Invalid request. Message field is required.'}), 400

        user_message = data['message']
        ai_response = interact_with_chatbot(user_message, chat_chain)
        return jsonify({'response': ai_response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def interact_with_chatbot(message,chat_chain):
    # Invoke the chatbot
    output = chat_model.invoke(message, stream= True)
    return output

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
