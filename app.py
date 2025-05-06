# app.py
from flask import Flask, render_template, request, jsonify, session
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()  # Generate a secure random key

# Load environment variables
load_dotenv()

CORS(app)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize MongoDB loader
loader = MongodbLoader(
    connection_string="mongodb+srv://deshcode0:helloworld@loan0.1ydti.mongodb.net/?retryWrites=true&w=majority&appName=loan0",
    db_name="loan_database",
    collection_name="loan_system",
    field_names=[
        "_id", "customer_id", "first_name", "last_name", "date_of_birth",
        "address", "phone", "email", "national_id", "loan_id", "loan_type",
        "principal_amount", "interest_rate", "loan_term_months", "start_date",
        "end_date", "loan_status", "collaterals", "repayments",
        "payment_schedule", "loan_application", "assigned_employee", "branch"
    ]
)

# Load documents and create vector store
all_docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
splits = text_splitter.split_documents(all_docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

document_cont = ""
for i in range(len(all_docs)):
    document_cont += all_docs[i].page_content + "\n"

# Create prompt template
system_prompt = """
You are a loan management assistant chatbot. Based on the customer and loan data provided and our conversation history, respond to user queries effectively.

Previous conversation:
{chat_history}

Current context:
{context}

RESPONSE STRUCTURE:
1. Always start with a brief, direct answer to the query.
2. When providing loan or customer information, use this format:

   📋 CUSTOMER DETAILS (if applicable):
   • Name: [First Name Last Name]
   • ID: [Customer ID]
   • Contact: [Phone/Email]

   💰 LOAN INFORMATION (if applicable):
   • Loan ID: [ID]
   • Type: [Loan Type]
   • Amount: $[Amount]
   • Status: [Status]
   • Term: [Duration] months

   💳 PAYMENT INFORMATION (if applicable):
   • Due Amount: $[Amount]
   • Next Payment: [Date]
   • Payment Status: [Status]

   👤 ASSIGNED STAFF (if applicable):
   • Name: [Name]
   • Branch: [Branch Name]

3. For repayment histories or schedules, use:
   
   📅 PAYMENT SCHEDULE:
   • [Date]: $[Amount] - [Status]
   • [Date]: $[Amount] - [Status]

GUIDELINES:
- Use clear formatting with bullet points and sections
- Include only relevant sections based on the query
- Keep responses concise but informative
- Use emojis as section markers for better readability
- If certain information is not available, indicate "Not Available" rather than omitting the field
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Create chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query', '')
    if not user_query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        # Get chat history from session
        chat_history = session.get('chat_history', [])
        
        # Add the new query to memory
        memory.chat_memory.add_user_message(user_query)
        
        # Generate response
        response = rag_chain.invoke({
            "input": user_query + "Here's the mongodb data" + document_cont,
            "chat_history": memory.chat_memory.messages
        })
        
        formatted_response = format_response(response['answer'])
        
        # Add response to memory
        memory.chat_memory.add_ai_message(formatted_response)
        
        # Update chat history in session
        chat_history.append({
            'type': 'user',
            'message': user_query
        })
        chat_history.append({
            'type': 'bot',
            'message': formatted_response
        })
        session['chat_history'] = chat_history
        
        return jsonify({
            'response': formatted_response,
            'chat_history': chat_history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def format_response(response_text):
    import re
    formatted_text = re.sub(r"\*\*", "", response_text)
    formatted_text = formatted_text.replace("\\n", "\n")
    return formatted_text

if __name__ == '__main__':
    app.run(debug=True)

