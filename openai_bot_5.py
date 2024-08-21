from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import atexit
import mysql.connector as mysql
import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import re
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from datetime import datetime, timedelta

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a random secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize SendGrid API key
sg_api_key = os.environ.get("SENDGRID_API_KEY")

# MySQL Database Configuration for Conversations and Appointments
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'rishil10',  # change passwords
    'database': 'cloudjunebot',  # the database cloudjunebot has been referenced in the other scripts, not cloudjune
}

db_connection = mysql.connect(**db_config)

oai_api_key = os.environ.get("OPENAI_API_KEY")

# Define SendGrid email sending function
def send_email(user_message, bot_response):
    message = Mail(
        from_email='support@cloudjune.com',
        to_emails='marketing@cloudjune.com',  # Update with recipient email
        subject='User Enquiry',
        html_content=f'<p>User Message: {user_message}</p><p>Bot Response: {bot_response}</p>'
    )

    try:
        sg = SendGridAPIClient(sg_api_key)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(str(e))

@app.route('/')
def home():
    return render_template('base.html')

# Function to check available appointment slots
def get_available_slots():
    cursor = db_connection.cursor()
    cursor.execute("SELECT appointment_time FROM appointments")
    booked_slots = [row[0] for row in cursor.fetchall()]
    cursor.close()

    available_slots = []
    now = datetime.now()
    for i in range(7):  # Next 7 days
        date = now.date() + timedelta(days=i)
        for hour in range(9, 18):  # 9 AM to 5 PM
            slot = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
            if slot not in booked_slots:
                available_slots.append(slot.strftime("%Y-%m-%d %H:00"))

    return available_slots[:5]  # Return the first 5 available slots

# Function to book an appointment
def book_appointment(user_id, appointment_time):
    cursor = db_connection.cursor()
    cursor.execute("INSERT INTO appointments (user_id, appointment_time) VALUES (%s, %s)", (user_id, appointment_time))
    db_connection.commit()
    cursor.close()
    
    
# Load the persisted vector store
embeddings = OpenAIEmbeddings()
docsearch = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)

# Trying different prompts:
prompt_1 = "You are June, the knowledgeable assistant of CloudJune. You are equipped with a rich repository of information about CloudJune, stored in a Chroma vector database. Your role is to assist users by answering questions strictly related to CloudJune. For any unrelated queries, unless for context-specific details provided by the user in previous chats, politely refuse to answer. You are capable of scheduling appointments with CloudJune and sending emails with the user's query and your response if they provide their email address. Ensure all responses are contextually aware and meaningful."
prompt_2 = "Welcome! You are the intelligent and insightful assistant of CloudJune, named June. You have access to extensive information about CloudJune via a Chroma vector database. Your task is to answer only those questions that pertain to CloudJune. For any off-topic questions, unless pertaining to user-specific details from earlier chats, you must decline to respond. Additionally, you can schedule appointments with CloudJune and send emails if the user provides their email address. Always provide contextually relevant and engaging responses."
prompt_3 = "Hello! You are June, the expert assistant for CloudJune. You have a deep understanding of all things CloudJune, with information sourced from a Chroma vector database. Your primary function is to respond to questions related to CloudJune. For queries that are unrelated, and not based on user-provided context from previous interactions, you should refuse to answer. You can also arrange appointments with CloudJune and send emails containing the conversation if the user supplies their email address. Ensure your responses are always contextually appropriate and valuable."
prompt_4 = "Hi there! You are June, CloudJune's highly knowledgeable assistant. You draw your information from a Chroma vector database, making you an expert on everything CloudJune. Your job is to answer questions related solely to CloudJune. If a question falls outside this scope and doesn't pertain to user-provided details from earlier chats, you should not answer. You can help schedule appointments with CloudJune and send emails if the user provides their email address. Your responses should always be contextually aware and meaningful."
prompt_5 = "Greetings! You are June, the well-informed assistant for CloudJune. Armed with a wealth of information from a Chroma vector database, you excel at answering questions about CloudJune. Your duty is to restrict your answers to CloudJune-related topics. Decline to respond to any unrelated inquiries, unless based on user-specific details from previous chats. Additionally, you have the capability to book appointments with CloudJune and email the user with their query and your response if they provide their email. Always ensure your answers are contextually accurate and helpful."
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    chat_history = []
    result = {}
    query = data['message']

    # Initialize chat history if not present in session
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'booking_stage' not in session:
        session['booking_stage'] = 'initial'

    # Prepare the conversation history
    chat_history = session['chat_history']
    
    
    # Add system message if it's the first message in the conversation
    if not chat_history:
        system_message = (prompt_2)
        chat_history.append(("system", system_message))

    # Generate response using the existing chain, including the conversation history
    result = chain({"question": query, "chat_history": chat_history})
    response = result['answer']

    # Check if the response includes booking appointment steps
    if "available slots" in response:
        session['booking_stage'] = 'started'
        available_slots = get_available_slots()
        response += "\n\nHere are the available slots:\n"
        for i, slot in enumerate(available_slots, 1):
            response += f"{i}. {slot}\n"
        response += "Please choose a slot by entering its number."

    elif session['booking_stage'] == 'started':
        try:
            slot_index = int(query) - 1
            available_slots = get_available_slots()
            chosen_slot = available_slots[slot_index]
            book_appointment(session.sid, chosen_slot)
            response = f"Great! Your appointment with CloudJune has been booked for {chosen_slot}. Is there anything else I can help you with?"
            session['booking_stage'] = 'initial'
        except (ValueError, IndexError):
            response = "I'm sorry, that's not a valid selection. Please choose a number from the list of available slots."

    # Update chat history
    chat_history.append((query, result['answer']))

    # Trim chat history if it gets too long (keep last 20 exchanges)
    if len(chat_history) > 40:
        chat_history = chat_history[-40:]

    # Save updated chat history to session
    session['chat_history'] = chat_history

    # Check if the user message contains an email address
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails_found = re.findall(email_pattern, query)

    # If an email address is found, send an email using SendGrid
    if emails_found:
        send_email(query, response)

    # Insert user query into conversations table
    cursor = db_connection.cursor()
    user_id = session.sid  # Using session ID as user identifier
    cursor.execute('INSERT INTO conversations (user_id, user_query, bot_response) VALUES (%s, %s, %s)', (user_id, query, response))
    db_connection.commit()
    cursor.close()

    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)

@atexit.register
def close_db_connection():
    if db_connection.is_connected():
        db_connection.close()
