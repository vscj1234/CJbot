import os
import logging
from mysql.connector import pooling
from flask import Flask, request, jsonify, render_template, send_file
from langchain.chains import LLMChain, RetrievalQA
from flask_session import Session
from langchain.chains.base import Chain
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import YamlOutputParser
from pydantic import BaseModel
from dotenv import load_dotenv
from flask import session 
from langchain.base_language import BaseLanguageModel
from langchain_core.callbacks import Callbacks
from typing import Any, Optional, Dict, List
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import openai
from gtts import gTTS
import tempfile
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage


###
# INTRODUCING AGENT SYSTEM

class Agent:
    """Base class for all agents."""
    def __init__(self, llm):
        self.llm = llm
    
    def handle(self, user_input):
        raise NotImplementedError("Agent needs to implement a handle method.")
    
class RAGAgent(Agent):
    """Agent for handling RAG-based queries"""
    def __init__(self, llm, rag_chain):
        super().__init__(llm)
        self.rag_chain = rag_chain

    def handle(self, user_input, chat_history):
        # Use the RAG chain to retrieve the answer based on the question
        formatted_history = self._format_chat_history(chat_history)
        context_prompt = f"""
        Recent conversation summary: {formatted_history}
        
        Current question: {user_input}
        """
        # Invoke the rag_chain to get the response
        response = self.rag_chain.invoke({"query": context_prompt})  # Correct method
        return response['result'] if isinstance(response, dict) else str(response)
    
    def _format_chat_history(self, chat_history):
        # Helper method to format the last few chat history entries for context
        formatted_history = []
        for message in chat_history:
            role = message.get('role', 'user')
            content = message.get('content', '')
            formatted_history.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted_history[-5:]) # Adjust as necessary
    
class AppointmentAgent(Agent):
    """Agent for handling appointment booking"""
    def __init__(self, llm):
        super().__init__(llm)
        
    def handle(self, user_input, chat_history):
        # Logic for booking appointments
        available_slots = ["9:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", "2:00 PM", "3:00 PM", "4:00 PM", "5:00 PM"]
        
        # First check if the user is selecting a slot or requesting availibility
        if "available slots" in user_input.lower():
            response = f"The available slots are: {', '.join(available_slots)}. Please choose one."
        elif any(slot in user_input for slot in available_slots):
            # If the user selects a slot, confirm the booking
            selected_slot = next((slot for slot in available_slots if slot in user_input), None)
            if selected_slot:
                response = f"Your appointment has been confirmed for {selected_slot}."
            else:
                response = "I didn't catch your preferred slot. Could you repeat it?"
        else:
            response = "Would you like to book an appointment? Available slots are from 9:00 AM to 6:00 PM."
            
        return response
###

# Helper function to detect user intent
def detect_intent(user_input):
    if "appointment" in user_input.lower() or "schedule" in user_input.lower():
        return "appointment"
    return "rag"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load environment variables
load_dotenv()

# Set your OpenAI API key
oai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the OpenAI embedding function
embeddings = OpenAIEmbeddings(api_key=oai_api_key)

# Load the vector store from disk with the embedding function
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Initialize the ChatOpenAI LLM (simulating GPT-4-mini)
llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=oai_api_key)

# Replace the MySQL connection setup with a connection pool
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "rishil10",
    "database": "cloudjunebot"
}

connection_pool = pooling.MySQLConnectionPool(
    pool_name="cloudjune_pool",
    pool_size=5,
    **db_config
)

@app.route('/')
def home():
    return render_template('base.html')

# Advanced RAG prompt template
rag_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are June, an AI assistant exclusively for CloudJune, a cloud service provider company. "
        "Your primary function is to assist with inquiries about CloudJune's products, services, and company information in multiple languages. "
        "Use the following context to answer the user's question:{context}"
        "IMPORTANT INSTRUCTIONS:"
        "1. Be conversational, polite, and adaptive. Respond appropriately to greetings, small talk, and CloudJune-related queries."
        "2. For greetings or small talk, engage briefly and naturally, then guide the conversation towards CloudJune topics."
        "3. Keep responses concise,professional and short, typically within 2-3 sentences unless more detail is necessary."
        "4. Use only the provided context for CloudJune-related information. Don't invent or assume details."
        "5. If a question isn't about CloudJune, politely redirect: 'I apologize, but I can only provide information about CloudJune, its products and services.Is there anything else you'd like to know?'"
        "6. For unclear questions, ask for clarification: 'To ensure I provide the most accurate information about CloudJune, could you please rephrase your question?'"
        "7. Adjust your language style to match the user's - formal or casual, but always maintain professionalism."
        "8. Always respond in the same language as the user's input."
        "9. If the context doesn't provide enough information for a comprehensive answer, be honest about the limitations and offer to assist with related topics you can confidently address."
        "10. Remember previous interactions within the conversation and maintain context continuity."
    ),
    HumanMessagePromptTemplate.from_template("{question}")
])

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": rag_prompt_template,
    }
)

REPHRASING_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "# Context #"
        "# Objective #"
        "Evaluate the given user question and determine if it requires reshaping according to chat history "
        "to provide necessary context and information for answering, or if it can be processed as it is."
        "#########"
        "# Style #"
        "The response should be clear, concise, and in the form of a straightforward decision - either 'Reshape required' or 'No reshaping required'."
        "#########"
        "# Tone #"
        "Professional and analytical."
        "#########"
        "# Audience #"
        "The audience is the internal system components that will act on the decision."
        "#########"
        "# Response #"
        "If the question should be rephrased return response in YAML file format:"
        "result: true"
        "Otherwise return in YAML file format:"
        "result: false"
    ),
    HumanMessagePromptTemplate.from_template(
        "##################"
        "# Chat History #"
        "{chat_history}"
        "#########"
        "# User question #"
        "{question}"
        "#########"
        "# Your Decision in YAML format: #"
    )
])

STANDALONE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "# Context #"
        "This is part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user questions."
        "#########"
        "# Objective #"
        "Take the original user question and chat history, and generate a new standalone question that can be understood and answered without relying on additional external information."
        "#########"
        "# Style #"
        "The reshaped standalone question should be clear, concise, and self-contained, while maintaining the intent and meaning of the original query."
        "#########"
        "# Tone #"
        "Neutral and focused on accurately capturing the essence of the original question."
        "#########"
        "# Audience #"
        "The audience is the internal system components that will act on the decision."
        "#########"
        "# Response #"
        "If the original question requires reshaping, provide a new reshaped standalone question that includes all necessary context and information to be self-contained."
        "If no reshaping is required, simply output the original question as is."
    ),
    HumanMessagePromptTemplate.from_template(
        "##################"
        "# Chat History #"
        "{chat_history}"
        "#########"
        "# User original question #"
        "{question}"
        "#########"
        "# The new Standalone question: #"
    )
])

ROUTER_DECISION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "# Context #"
        "This is part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user questions."
        "#########"
        "# Objective #"
        "Evaluate the given question and decide whether the RAG application is required to provide a comprehensive answer by retrieving relevant information from a knowledge base, or if the chat model's inherent knowledge is sufficient to generate an appropriate response."
        "#########"
        "# Style #"
        "The response should be a clear and direct decision, stated concisely."
        "#########"
        "# Tone #"
        "Analytical and objective."
        "#########"
        "# Audience #"
        "The audience is the internal system components that will act on the decision."
        "#########"
        "# Response #"
        "If the question should be rephrased return response in YAML file format:"
        "result: true"
        "Otherwise return in YAML file format:"
        "result: false"
    ),
    HumanMessagePromptTemplate.from_template(
        "##################"
        "# Chat History #"
        "{chat_history}"
        "#########"
        "# User question #"
        "{question}"
        "#########"
        "# Your Decision in YAML format: #"
    )
])

# Define the pydantic model for YAML output parsing
class ResultYAML(BaseModel):
    result: bool

class EnhancedConversationalRagChain(Chain):
    """Enhanced chain that encapsulates RAG application enabling natural conversations with improved context awareness."""
    rag_chain: Chain
    rephrasing_chain: LLMChain
    standalone_question_chain: LLMChain
    router_decision_chain: LLMChain
    yaml_output_parser: YamlOutputParser
    memory: ConversationBufferMemory
    llm: BaseLanguageModel
    
    input_key: str = "query"
    chat_history_key: str = "chat_history"
    output_key: str = "result"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key, self.chat_history_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        return "EnhancedConversationalRagChain"

    @classmethod
    def from_llm(
        cls,
        rag_chain: Chain,
        llm: BaseLanguageModel,
        callbacks: Optional[Callbacks] = None,
        **kwargs: Any,
    ) -> "EnhancedConversationalRagChain":
        """Initialize from LLM."""
        rephrasing_chain = LLMChain(llm=llm, prompt=REPHRASING_PROMPT, callbacks=callbacks)
        standalone_question_chain = LLMChain(llm=llm, prompt=STANDALONE_PROMPT, callbacks=callbacks)
        router_decision_chain = LLMChain(llm=llm, prompt=ROUTER_DECISION_PROMPT, callbacks=callbacks)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="query",
            output_key="result",
            return_messages=True
        )
        return cls(
            rag_chain=rag_chain,
            rephrasing_chain=rephrasing_chain,
            standalone_question_chain=standalone_question_chain,
            router_decision_chain=router_decision_chain,
            yaml_output_parser=YamlOutputParser(pydantic_object=ResultYAML),
            memory=memory,
            llm=llm,
            callbacks=callbacks,
            **kwargs,
        )

    def _format_chat_history(self, chat_history):
        formatted_history = []
        for message in chat_history:
            if isinstance(message, dict):
                role = message.get('role', '')
                content = message.get('content', '')
                formatted_history.append(f"{role.capitalize()}: {content}")
            elif isinstance(message, (HumanMessage, AIMessage)):
                formatted_history.append(f"{message.__class__.__name__}: {message.content}")
            else:
                formatted_history.append(str(message))
        return "\n".join(formatted_history[-5:])

    def _summarize_recent_context(self, formatted_history):
        if not formatted_history:
            return "No recent context available."
        
        summary_prompt = f"Summarize the following conversation history in a concise manner:\n{formatted_history}"
        summary_messages = [
            {"role": "system", "content": "Summarize the given conversation history concisely."},
            {"role": "user", "content": summary_prompt}
        ]
        summary = self.llm.invoke(summary_messages)
        return summary if isinstance(summary, str) else str(summary)

    def _extract_key_points(self, answer):
        extract_prompt = f"Extract 2-3 key points from the following answer:\n{answer}\nFormat the key points as a comma-separated string."
        extract_messages = [
            {"role": "system", "content": "Extract key points from the given answer."},
            {"role": "user", "content": extract_prompt}
        ]
        key_points = self.llm.invoke(extract_messages)
        return key_points if isinstance(key_points, str) else str(key_points)

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        """Call the chain."""
        chat_history = self.memory.chat_memory.messages
        question = inputs[self.input_key]

        try:
            formatted_history = self._format_chat_history(chat_history)
            recent_summary = self._summarize_recent_context(formatted_history)

            context_prompt = f"""
            Recent conversation summary: {recent_summary}

            Current question: {question}

            Please provide a response that takes into account the recent conversation context.
            """

            result = self.rag_chain.invoke({"query": context_prompt})
            answer = result['result'] if isinstance(result, dict) else str(result)

            key_points = self._extract_key_points(answer)

            self.memory.save_context(inputs, {"result": answer})

            return {self.output_key: answer, "key_points": key_points}
        except Exception as e:
            print(f"Error in _call: {str(e)}")  # Add this line for debugging
            answer = f"An error occurred while processing your request: {str(e)}"
            key_points = ""
            return {self.output_key: answer, "key_points": key_points}

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            audio_file.save(temp_audio.name)

        with open(temp_audio.name, "rb") as audio:
            transcript = openai.audio.transcriptions.create(
                model='whisper-1',
                file=audio,
                response_format='text',
                language='en'
            )

        return jsonify({"text": transcript})
    
    except Exception as e:
        print(f"Error in speech-to-text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        tts = gTTS(text=text, lang='en-uk')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
        
        return send_file(temp_audio.name, mimetype="audio/mpeg")
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    session_id = data.get('session_id', '')
    chat_history = data.get('chat_history', [])
    question = data.get('question', '')

    connection = None
    cursor = None
    try:
        connection = connection_pool.get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT id FROM users WHERE session_id = %s", (session_id,))
        user = cursor.fetchone()
        if not user:
            cursor.execute("INSERT INTO users (session_id) VALUES (%s)", (session_id,))
            connection.commit()
            user_id = cursor.lastrowid
        else:
            user_id = user[0]
            
        # Determine which agent to use based on user intent
        intent = detect_intent(question)
        if intent == "appointment":
            agent = AppointmentAgent(llm)
        else:
            agent = RAGAgent(llm, rag_chain)
            
        # Handle the user_query through the chosen agent
        response = agent.handle(question, chat_history)

        cursor.execute(
            "INSERT INTO conversations (user_id, user_query, bot_response) VALUES (%s, %s, %s)",
            (user_id, question, response)
        )
        connection.commit()

        return jsonify({"result": response})
    except Exception as e:
        print(f"Error in query route: {str(e)}")  # Add this line for debugging
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def update_chat_history(session_id, role, content):
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({"role": role, "content": content})
    session.modified = True
    
    # Limit chat history to last 10 messages (adjust as needed)
    session['chat_history'] = session['chat_history'][-10:]

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)