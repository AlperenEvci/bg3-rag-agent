"""
Database module for handling PostgreSQL operations
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import time
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database credentials from environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Function to wait for database to be ready
def wait_for_database(retries=5, delay=2):
    """
    Wait for the PostgreSQL database to be available.
    Retry mechanism for Docker startup sequence.
    """
    print(f"Attempting to connect to PostgreSQL at {DB_HOST}:{DB_PORT}...")
    current_try = 0

    while current_try < retries:
        try:
            conn = psycopg2.connect(
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
                dbname="postgres"  # Connect to default postgres database first
            )
            conn.close()
            print("Successfully connected to PostgreSQL!")
            return True
        except Exception as e:
            current_try += 1
            print(f"Connection attempt {current_try}/{retries} failed: {e}")
            if current_try < retries:
                print(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
    
    print("Failed to connect to the database after multiple attempts")
    return False

# Try to connect to the database first
wait_for_database()

# Construct database URL from environment variables
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine with extended timeout and retries
engine = create_engine(
    DATABASE_URL,
    connect_args={"connect_timeout": 10},
    pool_pre_ping=True,  # Verify connections before usage
)

# Create a base class for our ORM models
Base = declarative_base()

class Conversation(Base):
    """Model representing a conversation with the LLM"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=True)  # Can be used for user identification if needed
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    session_id = Column(String(100), nullable=True)  # To group conversations by session
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "query": self.query,
            "response": self.response,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id
        }

# Create all tables in the database
def init_db():
    """Initialize database by creating all tables"""
    Base.metadata.create_all(engine)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to get a database session
def get_db_session():
    """Get a new database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

# Add a new conversation to the database
def add_conversation(query, response, user_id=None, session_id=None):
    """
    Add a new conversation to the database
    
    Args:
        query (str): User's question
        response (str): LLM's response
        user_id (str, optional): User identifier
        session_id (str, optional): Session identifier
    
    Returns:
        Conversation: The created conversation record
    """
    db = SessionLocal()
    try:
        conversation = Conversation(
            query=query,
            response=response,
            user_id=user_id,
            session_id=session_id
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation
    finally:
        db.close()

# Get conversation history
def get_conversation_history(limit=10, offset=0, user_id=None, session_id=None):
    """
    Get conversation history from the database
    
    Args:
        limit (int): Maximum number of conversations to return
        offset (int): Number of records to skip (for pagination)
        user_id (str, optional): Filter by user_id
        session_id (str, optional): Filter by session_id
    
    Returns:
        list: List of conversation records
    """
    db = SessionLocal()
    try:
        query = db.query(Conversation).order_by(Conversation.timestamp.desc())
        
        if user_id:
            query = query.filter(Conversation.user_id == user_id)
        
        if session_id:
            query = query.filter(Conversation.session_id == session_id)
        
        conversations = query.offset(offset).limit(limit).all()
        return [conv.to_dict() for conv in conversations]
    finally:
        db.close()
