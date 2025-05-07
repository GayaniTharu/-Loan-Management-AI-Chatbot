# -Loan-Management-AI-Chatbot
This AI-powered chatbot interfaces with MongoDB to provide detailed loan management assistance, including customer information, loan details, payment schedules, and more. Built with modern AI/ML technologies and a focus on user experience.

📋 Overview
This AI-powered chatbot interfaces with MongoDB to provide detailed loan management assistance, offering real-time access to customer information, loan details, and payment schedules in a structured, easy-to-read format.

✨ Features

Real-time loan information retrieval
Structured responses with emoji-enhanced formatting
Persistent conversation memory
MongoDB integration for loan data management
Vector-based similarity search using FAISS
Responsive web interface
Session-based chat history
🛠️ Tech Stack

Backend: Flask
AI/ML:
LangChain Framework
Google Gemini 1.5 Pro LLM
FAISS Vector Store
Database: MongoDB Atlas
Frontend: HTML/CSS/JavaScript
Additional:
CORS support
Environment variable management
Session management

🚀 Getting Started

Prerequisites
Python 3.8+
MongoDB Atlas account
Google AI Platform API key

Installation

1.Clone the repository
git clone https://github.com/yourusername/loan-management-chatbot.git
cd loan-management-chatbot

2.Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

3.Install dependencies
pip install -r requirements.txt

4.Set up environment variables
# Create .env file
GOOGLE_API_KEY=your_google_api_key
MONGODB_URI=your_mongodb_connection_string

5.Run the application
python app.py

💻 Usage
The chatbot can handle various loan-related queries including:

Loan status inquiries
Payment schedule information
Customer detail retrieval
Loan application status
Payment history tracking

Example query:"What is the loan status for customer ID 12345?"

🙏 Acknowledgments
LangChain for the excellent LLM framework
Google for Gemini Pro API
MongoDB Atlas for database services


Made with ❤️ by GayaniTharu










