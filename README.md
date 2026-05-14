# RAG-Bloosm

# 🌸 BLOOSM RAG Chatbot

A simple and powerful **Retrieval-Augmented Generation (RAG)** chatbot built using:

- LangChain
- ChromaDB
- OpenAI Embeddings
- Mistral AI
- Python

This project retrieves relevant information from documents stored in a vector database and generates accurate answers using an LLM.

---

# 🚀 Features

✅ Document Retrieval using ChromaDB  
✅ Semantic Search with OpenAI Embeddings  
✅ MMR Search (Diverse Retrieval)  
✅ Mistral AI Integration  
✅ Clean Prompt Engineering  
✅ Interactive CLI Chatbot  
✅ Persistent Vector Database  

---

# 🛠️ Tech Stack

- Python
- LangChain
- ChromaDB
- OpenAI Embeddings
- Mistral AI
- dotenv

---
⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/BLOOSM-RAG.git

cd BLOOSM-RAG
2️⃣ Create Virtual Environment
Windows
python -m venv venv

venv\Scripts\activate
Linux / Mac
python3 -m venv venv

source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file in the root directory.

OPENAI_API_KEY=your_openai_api_key

MISTRAL_API_KEY=your_mistral_api_key
▶️ Run the Project
python main.py
💬 Example
You : What is machine learning?

AI : Machine learning is a field of AI that enables systems to learn from data.
🧠 How It Works
User asks a question.
ChromaDB retrieves relevant document chunks.
Context is passed to the LLM.
Mistral AI generates the final answer.
Response is displayed in the terminal.
🔍 Retrieval Settings
search_type = "mmr"

k = 4
fetch_k = 10
lambda_mult = 0.5
Meaning
MMR Search → Improves diversity in retrieved documents
k → Final documents returned
fetch_k → Initial documents fetched
lambda_mult → Balance between relevance and diversity
📦 Requirements

Example requirements.txt

langchain
langchain-openai
langchain-community
langchain-mistralai
chromadb
python-dotenv
tiktoken
openai

📌 Future Improvements
Enterprise RAG Chatbot with YouTube, PDF and Website Integration”

Pull requests are welcome.

For major changes, please open an issue first to discuss what you would like to change.

⭐ Support

If you like this project, give it a ⭐ on GitHub.

👨‍💻 Author

Ashu


# 📂 Project Structure

```bash
BLOOSM-RAG/
│
├── chroma_db/              # Vector database
├── main.py                 # Main chatbot code
├── requirements.txt
├── .env
└── README.md

That combination already looks good for internships and service-based companies like :contentReference[oaicite:0]{index=0}, :contentReference[oaicite:1]{index=1}, and :contentReference[oaicite:2]{index=2}.
