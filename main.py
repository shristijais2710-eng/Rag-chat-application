from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from knowledge import read_text_file, chunk_text, build_vector_store, retrieve_context, ask_groq

app = FastAPI()

# ------------------ CONFIG ------------------
FILE_PATH = "knowledge.txt"
CHUNK_SIZE = 500
OVERLAP = 50
TOP_K = 3

collection = None
embedder = None


# ------------------ STARTUP ------------------

@app.on_event("startup")
def startup_event():
    global collection, embedder
    text = read_text_file(FILE_PATH)
    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    collection, embedder = build_vector_store(chunks)
    print("✅ RAG system ready!")


# ------------------ API & FRONTEND ------------------

@app.get("/", response_class=HTMLResponse)
def home():
    """Serve ChatGPT-style chat UI."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>RAG Chat</title>
      <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
          background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%);
          color: #e8f1ff;
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 16px;
        }
        .container {
          max-width: 900px;
          width: 100%;
          background: #111827;
          border: 1px solid #1f2937;
          border-radius: 24px;
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
          display: flex;
          flex-direction: column;
          height: 90vh;
          max-height: 700px;
        }
        .header {
          padding: 24px;
          border-bottom: 1px solid #1f2937;
          text-align: center;
        }
        .header h1 {
          font-size: 1.8rem;
          font-weight: 700;
          background: linear-gradient(135deg, #3b82f6, #8b5cf6);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .header p {
          color: #9ca3af;
          font-size: 0.95rem;
          margin-top: 6px;
        }
        .chat-box {
          flex: 1;
          overflow-y: auto;
          padding: 24px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .chat-box::-webkit-scrollbar {
          width: 8px;
        }
        .chat-box::-webkit-scrollbar-track {
          background: #0f172a;
        }
        .chat-box::-webkit-scrollbar-thumb {
          background: #374151;
          border-radius: 4px;
        }
        .message {
          display: flex;
          gap: 12px;
          animation: slideIn 0.3s ease;
        }
        @keyframes slideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .message.user {
          justify-content: flex-end;
        }
        .bubble {
          padding: 12px 16px;
          border-radius: 16px;
          max-width: 70%;
          word-wrap: break-word;
          line-height: 1.5;
        }
        .message.user .bubble {
          background: #3b82f6;
          color: white;
          border-bottom-right-radius: 4px;
        }
        .message.bot .bubble {
          background: #1f2937;
          color: #d1d5db;
          border: 1px solid #374151;
          border-bottom-left-radius: 4px;
        }
        .input-section {
          padding: 16px 24px 24px;
          border-top: 1px solid #1f2937;
          background: #0f172a;
        }
        form {
          display: flex;
          gap: 12px;
        }
        input[type="text"] {
          flex: 1;
          padding: 12px 16px;
          border-radius: 12px;
          border: 1px solid #374151;
          background: #1f2937;
          color: #e5e7eb;
          font-size: 1rem;
          transition: all 0.2s;
        }
        input[type="text"]:focus {
          outline: none;
          border-color: #3b82f6;
          background: #111827;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        input[type="text"]::placeholder {
          color: #6b7280;
        }
        button {
          padding: 12px 24px;
          border-radius: 12px;
          border: none;
          background: linear-gradient(135deg, #3b82f6, #2563eb);
          color: white;
          cursor: pointer;
          font-weight: 600;
          font-size: 1rem;
          transition: all 0.2s;
        }
        button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        }
        button:active:not(:disabled) {
          transform: translateY(0);
        }
        button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .loading {
          display: inline-block;
          width: 8px;
          height: 8px;
          background: #9ca3af;
          border-radius: 50%;
          animation: pulse 1.4s infinite;
          margin-right: 4px;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>RAG Chat</h1>
          <p>Ask questions about your knowledge base</p>
        </div>
        <div class="chat-box" id="chatBox"></div>
        <div class="input-section">
          <form id="chatForm">
            <input type="text" id="question" placeholder="Type your question here..." autocomplete="off" />
            <button type="submit">Send</button>
          </form>
        </div>
      </div>

      <script>
        const chatBox = document.getElementById('chatBox');
        const form = document.getElementById('chatForm');
        const questionInput = document.getElementById('question');
        const button = form.querySelector('button');

        function addMessage(role, text) {
          const messageEl = document.createElement('div');
          messageEl.className = `message ${role}`;
          const bubble = document.createElement('div');
          bubble.className = 'bubble';
          bubble.textContent = text;
          messageEl.appendChild(bubble);
          chatBox.appendChild(messageEl);
          chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function askQuestion(question) {
          const response = await fetch(`/ask?question=${encodeURIComponent(question)}`);
          if (!response.ok) throw new Error('Request failed');
          return response.json();
        }

        form.addEventListener('submit', async (event) => {
          event.preventDefault();
          const question = questionInput.value.trim();
          if (!question) return;

          addMessage('user', question);
          questionInput.value = '';
          questionInput.disabled = true;
          button.disabled = true;

          try {
            const result = await askQuestion(question);
            addMessage('bot', result.answer || 'No answer returned.');
          } catch (err) {
            addMessage('bot', 'Error: ' + err.message);
          } finally {
            questionInput.disabled = false;
            button.disabled = false;
            questionInput.focus();
          }
        });

        questionInput.focus();
      </script>
    </body>
    </html>
    """


@app.get("/ask")
def ask(question: str):
    """API endpoint: retrieve context and answer question."""
    global collection, embedder
    
    context = retrieve_context(question, collection, embedder, TOP_K)
    answer = ask_groq(question, context)

    return {
        "question": question,
        "answer": answer
    }