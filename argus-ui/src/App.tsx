import { useState, useRef, useEffect } from 'react';
import './App.css';

interface Source {
  text: string;
  rerank_score: number;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  gate_passed?: boolean;
  tool_used?: string;
  isStreaming?: boolean;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(`session-${Math.random().toString(36).substring(7)}`);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: userMsg }),
      });
      
      const data = await response.json();
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        gate_passed: data.gate_passed,
        tool_used: data.tool_used
      }]);
    } catch (error) {
      console.error('Error fetching chat:', error);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Connection error. Please ensure the backend is running.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="logo-container">
          <div className="logo-orb"></div>
          <h1>Argus Agent</h1>
        </div>
        <div className="header-badges">
          <span className="badge">RAG</span>
          <span className="badge">Grounded</span>
        </div>
      </header>

      <main className="chat-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h2>Welcome to Argus</h2>
            <p>I am an enterprise-grade AI agent powered by hybrid search, semantic reranking, and groundedness gates.</p>
            <div className="suggested-prompts">
              <button onClick={() => setInput("What was MongoDB's total revenue for the fourth quarter?")}>What was MongoDB's total revenue for Q4?</button>
              <button onClick={() => setInput("What is 1542 * 8?")}>What is 1542 * 8?</button>
              <button onClick={() => setInput("What is the recipe for chocolate chip cookies?")}>What is the recipe for cookies?</button>
            </div>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message-wrapper ${msg.role}`}>
                <div className="message">
                  <div className="message-header">
                    {msg.role === 'user' ? 'You' : 'Argus'}
                  </div>
                  <div className="message-content">
                    {msg.content}
                  </div>
                  
                  {msg.role === 'assistant' && msg.tool_used && msg.tool_used !== 'none' && (
                    <div className="agent-metadata">
                      <div className="metadata-row">
                        <span className="metadata-label">Tool Routed:</span>
                        <span className="metadata-value tag">{msg.tool_used}</span>
                      </div>
                      
                      {msg.gate_passed !== undefined && (
                        <div className="metadata-row">
                          <span className="metadata-label">Groundedness Gate:</span>
                          <span className={`metadata-value tag ${msg.gate_passed ? 'gate-pass' : 'gate-fail'}`}>
                            {msg.gate_passed ? 'PASSED' : 'REJECTED (Insufficient Context)'}
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-container">
                      <h4>Retrieved Context (Reranked Top {msg.sources.length})</h4>
                      <div className="sources-list">
                        {msg.sources.map((source, sIdx) => (
                          <div key={sIdx} className="source-item">
                            <div className="source-score" title="Cohere Rerank Score">
                              {(source.rerank_score * 100).toFixed(1)}%
                            </div>
                            <div className="source-text">{source.text}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message-wrapper assistant">
                <div className="message loading">
                  <div className="dot-typing"></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      <footer className="input-area">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the MongoDB earnings report..."
            disabled={isLoading}
            className="text-input"
          />
          <button type="submit" disabled={!input.trim() || isLoading} className="submit-button">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </form>
      </footer>
    </div>
  );
}

export default App;
