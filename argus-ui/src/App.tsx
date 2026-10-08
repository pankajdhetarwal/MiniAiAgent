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
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isEvalOpen, setIsEvalOpen] = useState(false);
  const [isImpactOpen, setIsImpactOpen] = useState(false);
  
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalMetrics, setEvalMetrics] = useState({
    faithfulness: 42,
    answer_relevancy: 82,
    context_precision_after: 35,
    context_precision_before: 27,
    hallucination_block_rate: 100
  });

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleLiveEvaluation = async () => {
    setIsEvaluating(true);
    try {
      const response = await fetch('http://localhost:8000/api/evaluate', {
        method: 'POST',
      });
      const data = await response.json();
      if (data.status === 'success') {
        setEvalMetrics(data.metrics);
      }
    } catch (error) {
      console.error("Evaluation failed", error);
    } finally {
      setIsEvaluating(false);
    }
  };

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
          <button className="architecture-btn impact-btn" onClick={() => setIsImpactOpen(true)}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M13 10V3L4 14H11V21L20 10H13Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Reranker Impact
          </button>
          <button className="architecture-btn eval-btn" onClick={() => setIsEvalOpen(true)}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M18 20V10M12 20V4M6 20V14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Live Evaluation
          </button>
          <button className="architecture-btn" onClick={() => setIsModalOpen(true)}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" strokeWidth="2"/>
              <path d="M12 16V12M12 8H12.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            How it Works
          </button>
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

      {isModalOpen && (
        <div className="modal-overlay" onClick={() => setIsModalOpen(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <button className="close-modal" onClick={() => setIsModalOpen(false)}>×</button>
            <h2>Argus Architecture</h2>
            <p className="modal-subtitle">Enterprise-Grade Retrieval-Augmented Generation</p>
            
            <div className="architecture-steps">
              <div className="arch-step">
                <div className="step-number">1</div>
                <div className="step-details">
                  <h3>LLM Router</h3>
                  <p>Llama 3.2 intercepts the query and semantically routes it to the correct tool (Vector Search, Calculator, or standard conversation).</p>
                </div>
              </div>
              
              <div className="arch-step">
                <div className="step-number">2</div>
                <div className="step-details">
                  <h3>Hybrid Search & RRF</h3>
                  <p>Executes both keyword search (BM25) and semantic search (Cosine Similarity) on MongoDB Atlas. Normalizes and merges results using Reciprocal Rank Fusion.</p>
                </div>
              </div>
              
              <div className="arch-step">
                <div className="step-number">3</div>
                <div className="step-details">
                  <h3>Cross-Encoder Reranking</h3>
                  <p>A Cohere cross-encoder semantically scores the top chunks against the exact user query to float the most accurate context to the absolute top.</p>
                </div>
              </div>
              
              <div className="arch-step">
                <div className="step-number">4</div>
                <div className="step-details">
                  <h3>Groundedness Gate</h3>
                  <p>Empirically tuned firewall (Threshold: 0.50). If the top retrieved chunk scores below this, the gate short-circuits the agent to guarantee zero hallucination.</p>
                </div>
              </div>

              <div className="arch-step">
                <div className="step-number">5</div>
                <div className="step-details">
                  <h3>Local Generation</h3>
                  <p>Llama 3.2 generates the final answer strictly bounded by the gated context.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {isEvalOpen && (
        <div className="modal-overlay" onClick={() => setIsEvalOpen(false)}>
          <div className="modal-content eval-modal" onClick={e => e.stopPropagation()}>
            <button className="close-modal" onClick={() => setIsEvalOpen(false)}>×</button>
            <h2>System Evaluation</h2>
            <p className="modal-subtitle">RAGAS Framework Metrics</p>
            
            <div className="eval-actions" style={{ textAlign: 'center', marginBottom: '2rem' }}>
              <button 
                className="architecture-btn" 
                onClick={handleLiveEvaluation}
                disabled={isEvaluating}
                style={{ 
                  background: isEvaluating ? 'rgba(255,255,255,0.1)' : 'var(--accent-primary)',
                  color: isEvaluating ? '#888' : '#fff',
                  border: 'none',
                  padding: '0.75rem 2rem'
                }}
              >
                {isEvaluating ? 'Running Live Evaluation (~30s)...' : 'Evaluate Argus (Live)'}
              </button>
            </div>
            
            <div className="eval-grid">
              <div className="eval-card">
                <div className="eval-score">{evalMetrics.answer_relevancy}%</div>
                <div className="eval-name">Answer Relevancy</div>
                <div className="eval-desc">How directly the response answers the user's question.</div>
              </div>
              <div className="eval-card">
                <div className="eval-score">{evalMetrics.faithfulness}%</div>
                <div className="eval-name">Faithfulness</div>
                <div className="eval-desc">Measures how strictly grounded the answer is in the context.</div>
              </div>
              <div className="eval-card">
                <div className="eval-score">{evalMetrics.context_precision_after}%</div>
                <div className="eval-name">Context Precision</div>
                <div className="eval-desc">Signal-to-noise ratio of chunks retrieved by Hybrid Search.</div>
              </div>
              <div className="eval-card highlight-card">
                <div className="eval-score">{evalMetrics.hallucination_block_rate}%</div>
                <div className="eval-name">Hallucination Block Rate</div>
                <div className="eval-desc">Out-of-domain queries caught by the Groundedness Gate (Threshold 0.50).</div>
              </div>
            </div>

            <div className="eval-metadata">
              <p><strong>Judge Model:</strong> Groq Llama-3.1-120B</p>
              <p><strong>Embeddings:</strong> BAAI/bge-base-en-v1.5</p>
            </div>
          </div>
        </div>
      )}

      {isImpactOpen && (
        <div className="modal-overlay" onClick={() => setIsImpactOpen(false)}>
          <div className="modal-content impact-modal" onClick={e => e.stopPropagation()}>
            <button className="close-modal" onClick={() => setIsImpactOpen(false)}>×</button>
            <h2>Reranker Impact</h2>
            <p className="modal-subtitle">Cohere Cross-Encoder Performance Boost</p>
            
            <div className="impact-comparison">
              <div className="impact-side before">
                <h3>Before Reranking</h3>
                <div className="impact-score">{evalMetrics.context_precision_before}%</div>
                <p>Context Precision</p>
                <div className="impact-details">
                  Raw Hybrid Search (Top 20)<br/>
                  <small>Basic Vector + BM25 Fusion</small>
                </div>
              </div>
              
              <div className="impact-arrow">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              
              <div className="impact-side after">
                <h3>After Reranking</h3>
                <div className="impact-score">{evalMetrics.context_precision_after}%</div>
                <p>Context Precision</p>
                <div className="impact-details">
                  Cohere Rerank-v3.5 (Top 5)<br/>
                  <small>Semantic Cross-Encoder</small>
                </div>
              </div>
            </div>

            <div className="impact-explanation">
              <p><strong>Why this matters:</strong> The Reranker analyzes the semantic relationship between the user's exact query and the retrieved chunks, floating the most contextually relevant documents to the top. This <strong>+{evalMetrics.context_precision_after - evalMetrics.context_precision_before}% absolute boost</strong> in Context Precision ensures the LLM generates answers from the highest-quality signal, significantly reducing hallucinations.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
