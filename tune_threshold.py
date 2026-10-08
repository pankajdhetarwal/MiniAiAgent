import os
from tools import hybrid_search_tool, rerank_results
from dotenv import load_dotenv

load_dotenv()

# We need a tuning set separate from our final evaluation set.
# These are used strictly to find the real threshold, not for final reporting.

tuning_in_domain = [
    "What is MongoDB's total revenue for Q4?",
    "How many customers does MongoDB have?",
    "What is the non-GAAP gross profit margin for MongoDB?",
    "How much did Atlas revenue grow?",
    "What is the full year fiscal 2025 revenue for MongoDB?",
    "Who is the CEO of MongoDB?",
    "What was the operating loss in Q4?",
    "What are MongoDB's expectations for fiscal 2026 revenue?"
]

tuning_out_of_domain = [
    "How do I bake a chocolate cake?",
    "Who won the Super Bowl in 2021?",
    "What is the capital of Australia?",
    "Can you explain the plot of the Matrix?",
    "How many planets are in the solar system?",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
    "What is the formula for water?"
]

def analyze_scores():
    print("Gathering Cohere ReRank scores for IN-DOMAIN queries...")
    in_domain_scores = []
    for q in tuning_in_domain:
        results = hybrid_search_tool(q)
        if not results:
            print(f"  [!] No results for '{q}'")
            continue
        reranked = rerank_results(q, results, top_n=1)
        if reranked:
            score = reranked[0].get("rerank_score", 0)
            in_domain_scores.append((score, q))
            print(f"  {score:.4f} | {q}")
            
    print("\nGathering Cohere ReRank scores for OUT-OF-DOMAIN queries...")
    out_of_domain_scores = []
    for q in tuning_out_of_domain:
        results = hybrid_search_tool(q)
        if not results:
            print(f"  [!] No results for '{q}'")
            continue
        reranked = rerank_results(q, results, top_n=1)
        if reranked:
            score = reranked[0].get("rerank_score", 0)
            out_of_domain_scores.append((score, q))
            print(f"  {score:.4f} | {q}")

    # Analyze the gap
    if in_domain_scores and out_of_domain_scores:
        min_in_domain = min([s[0] for s in in_domain_scores])
        max_out_of_domain = max([s[0] for s in out_of_domain_scores])
        
        print("\n=== THRESHOLD ANALYSIS ===")
        print(f"Lowest In-Domain Score:      {min_in_domain:.4f}")
        print(f"Highest Out-of-Domain Score: {max_out_of_domain:.4f}")
        
        if min_in_domain > max_out_of_domain:
            recommended = (min_in_domain + max_out_of_domain) / 2
            print(f"\n=> CLEAR GAP DETECTED! Recommended threshold: {recommended:.4f}")
        else:
            print("\n=> OVERLAP DETECTED! A strict threshold might cause false positives/negatives.")
            print("You will need to balance recall vs precision.")

if __name__ == "__main__":
    analyze_scores()
