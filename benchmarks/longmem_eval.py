import os
import json
import time
import argparse
import tempfile
import shutil
from typing import Dict, Any, List

# LCEL and LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# SMRITI imports
from smriti_memcore.core import SMRITI, SmritiConfig
from smriti_memcore.integrations.langchain_memory import SmritiLangChainHistory

OLLAMA_BASE_URL = "http://localhost:11434/v1"

DEFAULT_MODELS = {
    "ollama": "mistral",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


def make_llm(provider: str, model: str):
    # max_retries=10 with exponential backoff so transient 429/529 (Anthropic
    # overloaded) doesn't kill a long benchmark run. Default in both ChatOpenAI
    # and ChatAnthropic is 2 retries — too few for a 50-case run during peak.
    if provider == "ollama":
        return ChatOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", model=model, temperature=0.0, max_retries=10)
    if provider == "anthropic":
        return ChatAnthropic(model=model, temperature=0.0, max_retries=10)
    return ChatOpenAI(model=model, temperature=0.0, max_retries=10)

# Simple exact/fuzzy match for evaluation
def compute_accuracy(prediction: str, ground_truth: str) -> float:
    # A robust LLM-as-a-judge is preferred, but for this script we do string inclusion.
    # LongMemEval answers are usually specific entities.
    pred_lower = prediction.lower()
    gt_lower = str(ground_truth).lower()
    
    # Simple check if the ground truth is somewhere in the prediction
    if gt_lower in pred_lower:
        return 1.0
        
    return 0.0

from datetime import datetime
from smriti_memcore.models import Episode, SalienceScore, MemorySource

def process_case_smriti(test_case: Dict[str, Any], temp_dir: str, hybrid: bool = True, llm=None, smriti_model: str = "mistral", rewrite_mode: str = "auto", snippet_mode: str = "auto") -> Dict[str, Any]:
    """Runs a single LongMemEval case through a SMRITI-augmented LangChain agent."""

    import os
    from datetime import timedelta

    # 1. Initialize SMRITI — use the same Anthropic model for consolidation LLM calls
    db_path = os.path.join(temp_dir, f"smriti_db_{test_case['question_id']}")
    config = SmritiConfig(
        storage_path=db_path,
        llm_model=smriti_model,
        rewrite_mode_default=rewrite_mode,
        snippet_mode_default=snippet_mode,
    )
    smriti_engine = SMRITI(config=config)
    if not hybrid:
        smriti_engine.retrieval_engine.fts_index = None
    
    # 2. Setup LangChain Environment
    smriti_history = SmritiLangChainHistory(smriti_client=smriti_engine, session_id="eval_session", top_k=5)
    
    # 3. Ingest the haystack sessions directly into LangChain history to emulate an integration
    # (Since we just want to load the history into SMRITI, we can push it onto the history object)
    sessions = test_case.get('haystack_sessions', [])
    
    print(f"[{test_case['question_id']}] Ingesting {len(sessions)} chat sessions into SMRITI...")
    base_time = datetime.now() - timedelta(days=len(sessions))
    
    for i, session in enumerate(sessions):
        # space sessions by roughly a day
        session_time = base_time + timedelta(days=i)
        
        for j, msg in enumerate(session):
            role = msg.get('role')
            content = msg.get('content', '')
            
            prefix = "Human: " if role == "user" else "AI: "
            
            # Inject directly as raw memory without spending LLM API credits to compute salience scores
            # Note: space individual messages by a few minutes to preserve strict temporal causality
            msg_time = session_time + timedelta(minutes=j*2)
            
            ep = Episode(
                content=f"{prefix} {content}",
                timestamp=msg_time,
                salience=SalienceScore(surprise=0.8, relevance=0.8, emotional=0.5, novelty=0.8, utility=0.8),
                source=MemorySource.DIRECT
            )
            smriti_engine.episode_buffer.add(ep)
            
        if (i + 1) % 10 == 0:
            print(f"  Ingested {i+1}/{len(sessions)} sessions into episodic buffer...")
                
    # Consolidate ONCE at the end of all history to build the graph
    print("  Triggering single batch consolidation...")
    smriti_engine.consolidate(depth="full")
        
    # 4. Build the QA chain using the injected LLM
    
    # 5. Ask the specific test question
    question = test_case['question']
    ground_truth = test_case['answer']
    
    start_time = time.time()
    
    # Dual-Process Fetch using the actual question string
    memories = smriti_engine.recall(question, top_k=5)
    episodes = smriti_engine.episode_buffer.search_semantic(question, top_k=5)
    
    context_blocks = []
    if memories:
        context_blocks.append("Abstract Knowledge:\n" + "\n".join(f"- {m.content}" for m in memories))
    if episodes:
        context_blocks.append("Specific Past Events:\n" + "\n".join(f"- {ep.content}" for ep in episodes))
        
    context_str = "Relevant Long-Term Memories:\n\n" + "\n\n".join(context_blocks) if context_blocks else "No relevant memories found."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with a perfect long-term memory. Answer the question using ONLY the provided memory context. If you don't know the answer based on the context, say 'I'm sorry, but I don't have that information.'\n\n{context}"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke(
        {"input": question, "context": context_str}
    )
    latency = time.time() - start_time
    
    prediction = response.content
    accuracy = compute_accuracy(prediction, ground_truth)
    
    print(f"  Q: {question}")
    print(f"  Expected: {ground_truth}")
    print(f"  SMRITI Answer: {prediction}")
    print(f"  Accuracy: {accuracy} (latency: {latency:.2f}s)\n")
    
    return {
        "question_id": test_case['question_id'],
        "question_type": test_case['question_type'],
        "accuracy": accuracy,
        "latency": latency,
        "prediction": prediction,
        "ground_truth": ground_truth
    }
    

from langchain_community.chat_message_histories import ChatMessageHistory

def process_case_baseline(test_case: Dict[str, Any], llm=None) -> Dict[str, Any]:
    """Runs a single LongMemEval case through a standard LangChain agent (Full Context)."""

    # 1. Setup Standard LangChain Memory (Infinite Buffer)
    history = ChatMessageHistory()

    # 2. Ingest histories directly
    sessions = test_case.get('haystack_sessions', [])
    print(f"[{test_case['question_id']}] Ingesting {len(sessions)} chat sessions into standard LLM context...")

    for session in sessions:
        for msg in session:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'user':
                history.add_user_message(content)
            elif role == 'assistant':
                history.add_ai_message(content)

    # 3. Create Chain
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use your recalled context to answer the user's question directly and concisely."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    # 4. Ask the test question
    question = test_case['question']
    ground_truth = test_case['answer']
    
    start_time = time.time()
    response = chain_with_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": "eval_session_baseline"}}
    )
    latency = time.time() - start_time
    
    prediction = response.content
    accuracy = compute_accuracy(prediction, ground_truth)
    
    print(f"  Q: {question}")
    print(f"  Expected: {ground_truth}")
    print(f"  BASELINE Answer: {prediction}")
    print(f"  Accuracy: {accuracy} (latency: {latency:.2f}s)\n")
    
    return {
        "question_id": test_case['question_id'],
        "question_type": test_case['question_type'],
        "accuracy": accuracy,
        "latency": latency,
        "prediction": prediction,
        "ground_truth": ground_truth
    }
    

def main():
    parser = argparse.ArgumentParser(description="Run LongMemEval Benchmark")
    parser.add_argument("--dataset", type=str, default="data/longmemeval/longmemeval_s_cleaned.json", help="Path to json dataset")
    parser.add_argument("--limit", type=int, default=5, help="Number of cases to evaluate (full set is 500)")
    parser.add_argument("--baseline", action="store_true", help="Run with standard ConversationBufferMemory instead of SMRITI")
    parser.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid",
                        help="Retrieval mode for SMRITI: hybrid (FTS+RRF, default) or vector-only")
    parser.add_argument("--llm", choices=["ollama", "openai", "anthropic"], default="ollama",
                        help="LLM provider for QA chain (default: ollama)")
    parser.add_argument("--llm-model", dest="llm_model", default=None,
                        help="Model name for QA chain (defaults: mistral / gpt-4o-mini / claude-haiku-4-5-20251001)")
    parser.add_argument("--smriti-model", dest="smriti_model", default="mistral",
                        help="Ollama/LLM model for SMRITI consolidation calls (default: mistral)")
    parser.add_argument("--output", type=str, default="results/longmemeval_results.json",
                        help="Path to write JSON results (default: results/longmemeval_results.json)")
    parser.add_argument("--rewrite-mode", dest="rewrite_mode", default="auto",
                        choices=["auto", "llm", "none"],
                        help="Smarter-recall query rewriting mode (default: auto). 'none' disables.")
    parser.add_argument("--snippet-mode", dest="snippet_mode", default="auto",
                        choices=["auto", "llm", "none"],
                        help="Smarter-recall snippet extraction mode (default: auto). 'none' disables.")
    args = parser.parse_args()

    llm_model = args.llm_model or DEFAULT_MODELS[args.llm]
    llm = make_llm(args.llm, llm_model)
    print(f"LLM: {args.llm}/{llm_model}")

    print(f"Loading dataset from {args.dataset}...")
    try:
        with open(args.dataset, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"Error loading {args.dataset}: {e}")
        return

    cases = dataset[:args.limit]
    print(f"Loaded {len(cases)} cases. Starting evaluation...")

    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, case in enumerate(cases):
            print(f"=== Evaluating Case {idx+1}/{len(cases)} ===")
            if args.baseline:
                res = process_case_baseline(case, llm=llm)
            else:
                res = process_case_smriti(
                    case, temp_dir,
                    hybrid=(args.mode == "hybrid"),
                    llm=llm,
                    smriti_model=args.smriti_model,
                    rewrite_mode=args.rewrite_mode,
                    snippet_mode=args.snippet_mode,
                )
            results.append(res)

    # Aggregate and print results
    total_acc = sum(r['accuracy'] for r in results) / len(results)
    avg_latency = sum(r['latency'] for r in results) / len(results)

    llm_label = f"{args.llm}/{llm_model}"
    smart_tag = f"rewrite={args.rewrite_mode},snippet={args.snippet_mode}"
    if args.baseline:
        method_label = f"Baseline (full context) [{llm_label}]"
    elif args.mode == "hybrid":
        method_label = f"SMRITI hybrid (FTS+RRF) [{smart_tag}] [{llm_label}]"
    else:
        method_label = f"SMRITI vector-only [{smart_tag}] [{llm_label}]"

    print("=" * 40)
    print(f"LONGMEMEVAL EVALUATION COMPLETE")
    print(f"Method: {method_label}")
    print(f"Cases Evaluated: {len(results)}")
    print(f"Overall Accuracy: {total_acc * 100:.1f}%")
    print(f"Average Inquiry Latency: {avg_latency:.2f}s")
    print("=" * 40)

    output_file = args.output
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "method": method_label,
                "total_cases": len(results),
                "accuracy": total_acc,
                "latency": avg_latency
            },
            "cases": results
        }, f, indent=2)
        
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
