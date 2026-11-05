"""
BEAM (Beyond a Million Tokens) Benchmark Evaluation Harness for SMRITI
-----------------------------------------------------------------------
Evaluates SMRITI's long-term memory retrieval and reasoning against the
BEAM benchmark (Tavakoli et al., ICLR 2026).
Dataset: https://huggingface.co/datasets/Mohammadta/BEAM

Evaluates across 10 distinct memory abilities:
  1. Abstention
  2. Contradiction Resolution
  3. Event Ordering
  4. Information Extraction
  5. Instruction Following
  6. Knowledge Update
  7. Multi-Session Reasoning
  8. Preference Following
  9. Summarization
  10. Temporal Reasoning

Usage:
  # Run on all 20 conversations of 100K split:
  python3 benchmarks/beam_eval.py --all --split 100K --top-k 10 --output results/beam_100k_full_results.json

  # Run on a single conversation:
  python3 benchmarks/beam_eval.py --conv-index 0 --split 100K --top-k 10 --output results/beam_eval_conv0.json
"""

import os
import sys

# Ensure repository root is at the head of sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import re
import time
import argparse
import tempfile
import urllib.request
from collections import defaultdict
from typing import Dict, Any, List, Set, Optional

from dotenv import load_dotenv
load_dotenv()

from smriti_memcore.core import SMRITI, SmritiConfig
from smriti_memcore.models import MemorySource

BEAM_HF_API = "https://datasets-server.huggingface.co/rows?dataset=Mohammadta/BEAM&config=default&split={split}&offset={offset}&limit={limit}"

CATEGORIES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]


def load_all_beam_conversations(split: str = "100K", cache_dir: str = "data/beam_cache") -> List[Dict[str, Any]]:
    """Load or fetch all conversations for a split."""
    os.makedirs(cache_dir, exist_ok=True)
    all_cache_file = os.path.join(cache_dir, f"beam_{split}_all.json")
    
    if os.path.exists(all_cache_file):
        with open(all_cache_file, "r") as f:
            return json.load(f)
            
    print(f"Downloading full {split} split from Hugging Face...")
    limit = 20 if split == "100K" else 35
    url = BEAM_HF_API.format(split=split, offset=0, limit=limit)
    req = urllib.request.Request(url, headers={"User-Agent": "smriti-eval/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    rows = [r["row"] for r in data["rows"]]
    with open(all_cache_file, "w") as f:
        json.dump(rows, f)
        
    return rows


def fetch_beam_conversation(split: str = "100K", conv_index: int = 0, cache_dir: str = "data/beam_cache") -> Dict[str, Any]:
    """Fetch a single conversation from local cache or Hugging Face dataset server."""
    os.makedirs(cache_dir, exist_ok=True)
    all_cache_file = os.path.join(cache_dir, f"beam_{split}_all.json")
    if os.path.exists(all_cache_file):
        with open(all_cache_file, "r") as f:
            all_rows = json.load(f)
            if 0 <= conv_index < len(all_rows):
                return all_rows[conv_index]
                
    cache_file = os.path.join(cache_dir, f"beam_{split}_conv_{conv_index}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    url = BEAM_HF_API.format(split=split, offset=conv_index, limit=1)
    req = urllib.request.Request(url, headers={"User-Agent": "smriti-eval/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    row = data["rows"][0]["row"]
    with open(cache_file, "w") as f:
        json.dump(row, f)
        
    return row


def extract_turn_ids(memories) -> List[int]:
    """Extract turn IDs from SMRITI memory content headers [Turn X | ...]."""
    ids = []
    for m in memories:
        match = re.search(r"\[Turn\s+(\d+)", m.content)
        if match:
            ids.append(int(match.group(1)))
    return ids


def parse_ground_truth_ids(src_raw: Any) -> Set[int]:
    """Normalize source_chat_ids from BEAM question metadata, recursively flattening nested structures."""
    if src_raw is None:
        return set()
    res = set()
    def _flatten(x):
        if isinstance(x, (list, tuple, set)):
            for item in x:
                _flatten(item)
        elif isinstance(x, dict):
            for v in x.values():
                _flatten(v)
        elif isinstance(x, int):
            res.add(x)
        elif isinstance(x, str) and x.isdigit():
            res.add(int(x))
    _flatten(src_raw)
    return res


def evaluate_single_conversation(
    conv_data: Dict[str, Any],
    conv_index: int,
    top_k: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    conv_id = conv_data["conversation_id"]
    seed = conv_data["conversation_seed"]
    chat = conv_data["chat"]
    pq = conv_data["probing_questions"]
    if isinstance(pq, str):
        pq = eval(pq)
        
    title = seed.get("title", f"Conversation {conv_index}")
    category_name = seed.get("category", "General")
    
    # 1. Ingestion Phase
    temp_dir = tempfile.mkdtemp(prefix=f"smriti_beam_{conv_index}_")
    config = SmritiConfig(
        storage_path=temp_dir,
        episode_buffer_trigger=100000,
        retrieval_top_k=top_k,
    )
    smriti = SMRITI(config)
    
    t_start_ingest = time.perf_counter()
    turn_count = 0
    for batch in chat:
        for turn in batch:
            tid = turn["id"]
            time_anchor = turn.get("time_anchor", "N/A")
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")
            formatted_text = f"[Turn {tid} | Time: {time_anchor} | {role}]: {content}"
            source = MemorySource.USER_STATED if role == "USER" else MemorySource.DIRECT
            smriti.encode(formatted_text, source=source, use_llm=False, force=True)
            turn_count += 1
            
    t_ingest = time.perf_counter() - t_start_ingest
    
    # 2. Probing Phase
    evaluated_cases = []
    category_hit1 = defaultdict(list)
    category_hit5 = defaultdict(list)
    category_hit10 = defaultdict(list)
    category_recall10 = defaultdict(list)
    category_mrr = defaultdict(list)
    
    for category in CATEGORIES:
        questions = pq.get(category, [])
        for idx, q in enumerate(questions):
            question_text = q["question"]
            gt_ids = parse_ground_truth_ids(q.get("source_chat_ids"))
            rubric = q.get("rubric", [])
            
            t0 = time.perf_counter()
            recalled_memories = smriti.recall(question_text, top_k=top_k)
            t_recall = time.perf_counter() - t0
            
            retrieved_turn_ids = extract_turn_ids(recalled_memories)
            
            # Abstention questions intentionally have no source turns
            is_abstention = (len(gt_ids) == 0)
            if is_abstention:
                hit1, hit5, hit10, recall10, mrr = 1.0, 1.0, 1.0, 1.0, 1.0
            else:
                hit1 = 1.0 if (retrieved_turn_ids and retrieved_turn_ids[0] in gt_ids) else 0.0
                hit5 = 1.0 if any(tid in gt_ids for tid in retrieved_turn_ids[:5]) else 0.0
                hit10 = 1.0 if any(tid in gt_ids for tid in retrieved_turn_ids[:top_k]) else 0.0
                overlap = gt_ids.intersection(set(retrieved_turn_ids[:top_k]))
                recall10 = len(overlap) / len(gt_ids) if gt_ids else 1.0
                
                mrr = 0.0
                for rank, tid in enumerate(retrieved_turn_ids, start=1):
                    if tid in gt_ids:
                        mrr = 1.0 / rank
                        break
                        
            category_hit1[category].append(hit1)
            category_hit5[category].append(hit5)
            category_hit10[category].append(hit10)
            category_recall10[category].append(recall10)
            category_mrr[category].append(mrr)
            
            case_record = {
                "category": category,
                "question": question_text,
                "ground_truth_turn_ids": list(gt_ids),
                "retrieved_turn_ids": retrieved_turn_ids,
                "hit@1": hit1,
                "hit@5": hit5,
                "hit@10": hit10,
                "recall@10": recall10,
                "mrr": mrr,
                "latency_sec": t_recall,
                "rubric": rubric,
            }
            evaluated_cases.append(case_record)
            
    all_hit1 = [h for hs in category_hit1.values() for h in hs]
    all_hit5 = [h for hs in category_hit5.values() for h in hs]
    all_hit10 = [h for hs in category_hit10.values() for h in hs]
    all_recall10 = [r for rs in category_recall10.values() for r in rs]
    all_mrr = [m for ms in category_mrr.values() for m in ms]
    
    conv_summary = {
        "conv_index": conv_index,
        "conv_id": conv_id,
        "domain": category_name,
        "title": title,
        "turns": turn_count,
        "questions": len(evaluated_cases),
        "ingest_time": t_ingest,
        "hit@1": sum(all_hit1) / len(all_hit1) if all_hit1 else 0,
        "hit@5": sum(all_hit5) / len(all_hit5) if all_hit5 else 0,
        "hit@10": sum(all_hit10) / len(all_hit10) if all_hit10 else 0,
        "recall@10": sum(all_recall10) / len(all_recall10) if all_recall10 else 0,
        "mrr": sum(all_mrr) / len(all_mrr) if all_mrr else 0,
        "category_hit10": {cat: sum(category_hit10[cat])/len(category_hit10[cat]) for cat in CATEGORIES if cat in category_hit10},
        "category_recall10": {cat: sum(category_recall10[cat])/len(category_recall10[cat]) for cat in CATEGORIES if cat in category_recall10},
        "cases": evaluated_cases,
    }
    
    if verbose:
        print(f"[{conv_index+1:02d}] {title[:48]:48s} | Turns: {turn_count:3d} | Hit@10: {conv_summary['hit@10']*100:5.1f}% | Recall@10: {conv_summary['recall@10']*100:5.1f}% | MRR: {conv_summary['mrr']:.3f} | {t_ingest:.1f}s")
        
    return conv_summary


def run_full_beam_evaluation(
    split: str = "100K",
    conv_indices: Optional[List[int]] = None,
    top_k: int = 10,
    output_path: Optional[str] = None,
    cache_dir: str = "data/beam_cache",
) -> Dict[str, Any]:
    if output_path is None:
        output_path = f"results/beam_{split.lower()}_full_results.json"
        
    print("=" * 96)
    print(f"STARTING BEAM BENCHMARK EVALUATION ({split} Split)")
    print("=" * 96)
    
    total_in_split = 20 if split == "100K" else 35
    if conv_indices is None:
        conv_indices = list(range(total_in_split))
        
    print(f"Evaluating {len(conv_indices)} conversations from {split} split.\n")
    print(f"{'#':4s} {'Conversation Title':48s} | {'Turns':5s} | {'Hit@10':7s} | {'Recall@10':9s} | {'MRR':5s} | {'Ingest':6s}")
    print("-" * 96)
    
    t_global_start = time.perf_counter()
    conv_results = []
    
    category_accs_hit10 = defaultdict(list)
    category_accs_recall10 = defaultdict(list)
    
    for idx in conv_indices:
        try:
            conv = fetch_beam_conversation(split=split, conv_index=idx, cache_dir=cache_dir)
        except Exception as e:
            print(f"[{idx+1:02d}] ERROR fetching conversation: {e}")
            continue
            
        res = evaluate_single_conversation(conv, conv_index=idx, top_k=top_k, verbose=True)
        conv_results.append(res)
        
        for cat, acc in res["category_hit10"].items():
            category_accs_hit10[cat].append(acc)
        for cat, rec in res["category_recall10"].items():
            category_accs_recall10[cat].append(rec)
            
        # Incremental save after each conversation
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "benchmark": "BEAM",
                "split": split,
                "completed_conversations": len(conv_results),
                "conversations": conv_results,
            }, f, indent=2)
            
    t_global = time.perf_counter() - t_global_start
    
    # Global aggregates
    total_turns = sum(c["turns"] for c in conv_results)
    total_questions = sum(c["questions"] for c in conv_results)
    mean_hit1 = sum(c["hit@1"] for c in conv_results) / len(conv_results)
    mean_hit5 = sum(c["hit@5"] for c in conv_results) / len(conv_results)
    mean_hit10 = sum(c["hit@10"] for c in conv_results) / len(conv_results)
    mean_recall10 = sum(c["recall@10"] for c in conv_results) / len(conv_results)
    mean_mrr = sum(c["mrr"] for c in conv_results) / len(conv_results)
    
    print("\n" + "=" * 80)
    print("FINAL BEAM BENCHMARK RESULTS SUMMARY (SMRITI)")
    print("=" * 80)
    print(f"Total Conversations Evaluated: {len(conv_results)}")
    print(f"Total Dialogue Turns:          {total_turns:,}")
    print(f"Total Probing Questions:       {total_questions:,}")
    print(f"Total Benchmark Time:          {t_global:.1f}s ({t_global/60:.2f} min)")
    print(f"Average Ingest Latency:        {(total_turns/sum(c['ingest_time'] for c in conv_results)):.1f} turns/sec")
    print("-" * 80)
    print(f"Overall Hit@1:                 {mean_hit1*100:6.2f}%")
    print(f"Overall Hit@5:                 {mean_hit5*100:6.2f}%")
    print(f"Overall Hit@10:                {mean_hit10*100:6.2f}%")
    print(f"Overall Recall@10:             {mean_recall10*100:6.2f}%")
    print(f"Overall MRR:                   {mean_mrr:6.3f}")
    print("=" * 80)
    
    print("\nCategory Breakdown Across All Conversations:")
    print(f"{'Category':28s} {'Hit@10':10s} {'Recall@10':12s}")
    print("-" * 52)
    category_summary = {}
    for cat in CATEGORIES:
        if cat in category_accs_hit10:
            h10 = sum(category_accs_hit10[cat]) / len(category_accs_hit10[cat]) * 100
            rec = sum(category_accs_recall10[cat]) / len(category_accs_recall10[cat]) * 100
            category_summary[cat] = {"hit@10": h10 / 100, "recall@10": rec / 100}
            print(f"{cat:28s} {h10:6.1f}%    {rec:6.1f}%")
    print("=" * 80)
    
    report = {
        "benchmark": "BEAM (Beyond a Million Tokens)",
        "split": split,
        "total_conversations": len(conv_results),
        "total_turns": total_turns,
        "total_questions": total_questions,
        "total_time_seconds": t_global,
        "metrics": {
            "overall_hit@1": mean_hit1,
            "overall_hit@5": mean_hit5,
            "overall_hit@10": mean_hit10,
            "overall_recall@10": mean_recall10,
            "overall_mrr": mean_mrr,
        },
        "by_category": category_summary,
        "conversations": conv_results,
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull benchmark results saved to {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate SMRITI on BEAM benchmark")
    parser.add_argument("--split", type=str, default="100K", choices=["100K", "500K", "1M"], help="BEAM split")
    parser.add_argument("--conv-index", type=int, default=None, help="Single conversation index to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all conversations in the split")
    parser.add_argument("--start", type=int, default=0, help="Start conversation index")
    parser.add_argument("--end", type=int, default=None, help="End conversation index (exclusive)")
    parser.add_argument("--top-k", type=int, default=10, help="Top K memories to retrieve")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON")
    parser.add_argument("--cache-dir", type=str, default="data/beam_cache", help="Directory to cache BEAM rows")
    args = parser.parse_args()
    
    total_default = 20 if args.split == "100K" else 35
    if args.conv_index is not None:
        conv_indices = [args.conv_index]
    elif args.all:
        conv_indices = list(range(total_default))
    elif args.end is not None:
        conv_indices = list(range(args.start, args.end))
    else:
        conv_indices = list(range(args.start, total_default))
        
    run_full_beam_evaluation(
        split=args.split,
        conv_indices=conv_indices,
        top_k=args.top_k,
        output_path=args.output,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
