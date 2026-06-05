from langchain_openai import ChatOpenAI
from smriti_memcore.compressors.code_crusher import crush_code

def run_fidelity_test():
    print("--- Running Targeted Fidelity Test with Qwen3.5 ---")
    
    # 1. Large code payload
    raw_code = '''
class PaymentProcessor:
    """Handles all Stripe transactions."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.retries = 3
        
    def charge(self, amount: float, currency: str = "USD") -> bool:
        # Complex logic to charge via stripe
        print(f"Charging {amount} {currency}")
        return True
        
    def refund(self, transaction_id: str):
        """Refund a previous transaction."""
        print(f"Refunding {transaction_id}")
        pass
'''
    
    payload = raw_code + ("\n# padding " * 20) * 10
    
    # 2. Crush the code directly
    compressed = crush_code(payload)
    
    print(f"Original size: {len(payload)} chars")
    print(f"Compressed size: {len(compressed)} chars")
    
    print("\nInjected Context to LLM:")
    print("-" * 40)
    print(compressed)
    print("-" * 40)
    
    # 3. Ask a local LLM
    print("\nAsking LLM (Ollama qwen3.5) based ONLY on compressed context...")
    llm = ChatOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", model="qwen3.5", temperature=0.0)
    
    prompt = f"""You are an assistant. Using ONLY the following context, answer the question.
    
CONTEXT:
{compressed}

QUESTION: What are the exact arguments for the charge method in PaymentProcessor?
"""
    
    response = llm.invoke(prompt)
    print("\nLLM Answer:")
    print(">" * 40)
    print(response.content)
    print(">" * 40)
    
if __name__ == "__main__":
    run_fidelity_test()
