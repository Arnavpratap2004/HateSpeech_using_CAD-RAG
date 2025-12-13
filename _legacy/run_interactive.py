import sys
import os

# Ensure script directory is in path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from cad_rag_engine import CADRAGEngine

def main():
    print("=" * 60)
    print("CAD-RAG Interactive Console")
    print("=" * 60)

    # 1. Initialize Engine (Once)
    try:
        engine = CADRAGEngine()
        engine.initialize()
    except Exception as e:
        print(f"CRITICAL ERROR during initialization: {e}")
        return

    print("\n" + "=" * 60)
    print("READY! Type a sentence to analyze. ('exit' or 'quit' to stop)")
    print("=" * 60)

    # 2. Loop
    while True:
        try:
            user_input = input("\n>> Enter sentence: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            
            if not user_input:
                continue

            print("Analyzing...", end="\r")
            
            # Analyze
            result = engine.analyze_sentence(user_input)
            
            # Print Result
            print("\n" + "-" * 40)
            print(f"Category: {result['category']}")
            
            # ML Result
            model_res = result['model_result']
            print(f"ML Model: {model_res.get('result', 'N/A')}")
            if model_res.get('labels'):
                print(f"Labels:   {model_res['labels']}")
            
            # LLM
            print(f"\nLLM Rationale:\n{result['llm_rationale']}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError processing sentence: {e}")

if __name__ == "__main__":
    main()
