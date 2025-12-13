
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.engine import analyze_sentence

def main():
    print("="*50)
    print("      CAD-RAG System CLI Runner")
    print("="*50)
    
    # Simple interactive loop or predefined test
    print("\nRunning standard test suite...")
    test_sentences = [
        "All muslims should be kicked out",
        "Have a great day everyone!",
    ]
    
    for sentence in test_sentences:
        analyze_sentence(sentence)

    print("\nDone. You can modify scripts/run_rag.py to run custom queries.")

if __name__ == "__main__":
    main()
