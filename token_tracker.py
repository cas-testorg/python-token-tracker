import sys
import os

try:
    import tiktoken
except ImportError:
    print("Error: 'tiktoken' package is not installed.")
    print("Please install it by running: pip install tiktoken")
    sys.exit(1)

def count_tokens(text, model="gpt-4"):
    """Counts the number of tokens in a string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base if model not explicitly recognized
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def analyze_workflow_files(directory, model="gpt-4"):
    """Scans a directory or specific files to estimate context window token weight."""
    print(f"\n=== Token Analysis Profile (Model: {model}) ===")
    print(f"{'File / Source':<50} | {'Token Count':<12}")
    print("-" * 66)
    
    total_tokens = 0
    
    # If a specific directory/file is provided
    if os.path.isfile(directory):
        files = [directory]
    elif os.path.isdir(directory):
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(directory) for f in fn]
    else:
        print(f"Path not found: {directory}")
        return

    for file_path in files:
        # Skip binary files or common ignore directories
        if any(ignored in file_path for ignored in ['.git', '__pycache__', 'node_modules', '.cursor']):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                tokens = count_tokens(content, model)
                total_tokens += tokens
                # Truncate path for cleaner printing
                display_name = os.path.relpath(file_path, os.path.dirname(directory)) if os.path.isdir(directory) else os.path.basename(file_path)
                if len(display_name) > 47:
                    display_name = "..." + display_name[-44:]
                print(f"{display_name:<50} | {tokens:<12,}")
        except Exception as e:
            pass
            
    print("-" * 66)
    print(f"{'TOTAL ESTIMATED WORKFLOW TOKENS':<50} | {total_tokens:<12,}\n")

if __name__ == "__main__":
    path_to_analyze = input("Enter the file or directory path to analyze (or press Enter for current dir): ").strip()
    if not path_to_analyze:
        path_to_analyze = "."
    
    model_choice = input("Enter target model (e.g., gpt-4, gpt-3.5-turbo, claude) [Default gpt-4]: ").strip()
    if not model_choice:
        model_choice = "gpt-4"
        
    analyze_workflow_files(path_to_analyze, model_choice)
