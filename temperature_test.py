import os
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

def load_prompt(path):
    """Load the prompt from a text file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def query_gpt4(client, prompt, temperature, top_p, model="gpt-4o"):
    """Query GPT-4 with the given parameters and return the response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p
        )
        return {
            "temperature": temperature,
            "top_p": top_p,
            "response": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    except Exception as e:
        print(f"Error with temperature={temperature}, top_p={top_p}: {e}")
        # Wait before retrying to avoid rate limits
        time.sleep(20)
        return {
            "temperature": temperature,
            "top_p": top_p,
            "response": f"ERROR: {str(e)}",
            "finish_reason": "error",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

def main():
    parser = argparse.ArgumentParser(description="Test GPT-4o with various parameter settings")
    parser.add_argument("--prompt_path", type=str, default="prompts/thesis_prompt.txt", 
                        help="Path to the prompt file")
    parser.add_argument("--api_key", type=str, default=None, 
                        help="OpenAI API key (if not provided, will use OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model to use (default: gpt-4o)")
    parser.add_argument("--output", type=str, default="results_gpt4o",
                        help="Output directory for results_gpt4o")
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=args.api_key if args.api_key else os.environ.get("OPENAI_API_KEY"))
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set of temperature and top_p values to test
    temperatures = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]
    top_ps = [0.3, 0.5, 0.7, 0.9, 1.0]
    
    # Load the prompt and replace placeholders
    prompt = load_prompt(args.prompt_path)
    prompt = prompt.replace("{central_question}", "What is Knowledge?")
    prompt = prompt.replace("{num_responses}", "100")
    print(f"Loaded prompt from {args.prompt_path} ({len(prompt)} characters)")
    
    # Run the tests
    results = []
    total_tests = len(temperatures) * len(top_ps)
    
    print(f"Running {total_tests} tests with different parameter combinations...")
    for temp in temperatures:
        for top_p in top_ps:
            print(f"Testing with temperature={temp:.1f}, top_p={top_p:.1f}")
            result = query_gpt4(client, prompt, temp, top_p, model=args.model)
            results.append(result)
            
            # Save intermediate results in case of interruption
            pd.DataFrame(results).to_csv(output_dir / "intermediate_results.csv", index=False)
            
            # Wait to avoid rate limits
            time.sleep(5)
    
    # Save results to CSV and JSON
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save to CSV (easier for analysis)
    df = pd.DataFrame(results)
    csv_path = output_dir / f"gpt4_parameter_tests_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # Save to JSON (preserves all data)
    json_path = output_dir / f"gpt4_parameter_tests_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Create a summary of results
    summary = pd.DataFrame([
        {
            "temperature": r["temperature"],
            "top_p": r["top_p"],
            "completion_tokens": r["usage"]["completion_tokens"],
            "total_tokens": r["usage"]["total_tokens"],
            "finish_reason": r["finish_reason"],
            "response_length": len(r["response"]),
        }
        for r in results
    ])
    
    summary_path = output_dir / f"summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    
    print(f"\nTesting completed! Results saved to:")
    print(f"- Full results (CSV): {csv_path}")
    print(f"- Full results (JSON): {json_path}")
    print(f"- Summary: {summary_path}")
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    main()