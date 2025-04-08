import json
import re
from tabulate import tabulate  # For nice table output, install with: pip install tabulate

def clean_and_display_json(file_path):
    # Load the JSON data
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Create a list to store our results
    results = []
    
    # Process each response
    for entry in data:
        temp = entry.get("temperature", "unknown")
        top_p = entry.get("top_p", "unknown")
        response = entry.get("response", "")
        
        # Extract content without tags
        cleaned_content = []
        
        # Regular expression to find theory sections
        pattern = r'\[START\]([\s\S]*?)\[END\]'
        sections = re.findall(pattern, response)
        
        for section in sections:
            # Split the section into theory name and content
            parts = section.strip().split('\n[BREAK]\n', 1)
            if len(parts) == 2:
                theory_name = parts[0].strip()
                theory_content = parts[1].strip()
                
                # Extract propositions without the double curly braces
                propositions = re.findall(r'{{(.*?)}}', theory_content)
                propositions = [prop.strip() for prop in propositions]
                
                # Add to cleaned content
                cleaned_section = f"{theory_name}:\n" + "\n".join(f"- {prop}" for prop in propositions)
                cleaned_content.append(cleaned_section)
        
        # Join all cleaned sections with separators
        combined_content = "\n\n".join(cleaned_content)
        
        # Add to results
        results.append({
            "temperature": temp,
            "top_p": top_p,
            "content": combined_content,
            "num_theories": len(cleaned_content)
        })
    
    # Sort results by temperature and then by top_p
    results.sort(key=lambda x: (x["temperature"], x["top_p"]))
    
    # Display results in a table format for parameters
    headers = ["Temperature", "Top_p", "Number of Theories"]
    table_data = [[r["temperature"], r["top_p"], r["num_theories"]] for r in results]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Ask which result to view
    print("\nWhich result would you like to view? Enter a number from 1 to", len(results))
    try:
        choice = int(input()) - 1
        if 0 <= choice < len(results):
            print(f"\n--- Temperature: {results[choice]['temperature']}, Top_p: {results[choice]['top_p']} ---")
            print(results[choice]['content'])
        else:
            print("Invalid selection")
    except ValueError:
        print("Please enter a number")
    
    # Ask if user wants to save results to text files
    print("\nWould you like to save all results to separate text files? (yes/no)")
    save_choice = input().lower()
    if save_choice == 'yes':
        for i, r in enumerate(results):
            filename = f"response_temp_{r['temperature']}_top_p_{r['top_p']}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Temperature: {r['temperature']}, Top_p: {r['top_p']}\n\n")
                f.write(r['content'])
            print(f"Saved: {filename}")

if __name__ == "__main__":
    file_path = "results_gpt4o/gpt4_parameter_tests_20250324-173602.json"  # Update with your actual file path
    clean_and_display_json(file_path)