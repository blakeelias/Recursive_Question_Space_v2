import os
import time
import logging
import asyncio
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv

prompt_dir: str = "./prompts"

PROMPT_FILES = {
    "thesis": "thesis_prompt.txt",
    "reasons": "reasons_prompt.txt",
    "antithesis": "antithesis_prompt.txt",
    "direct_reply": "direct_reply_prompt.txt",
    "synthesis": "synthesis_prompt.txt",
    "view_identity": "view_identity_prompt.txt",
    "nonsense": "nonsense_prompt.txt"
}

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
#api_key = os.getenv('DEEPSEEK_API_KEY')
if not api_key:
    raise ValueError("API key not found in environment variables")
client = OpenAI(api_key=api_key) #, base_url="https://api.deepseek.com"
async_client = AsyncOpenAI(api_key=api_key) #, base_url="https://api.deepseek.com"
default_model="gpt-4o" # Default model for completions deepseek-reasoner deepseek-chat

def load_prompts() -> Dict[str, str]:
    """Load all prompt templates from files"""
    prompts = {}
    try:
        for prompt_type, filename in PROMPT_FILES.items():
            file_path = os.path.join(prompt_dir, filename)
            with open(file_path, 'r') as file:
                prompts[prompt_type] = file.read().strip()
        return prompts
    except Exception as e:
        raise Exception(f"Error loading prompts: {e}")

def generate_completion(prompt: str, system_role: str ="", temperature = 0.7, top_p = 0.3) -> str:
    """Generate a completion with error handling, timeout, and simple retry logic"""
    max_retries = 2  # Try up to 3 times total (initial attempt + 2 retries)
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Add timeout parameter to the API call (30 seconds)
            response = client.chat.completions.create(
                model=default_model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                #temperature=temperature,
                # top_p=top_p,
               timeout=30  # Add 30-second timeout to prevent hanging
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            # Log the error with retry information
            error_msg = f"Error in API call (attempt {retry_count + 1}/{max_retries + 1}): {e}"
            logging.warning(error_msg)
            
            # If we've used all retries, give up and return an error message
            if retry_count >= max_retries:
                logging.error(f"All {max_retries + 1} attempts failed, giving up.")
                # Return a message that can be safely parsed
                return f"[START]API Error[BREAK]Failed after {max_retries + 1} attempts: {e}[END]"
            
            # Otherwise wait a bit before retrying (simple backoff: 2s, then 4s)
            retry_delay = 2 * (retry_count + 1)
            logging.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_count += 1

async def generate_completion_async(prompt: str, system_role: str = "", temperature=0.7, top_p=0.3) -> str:
    """Async version of generate_completion for concurrent processing"""
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            response = await async_client.chat.completions.create(
                model=default_model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                #temperature=temperature,
                #top_p=top_p,
                timeout=30
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            error_msg = f"Error in async API call (attempt {retry_count + 1}/{max_retries + 1}): {e}"
            logging.warning(error_msg)
            
            if retry_count >= max_retries:
                logging.error(f"All async {max_retries + 1} attempts failed, giving up.")
                return f"[START]API Error[BREAK]Failed after {max_retries + 1} attempts: {e}[END]"
            
            retry_delay = 2 * (retry_count + 1)
            logging.info(f"Retrying async in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            retry_count += 1

async def generate_completions_concurrent(prompts: List[str], system_role: str = "", temperature=0.7, top_p=0.3) -> List[str]:
    """Generate multiple completions concurrently using asyncio"""
    tasks = []
    for prompt in prompts:
        tasks.append(generate_completion_async(prompt, system_role, temperature, top_p))
    
    return await asyncio.gather(*tasks)

def generate_completions_batch(prompts: List[str], system_role: str = "", temperature=0.7, top_p=0.3) -> List[str]:
    """
    Generate completions for multiple prompts concurrently
    This provides a way to process multiple prompts efficiently
    """
    return asyncio.run(generate_completions_concurrent(prompts, system_role, temperature, top_p))
            
def parse_llm_output(llm_response: str) -> Optional[List[Tuple[str, str]]]:
    """
    Parse LLM response into list of (summary, description) tuples.
    Automatically appends [END] tag if missing.
    Ignores any text before the first [START] tag.
    
    Returns None if there are any validation errors or parsing issues
    instead of raising exceptions.
    """
    
    #print("LLM Response: ", llm_response)
    
    # Find the first occurrence of [START] and ignore everything before it
    first_start = llm_response.find('[START]')
    if first_start == -1:
        print(f"No [START] tags found in response: {llm_response}")
        return None
        
    llm_response = llm_response[first_start:]
    
    # Split into individual items
    items = llm_response.split('[START]')
    items = [item.strip() for item in items if item.strip()]
    
    if not items:
        print("No valid items found after splitting by [START]")
        return None
    
    parsed_items = []
    try:
        for item in items:
            # Append [END] tag if missing
            if not item.endswith('[END]'):
                item = item + '[END]'
                
            # Remove [END] tag
            item = item[:-5].strip()
            
            # Split into summary and description
            parts = item.split('[BREAK]')
            if len(parts) != 2:
                print(f"Item does not have exactly 2 parts: {item}")
                # Instead of raising an error, we'll return None
                return None
            
            summary, description = parts
            parsed_items.append((summary.strip(), description.strip()))
        
        return parsed_items
    
    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        return None

def parse_batch_outputs(batch_responses: List[str]) -> List[Optional[List[Tuple[str, str]]]]:
    """
    Parse multiple LLM responses from a batch
    Returns a list of parsed responses, each parsed using parse_llm_output
    """
    return [parse_llm_output(response) for response in batch_responses]
    
if __name__ == "__main__":
    print("Prompt LLM Module")
    # Example usage
    prompts = load_prompts()
    print("Loaded Prompts")
    # Generate a completion using the thesis prompt
    thesis_prompt = prompts['thesis'].format(
        central_question="What is knowledge?",
        num_responses=3,
        num_reasons=2
    )
    
    response = generate_completion(thesis_prompt)
    print(response)
    # Parse the LLM output
    parsed_response = parse_llm_output(response)
    
    if parsed_response:
        print("Parsed Response:")
        for summary, description in parsed_response:
            print(f"Summary: {summary}, Description: {description}")
    else:
        print("Failed to parse LLM output.")
        