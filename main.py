import os
import logging
import json
from dotenv import load_dotenv
from dialectical_question_graph import DialecticalGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")

    # Test configuration
    central_question = "What is knowledge?"
    prompt_dir = "./prompts"
    num_responses = 2
    max_depth = 3

    try:
        # Initialize graph
        logging.info(f"Initializing graph with:")
        logging.info(f"Central Question: {central_question}")
        logging.info(f"Number of responses per level: {num_responses}")
        logging.info(f"Maximum depth: {max_depth}")
        
        graph = DialecticalGraph(
            api_key=api_key,
            central_question=central_question,
            prompt_dir=prompt_dir,
            num_responses=num_responses,
            max_depth=max_depth
        )
        
    
        # Test basic graph operations
        logging.info("Testing graph operations...")
        root_children = graph.get_children(central_question)
        logging.info(f"Root node has {len(root_children)} children")
        
        for child_id in root_children:
            node_content = graph.get_node_content(child_id)
            logging.info(f"Child node {child_id}: {node_content['type']} - {node_content['content'][:50]}...")
            
        # Save the graph to a file
        graph_file_path = "./graph.json"
        with open(graph_file_path, 'w') as f:
            json.dump(graph.graph, f, indent=2)
        logging.info(f"Graph saved to {graph_file_path}")

    except Exception as e:
        logging.error(f"Error during test: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()