import os
import logging
import json
from dotenv import load_dotenv
from dialectical_question_graph import DialecticalGraph
from datetime import datetime


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
    num_responses = 100  # Number of responses per level
    num_reasons = 100 # Number of reasons per thesis
    max_time_seconds = 1*60*60 #seconds 
    max_depth = None
    
    check_termination = False # Check for termination of the graph at each node add

    try:
        # Initialize graph
        logging.info(f"Initializing graph with:")
        logging.info(f"Central Question: {central_question}")
        logging.info(f"Number of responses per level: {num_responses}")
        logging.info(f"Maximum time: {max_time_seconds} seconds")
        
        graph = DialecticalGraph(
            api_key=api_key,
            central_question=central_question,
            prompt_dir=prompt_dir,
            num_responses=num_responses,
            num_reasons=num_reasons,
            max_time_seconds=max_time_seconds,
            nonsense_threshold=95, # Threshold for nonsense detection
            view_identity_threshold=95, # Threshold for view identity detection
            check_termination=check_termination
        )
    
            
        # Create timestamp for consistent file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./graph_export_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the graph to a file
        graph_file_path = os.path.join(save_dir, "graph.json")
        with open(graph_file_path, 'w') as f:
            json.dump(graph.graph, f, indent=2)
        logging.info(f"Graph saved to {graph_file_path}")

        # Save edges to a separate file
        edges_file_path = os.path.join(save_dir, "edges.json")
        with open(edges_file_path, 'w') as f:
            json.dump(graph.edges, f, indent=2)
        logging.info(f"Edges saved to {edges_file_path}")
        
        if check_termination:

            # Save FAISS index and node mapping
            logging.info("Saving embeddings and FAISS index...")
            graph.save_faiss_index(save_dir)
            logging.info(f"Embeddings and index saved to {save_dir}")

            # Verify FAISS index
            total_nodes = len(graph.graph)
            total_embeddings = graph.faiss_index.ntotal
            logging.info(f"Total nodes: {total_nodes}")
            logging.info(f"Total embeddings: {total_embeddings}")
            if total_nodes - 1 != total_embeddings:  # -1 for central question
                logging.warning("Mismatch between number of nodes and embeddings")

    except Exception as e:
        logging.error(f"Error during test: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()