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
    # Test configuration
    central_question = "What is virtue?"
    print("Central Question: ", central_question)
    num_responses = 100  # Max number of responses per level. Prompt instructs not to reach max if possible but rather to stop at a comprehensive number
    num_reasons = 100 # Max number of reasons per thesis. Prompt instructs not to reach max if possible but rather to stop at a comprehensive number
    max_time_seconds = 30*60 #seconds. Maximum time until the graph gen is terminated 
    max_depth = None # Maximum depth of the graph
    
    check_termination = False # Check for termination of the graph at each node add
    
    # Define save directory for the graph state backup
    save_dir = "./temp"  # Where your pickle file is stored

    try:
        # Try to load existing graph
        try:
            logging.info(f"Attempting to load graph from {save_dir}")
            graph = DialecticalGraph.load(save_dir=save_dir, default_question=central_question)
            logging.info(f"Successfully loaded existing graph")
            
            # Important: We need to call initialize_graph() manually after loading
            # since we're replacing the normal initialization flow
            graph.initialize_graph()
            
        except Exception as e:
            logging.info(f"Could not load existing graph: {e}")
            logging.info(f"Creating new graph instead")
            
            # Initialize new graph
            logging.info(f"Initializing graph with:")
            logging.info(f"Central Question: {central_question}")
            logging.info(f"Number of responses per level: {num_responses}")
            logging.info(f"Maximum time: {max_time_seconds} seconds")
            logging.info(f"Maximum depth: {max_depth}")
            logging.info(f"Check for termination: {check_termination}")
            
            graph = DialecticalGraph(
                central_question=central_question,
                num_responses=num_responses,
                num_reasons=num_reasons,
                max_depth=max_depth,
                nonsense_threshold=95, # Threshold for nonsense detection
                view_identity_threshold=95, # Threshold for view identity detection
                check_termination=check_termination,
                save_dir=save_dir  # Ensure same save directory
            )
    
        # Create timestamp for consistent file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = f"./Graph_Exports/graph_export_{timestamp}"
        os.makedirs(export_dir, exist_ok=True)
        
        # Save the graph to a file
        graph_file_path = os.path.join(export_dir, "graph.json")
        with open(graph_file_path, 'w') as f:
            json.dump(graph.graph, f, indent=2)
        logging.info(f"Graph saved to {graph_file_path}")

        # Save edges to a separate file
        edges_file_path = os.path.join(export_dir, "edges.json")
        with open(edges_file_path, 'w') as f:
            json.dump(graph.edges, f, indent=2)
        logging.info(f"Edges saved to {edges_file_path}")
        
        if check_termination:
            # Save FAISS index and node mapping
            logging.info("Saving embeddings and FAISS index...")
            graph.save_faiss_index(export_dir)
            logging.info(f"Embeddings and index saved to {export_dir}")

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