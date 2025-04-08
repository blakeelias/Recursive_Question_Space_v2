import logging
import faiss
import numpy as np
import os
import json
from datetime import datetime

def _calculate_embedding(self, text: str) -> np.ndarray:
    """Calculate embedding for node text using OpenAI's text-embedding-3-small model"""
    # Skip if termination checks are disabled
    if not self.check_termination:
        logging.info("Skipping embedding calculation (check_termination=False)")
        return np.zeros((1, 1536), dtype=np.float32)  # Return dummy embedding
        
    try:
        # Get embedding from OpenAI API
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        # Convert to numpy array
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Reshape to (1, dimension) as required by FAISS
        return embedding.reshape(1, -1)
        
    except Exception as e:
        logging.error(f"Error calculating embedding: {e}")
        return np.zeros((1, 1536), dtype=np.float32)  # Return dummy embedding on error
    
    
def _initialize_faiss(self):
    """Initialize or reset FAISS index - only used when check_termination is True"""
    if self.check_termination:
        dimension = 1536  # for text-embedding-3-small
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.node_id_to_index = {}
        logging.info("FAISS index initialized")
    else:
        # Create empty placeholders when not using termination checks
        self.faiss_index = None
        self.node_id_to_index = {}
        logging.info("FAISS index disabled (check_termination=False)")
        
    
def load_faiss_index(self) -> None:
    """Load most recent FAISS index and mapping if they exist"""
    # Skip if termination checks are disabled
    if not self.check_termination:
        logging.info("Skipping FAISS loading (check_termination=False)")
        return
        
    try:
        # Get timestamp from object attribute instead of graph
        timestamp = getattr(self, "embedding_save_timestamp", None)
        if not timestamp:
            # For backward compatibility, check if it's in the graph
            if "embedding_save_timestamp" in self.graph:
                timestamp = self.graph["embedding_save_timestamp"]
                # Remove it from the graph to clean up the structure
                del self.graph["embedding_save_timestamp"]
                # Store it as a proper attribute
                self.embedding_save_timestamp = timestamp
                logging.info("Migrated embedding timestamp from graph to attribute")
            else:
                return
                
        # Load FAISS index
        index_path = os.path.join(self.save_dir, f"faiss_index_{timestamp}.index")
        if os.path.exists(index_path):
            self.faiss_index = faiss.read_index(index_path)
            
        # Load node mapping
        mapping_path = os.path.join(self.save_dir, f"node_mapping_{timestamp}.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                self.node_id_to_index = json.load(f)
                
    except Exception as e:
        raise Exception(f"Error loading FAISS index and mapping: {e}")
        
def save_faiss_index(self, save_path: str = None) -> None:
    """Save FAISS index and node mapping to disk
    
    Args:
        save_path: Directory to save index. If None, uses self.save_dir
    """
    # Skip if termination checks are disabled
    if not self.check_termination or self.faiss_index is None:
        logging.info("Skipping FAISS saving (check_termination=False or faiss_index=None)")
        return
        
    if save_path is None:
        save_path = self.save_dir
        
    try:
        # Save FAISS index
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        index_path = os.path.join(save_path, f"faiss_index_{timestamp}.index")
        faiss.write_index(self.faiss_index, index_path)
        
        # Save node ID to index mapping
        mapping_path = os.path.join(save_path, f"node_mapping_{timestamp}.json")
        with open(mapping_path, 'w') as f:
            json.dump(self.node_id_to_index, f)
            
        # Store timestamp in a separate attribute (not in the graph)
        self.embedding_save_timestamp = timestamp
        
        # Save updated graph state
        self._save_state()
        
    except Exception as e:
        raise Exception(f"Error saving FAISS index and mapping: {e}")