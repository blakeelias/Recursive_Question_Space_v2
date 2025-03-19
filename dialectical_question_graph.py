from openai import OpenAI
from typing import Dict, List, Tuple, Optional, Set
import os
import uuid
from datetime import datetime
import json
import logging
import faiss # For similarity search
import numpy as np


class DialecticalGraph:
    PROMPT_FILES = {
        "thesis": "thesis_prompt.txt",
        "antithesis": "antithesis_prompt.txt",
        "synthesis": "synthesis_prompt.txt",
        "view_identity": "view_identity_prompt.txt",
        "nonsense": "nonsense_prompt.txt",
        "reasons": "reasons_prompt.txt"
    }
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
            
    def __init__(self, api_key: str, central_question: str, nonsense_threshold = 95, view_identity_threshold = 95, prompt_dir: str = "./prompts", 
                num_responses: int = 3, num_reasons: int = 10, max_depth: Optional[int] = None, 
                max_time_seconds: Optional[float] = None, save_dir: str = "./temp", check_termination: bool = True):
        
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.client = OpenAI(api_key=api_key)
        self.default_model="gpt-4o" # Default model for completions
        self.max_depth = max_depth
        self.max_time_seconds = max_time_seconds
        # Always reset the start time when creating a new instance or resuming
        self.start_time = datetime.now() if max_time_seconds is not None else None
        self.nonsense_threshold = nonsense_threshold
        self.view_identity_threshold = view_identity_threshold
        self.num_responses = num_responses
        self.num_reasons = num_reasons
        self.central_question_text = central_question  # Store the text separately
        self.check_termination = check_termination  # Flag to enable/disable termination checks
        
        # Only initialize FAISS if we need it
        if self.check_termination:
            dimension = 1536  # for text-embedding-3-small
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.node_id_to_index = {}
            logging.info("FAISS index initialized")
        else:
            self.faiss_index = None
            self.node_id_to_index = {}
            logging.info("FAISS index not initialized (check_termination=False)")
        
        # Track processed nodes by type
        self.processed_theses = set()
        self.processed_antitheses = set()
        self.processed_syntheses = set()
        
        # Set prompt directory and load prompts
        self.prompt_dir = prompt_dir
        self.prompts = self._load_prompts()
        
        # Track whether we're resuming from a save
        resuming_from_save = False
        
        # Try to load from most recent save
        try:
            latest_save = self._get_latest_save()
            if latest_save:
                self._load_from_save(latest_save)
                resuming_from_save = True
                print(f"Successfully loaded from save: {latest_save}")
        except Exception as e:
            print(f"Warning: Failed to load from save: {e}")
        
        # If we loaded from a save, try to restore FAISS index only if termination checks are enabled
        if resuming_from_save and self.check_termination:
            try:
                self.load_faiss_index()
                print("Successfully loaded FAISS index")
            except Exception as e:
                print(f"Warning: Failed to load FAISS index: {e}")
        
        # Initialize new graph if no save exists or loading failed
        if not hasattr(self, 'graph') or not self.graph:
            self.graph = {}
            self.edges = []
            
            # Generate UUID for central question
            self.central_question = str(uuid.uuid4())
            
            # Add central question node with UUID
            self.graph[self.central_question] = {
                "summary": central_question,
                "content": central_question,
                "node_type": "question",
                "parent_id": None,
                "depth": 0,
                "terminal": False,
                "nonsense": False,
                "identical_to": None,
                "is_central_question": True  # Mark as central question
            }
            
            # Save initial state
            try:
                self._save_state()
            except Exception as e:
                print(f"Warning: Failed to save initial state: {e}")
        
        # Whether we're resuming or starting fresh, continue the graph initialization
        # This will pick up where it left off if resuming from save
        self.initialize_graph()
        
    def _check_time_exceeded(self) -> bool:
        """Check if maximum allowed computation time has been exceeded"""
        if self.max_time_seconds is None or self.start_time is None:
            return False
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > self.max_time_seconds:
            print(f"Time limit of {self.max_time_seconds} seconds exceeded. Elapsed time: {elapsed:.2f} seconds")
            return True
        return False

    def _check_depth_exceeded(self, node_id: str) -> bool:
        """Check if maximum depth has been exceeded"""
        if self.max_depth is None:
            return False
        current_depth = self.graph[node_id]["depth"]
        if current_depth >= self.max_depth:
            print(f"Maximum depth of {self.max_depth} reached at node {node_id}")
            return True
        return False  
    
    def _get_save_filename(self) -> str:
        """Generate a filename for the current save"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.save_dir, f"graph_state_{timestamp}.json")
    
    def _get_latest_save(self) -> Optional[str]:
        """Get the most recent valid save file if it exists"""
        if not os.path.exists(self.save_dir):
            return None
            
        save_files = [f for f in os.listdir(self.save_dir) if f.startswith("graph_state_")]
        if not save_files:
            return None
        
        # Sort files by creation time, newest first
        save_files.sort(key=lambda x: os.path.getctime(os.path.join(self.save_dir, x)), reverse=True)
        
        # Try each save file until we find a valid one
        for save_file in save_files:
            full_path = os.path.join(self.save_dir, save_file)
            try:
                # Check if file is empty
                if os.path.getsize(full_path) == 0:
                    os.remove(full_path)  # Remove empty file
                    continue
                    
                # Try to read and parse the JSON
                with open(full_path, 'r') as f:
                    state = json.load(f)
                    
                # Verify required keys exist
                required_keys = ["graph", "edges", "central_question", "num_responses", "max_depth"]
                if all(key in state for key in required_keys):
                    return full_path
                    
            except (json.JSONDecodeError, KeyError):
                # If file is corrupted, remove it
                os.remove(full_path)
                continue
                
        return None
        
    def _save_state(self) -> None:
        """Save current graph state to file with error handling"""
        state = {
            "graph": self.graph,
            "edges": self.edges,
            "central_question": self.central_question,  # This is now the UUID
            "central_question_text": self.central_question_text,  # The actual text
            "num_responses": self.num_responses,
            "num_reasons": self.num_reasons,
            "max_depth": self.max_depth,
            "nonsense_threshold": self.nonsense_threshold,
            "view_identity_threshold": self.view_identity_threshold,
            "check_termination": self.check_termination,
            # Save processing state to track progress
            "processed_theses": list(getattr(self, "processed_theses", set())),
            "processed_antitheses": list(getattr(self, "processed_antitheses", set())),
            "processed_syntheses": list(getattr(self, "processed_syntheses", set())),
            # Store FAISS timestamp as a top-level attribute, not in the graph
            "embedding_save_timestamp": getattr(self, "embedding_save_timestamp", None)
        }
        
        filename = self._get_save_filename()
        temp_filename = filename + '.tmp'
        
        try:
            # Write to temporary file first
            with open(temp_filename, 'w') as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Ensure write is complete
                
            # Rename temporary file to final filename (atomic operation)
            os.replace(temp_filename, filename)
            
            # Keep only last 5 valid saves
            save_files = sorted([f for f in os.listdir(self.save_dir) if f.startswith("graph_state_")],
                            key=lambda x: os.path.getctime(os.path.join(self.save_dir, x)))
            
            while len(save_files) > 5:
                try:
                    os.remove(os.path.join(self.save_dir, save_files.pop(0)))
                except OSError:
                    continue
                    
        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise Exception(f"Failed to save graph state: {e}")
            
    def _load_from_save(self, save_file: str) -> None:
        """Load graph state from a save file with validation"""
        try:
            with open(save_file, 'r') as f:
                state = json.load(f)
                    
            # Validate required keys
            required_keys = ["graph", "edges", "central_question"]
            if not all(key in state for key in required_keys):
                raise KeyError("Save file missing required keys")
                    
            # Load state
            self.graph = state["graph"]
            self.edges = state["edges"]
            self.central_question = state["central_question"]
                
            # Load the central question text
            self.central_question_text = state.get("central_question_text", 
                                                self.graph[self.central_question].get("content", "Unknown Question"))
                
            # Load additional attributes with defaults if not present
            self.num_responses = state.get("num_responses", 3)
            self.num_reasons = state.get("num_reasons", 10)
            self.max_depth = state.get("max_depth", None)
            self.nonsense_threshold = state.get("nonsense_threshold", 95)
            self.view_identity_threshold = state.get("view_identity_threshold", 95)
            self.check_termination = state.get("check_termination", True)
                
            # Load embedding timestamp as a separate attribute
            self.embedding_save_timestamp = state.get("embedding_save_timestamp", None)
                
            # For backward compatibility, if timestamp is in the graph, migrate it
            if "embedding_save_timestamp" in self.graph:
                self.embedding_save_timestamp = self.graph["embedding_save_timestamp"]
                # Remove it from the graph to clean up the structure
                del self.graph["embedding_save_timestamp"]
                logging.info("Migrated embedding timestamp from graph to attribute")
                
            # Load processing state or initialize if not present
            self.processed_theses = set(state.get("processed_theses", []))
            self.processed_antitheses = set(state.get("processed_antitheses", []))
            self.processed_syntheses = set(state.get("processed_syntheses", []))
                
            # Handle backwards compatibility with old saves where central question wasn't a UUID
            # This converts old format to new format if needed
            self._ensure_central_question_has_uuid()
                
        except (json.JSONDecodeError, KeyError) as e:
            # If loading fails, remove corrupted save and start fresh
            os.remove(save_file)
            raise Exception(f"Failed to load save file: {e}")
            
    def initialize_graph(self) -> None:
        """
        Initialize or continue the dialectical graph with a simplified, deterministic order.
        Process steps:
        1. Generate thesis nodes from central question
        2. Add reasons for each thesis node
        3. Generate antithesis nodes for each thesis
        4. Generate synthesis nodes for each antithesis
        5. Repeat steps 3-4 until termination or maximum depth
        """
        logging.info(f"Starting/resuming dialectical exploration for question: {self.central_question_text}")
        logging.info(f"Termination checks enabled: {self.check_termination}")
        
        # Initialize tracking sets if they don't exist
        if not hasattr(self, 'theses_with_reasons'):
            self.theses_with_reasons = set()
        if not hasattr(self, 'theses_with_antitheses'):
            self.theses_with_antitheses = set()
        
        # Step 1: Generate initial theses nodes if they don't exist yet
        if not self.get_children(self.central_question):
            logging.info("Generating initial theses...")
            self._generate_theses()
            if self._check_time_exceeded():
                logging.info("Time limit reached after generating initial theses")
                return
        
        # Step 2: Generate reasons for each thesis (if not already done)
        thesis_nodes = self.get_children_by_type(self.central_question, "thesis")
        for thesis_id in thesis_nodes:
            if thesis_id not in self.theses_with_reasons:
                logging.info(f"Generating reasons for thesis {thesis_id}")
                self._generate_reasons(thesis_id)
                self.theses_with_reasons.add(thesis_id)
                self._save_state()
                
                if self._check_time_exceeded():
                    logging.info("Time limit reached during reason generation")
                    return
        
        # Steps 3-4: Cycle between antitheses and syntheses until termination
        current_depth = 1  # Start with theses at depth 1
        max_iterations = 1000  # Safety limit to prevent infinite loops
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            logging.info(f"Starting iteration {iterations} at depth {current_depth}")
            
            # Process all nodes at the current depth
            if current_depth % 2 == 1:  # Odd depths are theses/syntheses
                # Generate antitheses for theses/syntheses at this depth
                has_progress = self._process_thesis_synthesis_level(current_depth)
                logging.info(f"Processed thesis/synthesis nodes at depth {current_depth}, progress: {has_progress}")
            else:  # Even depths are antitheses
                # Generate syntheses for antitheses at this depth
                has_progress = self._process_antithesis_level(current_depth)
                logging.info(f"Processed antithesis nodes at depth {current_depth}, progress: {has_progress}")
            
            # Check if we should continue
            if not has_progress:
                # If no nodes were processed at this depth, try the next depth
                current_depth += 1
                logging.info(f"No progress at depth {current_depth-1}, moving to depth {current_depth}")
                
                # If we've reached max_depth, stop
                if self.max_depth is not None and current_depth > self.max_depth:
                    logging.info(f"Maximum depth of {self.max_depth} reached")
                    break
            
            # Check time limit
            if self._check_time_exceeded():
                logging.info("Time limit reached during processing")
                return
                
            # Save state after each level is processed
            self._save_state()
        
        logging.info("Graph exploration completed")

    def _generate_theses(self) -> None:
        """Generate initial theses in response to the central question"""
        theses = self.generate_theses()
        for thesis in theses:
            thesis_id = self.add_node(
                summary=thesis[0],
                content=thesis[1],
                node_type="thesis",
                parent_id=self.central_question
            )
            
            if self._check_time_exceeded():
                return

    def _generate_reasons(self, thesis_id: str) -> None:
        """Generate supporting reasons for a thesis"""
        thesis_data = self.graph[thesis_id]
        thesis_content = (thesis_data["summary"], thesis_data["content"])
        
        try:
            # Generate reasons
            reasons = self.generate_reasons(thesis_content)
            
            # Add reason nodes
            for reason in reasons:
                self.add_node(
                    summary=reason[0],
                    content=reason[1],
                    node_type="reason",
                    parent_id=thesis_id
                )
                
                if self._check_time_exceeded():
                    return
                    
        except Exception as e:
            logging.error(f"Error generating reasons for thesis {thesis_id}: {e}")
    
    def _process_thesis_synthesis_level(self, depth: int) -> bool:
        """
        Process all thesis/synthesis nodes at the given depth.
        Generate antitheses for each node that hasn't been processed yet.
        Returns True if any nodes were processed.
        """
        # Find all thesis/synthesis nodes at this depth that haven't been processed
        nodes_to_process = []
        
        for node_id, node_data in self.graph.items():
            if node_data.get("depth") == depth and not node_data.get("terminal", False):
                if node_data.get("node_type") == "thesis":
                    # For thesis nodes, check if they have antitheses yet
                    if node_id not in self.theses_with_antitheses:
                        nodes_to_process.append(node_id)
                elif node_data.get("node_type") == "synthesis":
                    # For synthesis nodes, check if they have been processed
                    if node_id not in self.processed_syntheses:
                        nodes_to_process.append(node_id)
        
        logging.info(f"Found {len(nodes_to_process)} thesis/synthesis nodes to process at depth {depth}")
        
        if not nodes_to_process:
            return False  # No nodes to process at this level
            
        # Process each node
        for node_id in nodes_to_process:
            node_data = self.graph[node_id]
            node_content = (node_data["summary"], node_data["content"])
            node_type = node_data["node_type"]
            
            logging.info(f"Generating antitheses for {node_type} node {node_id}")
            
            # Generate antitheses
            antitheses = self.generate_antitheses(node_content)
            
            # Add antithesis nodes
            for antithesis in antitheses:
                antithesis_id = self.add_node(
                    summary=antithesis[0],
                    content=antithesis[1],
                    node_type="antithesis",
                    parent_id=node_id
                )
                
                if self._check_time_exceeded():
                    # Mark this node as processed before returning
                    if node_type == "thesis":
                        self.theses_with_antitheses.add(node_id)
                    else:  # synthesis
                        self.processed_syntheses.add(node_id)
                    return True
            
            # Mark this node as processed
            if node_type == "thesis":
                self.theses_with_antitheses.add(node_id)
            else:  # synthesis
                self.processed_syntheses.add(node_id)
                
            # Save state after processing each node
            self._save_state()
        
        return True  # Successfully processed nodes
        """
        Process all thesis/synthesis nodes at the given depth.
        Generate antitheses for each node that hasn't been processed yet.
        Returns True if any nodes were processed.
        """
        # Find all thesis/synthesis nodes at this depth that haven't been processed
        nodes_to_process = []
        
        for node_id, node_data in self.graph.items():
            if (node_data.get("depth") == depth and 
                node_data.get("node_type") in ["thesis", "synthesis"] and
                not node_data.get("terminal", False)):
                
                # Skip if already processed
                if node_id in self.processed_theses or node_id in self.processed_syntheses:
                    continue
                    
                nodes_to_process.append(node_id)
        
        if not nodes_to_process:
            return False  # No nodes to process at this level
            
        # Process each node
        for node_id in nodes_to_process:
            node_data = self.graph[node_id]
            node_content = (node_data["summary"], node_data["content"])
            node_type = node_data["node_type"]
            
            # Generate antitheses
            antitheses = self.generate_antitheses(node_content)
            
            # Add antithesis nodes
            for antithesis in antitheses:
                antithesis_id = self.add_node(
                    summary=antithesis[0],
                    content=antithesis[1],
                    node_type="antithesis",
                    parent_id=node_id
                )
                
                if self._check_time_exceeded():
                    # Mark this node as processed before returning
                    if node_type == "thesis":
                        self.processed_theses.add(node_id)
                    else:  # synthesis
                        self.processed_syntheses.add(node_id)
                    return True
            
            # Mark this node as processed
            if node_type == "thesis":
                self.processed_theses.add(node_id)
            else:  # synthesis
                self.processed_syntheses.add(node_id)
                
            # Save state after processing each node
            self._save_state()
        
        return True  # Successfully processed nodes
    
    def _process_antithesis_level(self, depth: int) -> bool:
        """
        Process all antithesis nodes at the given depth.
        Generate syntheses for each node that hasn't been processed yet.
        Returns True if any nodes were processed.
        """
        # Find all antithesis nodes at this depth that haven't been processed
        nodes_to_process = []
        
        for node_id, node_data in self.graph.items():
            if (node_data.get("depth") == depth and 
                node_data.get("node_type") == "antithesis" and
                not node_data.get("terminal", False)):
                
                # Skip if already processed
                if node_id in self.processed_antitheses:
                    continue
                    
                nodes_to_process.append(node_id)
        
        if not nodes_to_process:
            return False  # No nodes to process at this level
            
        # Process each node
        for node_id in nodes_to_process:
            node_data = self.graph[node_id]
            antithesis_content = (node_data["summary"], node_data["content"])
            
            # Get parent thesis/synthesis content
            parent_id = node_data["parent_id"]
            parent_data = self.graph[parent_id]
            parent_content = (parent_data["summary"], parent_data["content"])
            
            # Generate syntheses
            syntheses = self.generate_syntheses(parent_content, antithesis_content)
            
            # Add synthesis nodes
            for synthesis in syntheses:
                synthesis_id = self.add_node(
                    summary=synthesis[0],
                    content=synthesis[1],
                    node_type="synthesis",
                    parent_id=node_id
                )
                
                if self._check_time_exceeded():
                    # Mark this node as processed before returning
                    self.processed_antitheses.add(node_id)
                    return True
            
            # Mark this node as processed
            self.processed_antitheses.add(node_id)
            
            # Save state after processing each node
            self._save_state()
        
        return True  # Successfully processed nodes
    
    def _ensure_central_question_has_uuid(self):
        """
        Ensure the central question has a UUID. 
        This handles backwards compatibility with old saves.
        """
        # Check if central_question is not a UUID (it's the text of the question instead)
        if self.central_question in self.graph and not self._is_uuid(self.central_question):
            # This is an old format save, need to convert
            old_key = self.central_question
            new_key = str(uuid.uuid4())
            
            # Store the actual text
            self.central_question_text = old_key
            
            # Move the node data to the new UUID key
            self.graph[new_key] = self.graph[old_key]
            self.graph[new_key]["is_central_question"] = True
            
            # Update any edges that reference the old key
            for i, (src, dst) in enumerate(self.edges):
                if src == old_key:
                    self.edges[i] = (new_key, dst)
                    
            # Update parent references in child nodes
            for node_id, node_data in self.graph.items():
                if node_data.get("parent_id") == old_key:
                    node_data["parent_id"] = new_key
                    
            # Remove the old key and update central_question
            del self.graph[old_key]
            self.central_question = new_key
            
            print(f"Converted central question from text key to UUID: {new_key}")
            
        # If we don't have a central_question_text yet, extract it
        if not hasattr(self, 'central_question_text') or not self.central_question_text:
            self.central_question_text = self.graph[self.central_question].get("content", "Unknown Question")
            
    def _is_uuid(self, s):
        """Check if a string is a valid UUID"""
        try:
            uuid.UUID(str(s))
            return True
        except ValueError:
            return False
        
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
        
    def _load_prompts(self) -> Dict[str, str]:
        """Load all prompt templates from files"""
        prompts = {}
        try:
            for prompt_type, filename in self.PROMPT_FILES.items():
                file_path = os.path.join(self.prompt_dir, filename)
                with open(file_path, 'r') as file:
                    prompts[prompt_type] = file.read().strip()
            return prompts
        except Exception as e:
            raise Exception(f"Error loading prompts: {e}")

    def generate_completion(self, prompt: str, system_role: str) -> str:
        """Generate a completion with error handling"""
        #print(prompt)
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ]
                
            )
            #print(response.choices[0].message.content.strip())
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error in API call: {e}")

    def parse_llm_output(self, llm_response: str) -> List[Tuple[str, str]]:
        """
        Parse LLM response into list of (summary, description) tuples.
        Automatically appends [END] tag if missing.
        Raises ValueError if other formatting is incorrect.
        """
        
        print("LLM Response: ", llm_response)
        
        # Split into individual items
        items = llm_response.split('[START]')
        items = [item.strip() for item in items if item.strip()]
        
        parsed_items = []
        for item in items:
            # Append [END] tag if missing
            if not item.endswith('[END]'):
                item = item + '[END]'
                
            # Remove [END] tag
            item = item[:-5].strip()
            
            # Split into summary and description
            parts = item.split('[BREAK]')
            if len(parts) != 2:
                raise ValueError(f"Item does not have exactly 2 parts: {item}")
            
            summary, description = parts
            parsed_items.append((summary.strip(), description.strip()))
        
        return parsed_items

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

    def search_similar_nodes(self, query: str, k: int = 5) -> List[str]:
        """Search for k most similar nodes to query"""
        query_embedding = self._calculate_embedding(query)
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Convert FAISS indices back to node IDs
        node_ids = []
        for idx in indices[0]:
            for node_id, faiss_idx in self.node_id_to_index.items():
                if faiss_idx == idx:
                    node_ids.append(node_id)
        return node_ids

    def generate_theses(self) -> list[tuple[str, str]]:
        """Generate N thesis [summary, description] response pairs to the central question"""
        system_role = "You are an experienced analytic philosophy professor generating candidate philosophical views in response to a central quetion of interest."
        prompt = self.prompts["thesis"].format(
            num_responses=self.num_responses,
            central_question=self.central_question_text
        )

        try:
            response = self.generate_completion(prompt, system_role)
            theses = self.parse_llm_output(response)
            return theses
        except Exception as e:
            raise Exception(f"Error generating theses: {e}")

    def generate_antitheses(self, thesis: tuple[str, str]) -> list[tuple[str, str]]:
        """Generate N antitheses for a given thesis"""
        system_role = "You are an experienced analytic philosophy professor generating objections to philosophical views."
        prompt = self.prompts["antithesis"].format(
            num_responses=self.num_responses,
            thesis=thesis
        )

        try:
            response = self.generate_completion(prompt, system_role)
            antitheses = self.parse_llm_output(response)
            return antitheses
        except Exception as e:
            raise Exception(f"Error generating antitheses: {e}")

    def generate_syntheses(self, thesis: tuple[str, str], antithesis: tuple[str,str]) -> list[tuple[str, str]]:
        """Generate N syntheses from a thesis-antithesis pair"""
        system_role = "You are an experienced analytic philosophy professor synthesising phiosophical views and their objections into new views that do not suffer from those objections"
        prompt = self.prompts["synthesis"].format(
            num_responses=self.num_responses,
            thesis=thesis,
            antithesis=antithesis
        )

        try:
            response = self.generate_completion(prompt, system_role)
            syntheses = self.parse_llm_output(response)
            return syntheses
        except Exception as e:
            raise Exception(f"Error generating syntheses: {e}")
        
    def generate_reasons(self, thesis: tuple[str, str]) -> list[tuple[str, str]]:
        """Generate supporting reasons for a given thesis"""
        system_role = "You are a philosophy professor in the analytic tradition trained at Princeton and Oxford."
        prompt = self.prompts["reasons"].format(
            num_reasons=self.num_reasons,
            thesis=thesis
        )

        try:
            response = self.generate_completion(prompt, system_role)
            reasons = self.parse_llm_output(response)
            return reasons
        except Exception as e:
            raise Exception(f"Error generating reasons: {e}")

    def generate_view_identity(self, view_a: str, view_b: str) -> float:
        """Compare two philosophical views and return a matching score from 0-100"""
        # Skip if termination checks are disabled
        if not self.check_termination:
            return 0.0  # Always return "not identical" when checks are disabled
            
        system_role = "You are an experienced analytic philosophy professor judging whether two philosophical views express the same or distict positions."
        prompt = self.prompts["view_identity"].format(
            view_a=view_a,
            view_b=view_b
        )

        try:
            result = self.generate_completion(prompt, system_role)
            # Extract numeric score from result
            try:
                score = float(result.strip())
                return score
            except ValueError:
                logging.error(f"Could not parse score from response: {result}")
                return 0.0
        except Exception as e:
            logging.error(f"Error generating view identity: {e}")
            return 0.0

    def generate_nonsense_check(self, view: str) -> float:
        """
        Check if a synthesis is meaningful or nonsense
        Returns: float score from 0-100, where higher scores indicate more nonsensical content
        """
        # Skip if termination checks are disabled
        if not self.check_termination:
            return 0.0  # Always return "not nonsense" when checks are disabled
            
        system_role = "You are an experienced analytic philosophy professor judging whether a statement is a legitimate philosophical view or whether it is nonsense."
        prompt = self.prompts["nonsense"].format(
            view=view
        )

        try:
            result = self.generate_completion(prompt, system_role)
            # Extract numeric score from result
            try:
                score = float(result.strip())
                return score
            except ValueError:
                logging.error(f"Could not parse nonsense score from response: {result}")
                return 0.0
        except Exception as e:
            logging.error(f"Error generating nonsense check: {e}")
            return 0.0

    def add_node(self, summary: str, content: str, node_type: str, parent_id: str) -> str:
        """Add a new node and save state"""
        node_id = str(uuid.uuid4())
        logging.info(f"Adding node {node_id} of type {node_type}")
        
        # Calculate depth
        if parent_id == self.central_question:
            depth = 1  # Regular depth for nodes directly under central question
        elif node_type == "reason":
            depth = -1  # Special depth for reason nodes
        else:
            depth = self.graph[parent_id]["depth"] + 1
        
        # Initialize node
        self.graph[node_id] = {
            "summary": summary,
            "content": content,
            "node_type": node_type,
            "parent_id": parent_id,
            "depth": depth,
            "terminal": False,
            "nonsense": False,
            "identical_to": None
        }
        
        # Add edge
        self.edges.append((parent_id, node_id))
        
        if self.check_termination:
        
            # Calculate and add embedding
            logging.info(f"Calculating embedding for node {node_id}")
            node_text = f"{summary} {content}"
            
            try:
                embedding = self._calculate_embedding(node_text)
                self.faiss_index.add(embedding)
                self.node_id_to_index[node_id] = self.faiss_index.ntotal - 1
                logging.info(f"Successfully added embedding for node {node_id}")
                
                # Save FAISS backup after successful embedding addition
                temp_index_path = os.path.join(self.save_dir, "temp_faiss_index.index")
                temp_mapping_path = os.path.join(self.save_dir, "temp_node_mapping.json")
                
                try:
                    # Save FAISS index
                    faiss.write_index(self.faiss_index, temp_index_path)
                    
                    # Save node mapping
                    with open(temp_mapping_path, 'w') as f:
                        json.dump(self.node_id_to_index, f)
                        
                    logging.info(f"Saved backup FAISS index after adding node {node_id}")
                except Exception as e:
                    logging.error(f"Failed to save backup FAISS index: {e}")
                    
            except Exception as e:
                logging.error(f"Failed to calculate embedding for node {node_id}: {e}")
            
                        # Run termination check for non-reason nodes
                if node_type != "reason":
                    self._check_termination(node_id, (summary, content))
                    
                # Save graph state after each node addition
                self._save_state()
        
        return node_id

    def get_ancestral_chain(self, node_id: str) -> List[str]:
        """Get the list of thesis and synthesis contents in the ancestral chain"""
        chain = []
        current_id = node_id
        
        while current_id is not None:
            current_node = self.graph[current_id]
            # Only include thesis and synthesis nodes in the chain
            if current_node["node_type"] in ["thesis", "synthesis"]:
                chain.append(current_node["content"])
            current_id = current_node["parent_id"]
                
        # Reverse to get chronological order (thesis first)
        return list(reversed(chain))

    def get_ancestors(self, node_id: str) -> List[str]:
        """Get all ancestor nodes of a given node"""
        ancestors = []
        current_id = self.graph[node_id]["parent_id"]
        while current_id is not None:
            ancestors.append(current_id)
            current_id = self.graph[current_id]["parent_id"]
        return ancestors

    def find_identical_view_id(self, node_id: str) -> Optional[str]:
        """
        Find if any existing view in the graph is semantically identical to this node.
        Uses embedding similarity to find top candidates before checking with LLM.
        """
        node_content = self.graph[node_id]["content"]
        
        # Get top 5 most similar nodes using FAISS
        similar_nodes = self.search_similar_nodes(node_content, k=5)
        logging.info(f"Found {len(similar_nodes)} similar nodes to check for identity")
        
        # For each similar node, check semantic similarity with LLM
        for similar_id in similar_nodes:
            # Skip self-comparison
            if similar_id == node_id:
                continue
                
            # Get the similar node's content
            similar_content = self.graph[similar_id]["content"]
            
            try:
                match_score = self.generate_view_identity(node_content, similar_content)
                if match_score >= self.view_identity_threshold:
                    logging.info(f"High match score ({match_score}) detected between node {node_id} and node {similar_id}")
                    return similar_id
            except Exception as e:
                logging.error(f"Error in semantic comparison: {e}")
                continue
        
        return None
            
    def is_terminal(self, node_id: str) -> bool:
        """Check if a node is terminal (due to nonsense or identity)"""
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} not found in graph")
        return self.graph[node_id]["terminal"]

    def get_identical_node(self, node_id: str) -> Optional[str]:
        """Get the ID of the node this one is identical to, if any"""
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} not found in graph")
        return self.graph[node_id]["identical_to"]

    def get_node_content(self, node_id: str) -> Dict:
        """Retrieve the content and metadata for a node"""
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} not found in graph")
        return self.graph[node_id]

    def get_children(self, node_id: str) -> List[str]:
        """Get the children of a node by looking through edges"""
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} not found in graph")
        return [child_id for parent_id, child_id in self.edges if parent_id == node_id]
        
    def get_children_by_type(self, node_id: str, node_type: str) -> List[str]:
        """Get children of a node with a specific type"""
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} not found in graph")
        
        children = self.get_children(node_id)
        return [child_id for child_id in children if self.graph[child_id]["node_type"] == node_type]

    def _check_termination(self, node_id: str, node_content: tuple) -> bool:
        """
        Check termination conditions for a node.
        If check_termination flag is False, always returns False.
        """
        # Skip termination check if disabled
        if not self.check_termination:
            return False
            
        try:
            # First check for nonsense
            print(f"Checking for nonsense: {node_id}")
            nonsense_score = self.generate_nonsense_check(node_content[1])
            print(f"Nonsense score: {nonsense_score} for node {node_id}")
            
            if nonsense_score > self.nonsense_threshold:
                self.graph[node_id]["terminal"] = True
                self.graph[node_id]["nonsense"] = True
                logging.info(f"Nonsense detected! Score: {nonsense_score} for node {node_id}")
                return True
                
            # Then check for identity with any existing view
            print(f"Checking for identical view: {node_id}")
            identical_view_id = self.find_identical_view_id(node_id)
            print(f"Identity check result: {identical_view_id} for node {node_id}")
            
            if identical_view_id:
                self.graph[node_id]["terminal"] = True
                self.graph[node_id]["identical_to"] = identical_view_id
                logging.info(f"Node {node_id} is identical to existing view {identical_view_id}")
                return True
                
        except Exception as e:
            print(f"Error in termination check: {e}")
            logging.error(f"Termination check failed for node {node_id}: {e}")
            # might want to decide what to return here - currently it will fall through to False
            
        # Not terminal
        self.graph[node_id]["terminal"] = False
        self.graph[node_id]["nonsense"] = False
        self.graph[node_id]["identical_to"] = None
        return False