from openai import OpenAI
from typing import Dict, List, Tuple, Optional
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
    def __init__(self, api_key: str, central_question: str, nonsense_threshold = 95, view_identity_threshold = 95, prompt_dir: str = "./prompts", 
                num_responses: int = 3, num_reasons: int = 10, max_depth: Optional[int] = None, 
                max_time_seconds: Optional[float] = None, save_dir: str = "./temp"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.client = OpenAI(api_key=api_key)
        self.default_model="gpt-4o" # Default model for completions
        self.max_depth = max_depth
        self.max_time_seconds = max_time_seconds
        self.start_time = datetime.now() if max_time_seconds is not None else None
        self.nonsense_threshold = nonsense_threshold
        self.view_identity_threshold = view_identity_threshold
        
        self.embedding_model = None  # For generating embeddings
        self.faiss_index = None     # For similarity search
        self.node_id_to_index = {}  # Map node IDs to their position in FAISS index
        
        # Add to __init__
        dimension = 1536  # for text-embedding-3-small
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.node_id_to_index = {}
        
        # Initialize client first as it's needed for either path
        self.client = OpenAI(api_key=api_key)
        
        # Try to load from most recent save
        try:
            latest_save = self._get_latest_save()
            if latest_save:
                self._load_from_save(latest_save)
                self.prompt_dir = prompt_dir
                self.prompts = self._load_prompts()
                return
        except Exception as e:
            print(f"Warning: Failed to load from save: {e}")
            
        
        # After loading graph state
        if hasattr(self, 'graph') and self.graph:
            self.load_faiss_index()
        
        # Initialize new graph if no save exists or loading failed
        self.graph = {}
        self.edges = []
        self.central_question = central_question
        self.graph[self.central_question] = {
            "summary": central_question,
            "content": central_question,
            "node_type": "question",
            "parent_id": None,
            "depth": 0,
            "terminal": False,
            "nonsense": False,
            "identical_to": None
        }
        self.num_responses = num_responses
        self.num_reasons = num_reasons
        self.max_depth = max_depth
        self.prompt_dir = prompt_dir
        self.prompts = self._load_prompts()
        
        # Save initial state
        try:
            self._save_state()
        except Exception as e:
            print(f"Warning: Failed to save initial state: {e}")
            
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
        
        
    #Save and Load functions (for Graph and FAISS)   
    def _save_state(self) -> None:
        """Save current graph state to file with error handling"""
        state = {
            "graph": self.graph,
            "edges": self.edges,
            "central_question": self.central_question,
            "num_responses": self.num_responses,
            "max_depth": self.max_depth
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
            required_keys = ["graph", "edges", "central_question", "num_responses", "max_depth"]
            if not all(key in state for key in required_keys):
                raise KeyError("Save file missing required keys")
                
            # Load state
            self.graph = state["graph"]
            self.edges = state["edges"]
            self.central_question = state["central_question"]
            self.num_responses = state["num_responses"]
            self.max_depth = state["max_depth"]
            
        except (json.JSONDecodeError, KeyError) as e:
            # If loading fails, remove corrupted save and start fresh
            os.remove(save_file)
            raise Exception(f"Failed to load save file: {e}")
        
        
    def save_faiss_index(self, save_path: str = None) -> None:
        """Save FAISS index and node mapping to disk
        
        Args:
            save_path: Directory to save index. If None, uses self.save_dir
        """
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
                
            # Update save timestamp in main graph state
            self.graph["embedding_save_timestamp"] = timestamp
            
            # Save updated graph state
            self._save_state()
            
        except Exception as e:
            raise Exception(f"Error saving FAISS index and mapping: {e}")

    def load_faiss_index(self) -> None:
        """Load most recent FAISS index and mapping if they exist"""
        try:
            # Get timestamp from graph state
            timestamp = self.graph.get("embedding_save_timestamp")
            if not timestamp:
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
        """Calculate embedding for node text using OpenAI's text-embedding-3-small model
        
        Args:
            text: The text to embed (summary and content concatenated)
        
        Returns:
            np.ndarray: Embedding vector of shape (1, 1536)  # 1536 is embedding dimension for text-embedding-3-small
            
        Raises:
            Exception: If embedding calculation fails
        """
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
            raise Exception(f"Error calculating embedding: {e}")

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
            central_question=self.central_question
        )

        try:
            response = self.generate_completion(prompt, system_role)
            theses = self.parse_llm_output(response)
            #print("Theses: ", theses)
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

    def _add_reasons_for_thesis(self, thesis_id: str, thesis_content: tuple[str, str]) -> None:
        """Add supporting reason nodes for a given thesis"""
        try:
            # Generate reasons for this thesis
            reasons = self.generate_reasons(thesis_content)
            
            # Add each reason as a node
            for reason in reasons:
                reason_id = self.add_node(
                    summary=reason[0],
                    content=reason[1],
                    node_type="reason",
                    parent_id=thesis_id
                )
                
                # Save state after each reason
                self._save_state()    
                
        except Exception as e:
            logging.error(f"Error adding reasons for thesis {thesis_id}: {e}")

    def generate_view_identity(self, view_a: str, view_b: str) -> float:
        """Compare two philosophical views and return a matching score from 0-100"""
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
            raise Exception(f"Error generating view identity: {e}")

    def generate_nonsense_check(self, view: str) -> float:
        """
        Check if a synthesis is meaningful or nonsense
        Returns: float score from 0-100, where higher scores indicate more nonsensical content
        """
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
            raise Exception(f"Error generating nonsense check: {e}")

    def add_node(self, summary: str, content: str, node_type: str, parent_id: str) -> str:
        """Add a new node and save state"""
        node_id = str(uuid.uuid4())
        logging.info(f"Adding node {node_id} of type {node_type}")
        
        # Calculate depth
        if parent_id == self.central_question:
            depth = 1
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
        current_id = self.graph[node_id]["parent_id"]  # Changed from "parent" to "parent_id"
        while current_id is not None:
            ancestors.append(current_id)
            current_id = self.graph[current_id]["parent_id"]  # Changed here too
        return ancestors

    def find_identical_view_id(self, node_id: str) -> Optional[str]:
        """
        Find if any existing view in the graph is semantically identical to this node.
        Uses embedding similarity to find top candidates before checking with LLM.
        
        Args:
            node_id: ID of the node to check for identity
            
        Returns:
            Optional[str]: ID of the matching node if found, None otherwise
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


    def _check_termination(self, synthesis_id: str, synthesis: tuple) -> bool:
        """
        Check termination conditions for a synthesis node.
        """
        try:
            # First check for nonsense
            print("Checking for nonsense")
            nonsense_score = self.generate_nonsense_check(synthesis[1])
            print(f"Nonsense score: {nonsense_score}")
            
            if nonsense_score > self.nonsense_threshold:
                self.graph[synthesis_id]["terminal"] = True
                self.graph[synthesis_id]["nonsense"] = True
                logging.info(f"Nonsense detected! Score: {nonsense_score} for node {synthesis_id}")
                return True
                
            # Then check for identity with any existing view
            print("Checking for identical view")
            identical_view_id = self.find_identical_view_id(synthesis_id)
            print(f"Identity check result: {identical_view_id}")
            
            if identical_view_id:
                self.graph[synthesis_id]["terminal"] = True
                self.graph[synthesis_id]["identical_to"] = identical_view_id
                logging.info(f"Node {synthesis_id} is identical to existing view {identical_view_id}")
                return True
                
        except Exception as e:
            print(f"Error in termination check: {e}")
            logging.error(f"Termination check failed: {e}")
            # might want to decide what to return here - currently it will fall through to False
            
        # Not terminal
        self.graph[synthesis_id]["terminal"] = False
        self.graph[synthesis_id]["nonsense"] = False
        self.graph[synthesis_id]["identical_to"] = None
        return False
    
    def initialize_graph(self) -> None:
        """Initialize the complete dialectical graph with breadth-first progression"""
        if self.central_question not in self.graph:
            self.graph[self.central_question] = {
                "summary": "Central Question",
                "content": self.central_question,
                "node_type": "question",
                "parent_id": None,
                "depth": 0,
                "terminal": False,
                "nonsense": False,
                "identical_to": None
            }
            
        # Generate initial theses (depth 1)
        theses = self.generate_theses()
        thesis_nodes = []
        for thesis in theses:
            thesis_id = self.add_node(
                summary=thesis[0], 
                content=thesis[1], 
                node_type="thesis", 
                parent_id=self.central_question
            )
            
            # Add supporting reasons for this thesis
            self._add_reasons_for_thesis(thesis_id, thesis)
            
            thesis_nodes.append((thesis_id, thesis))
            if self._check_time_exceeded():
                logging.warning("Time limit exceeded after thesis generation")
                return

        # Process level by level
        current_depth_nodes = thesis_nodes
        while current_depth_nodes:
            next_depth_nodes = []
            
            # Process all nodes at current depth
            for node_id, node_content in current_depth_nodes:
                if self._check_time_exceeded():
                    logging.warning("Time limit exceeded during node processing")
                    return
                    
                if self._check_depth_exceeded(node_id):
                    logging.warning(f"Maximum depth exceeded at node {node_id}")
                    continue
                    
                current_type = self.graph[node_id]["node_type"]
                
                if current_type == "thesis":
                    # Generate all antitheses for current thesis
                    antitheses = self.generate_antitheses(node_content)
                    for antithesis in antitheses:
                        antithesis_id = self.add_node(
                            summary=antithesis[0],
                            content=antithesis[1],
                            node_type="antithesis",
                            parent_id=node_id
                        )
                        
                        # Check antithesis for termination conditions
                        is_terminal = self._check_termination(antithesis_id, antithesis)
                        
                        # Only continue with non-terminal antitheses
                        if not is_terminal:
                            next_depth_nodes.append((antithesis_id, antithesis))
                            
                elif current_type == "antithesis":
                    # Get parent thesis content
                    parent_id = self.graph[node_id]["parent_id"]
                    parent_content = (
                        self.graph[parent_id]["summary"], 
                        self.graph[parent_id]["content"]
                    )
                    
                    # Generate syntheses
                    syntheses = self.generate_syntheses(parent_content, node_content)
                    for synthesis in syntheses:
                        synthesis_id = self.add_node(
                            summary=synthesis[0],
                            content=synthesis[1],
                            node_type="synthesis",
                            parent_id=node_id
                        )
                        
                        # Check synthesis for termination conditions
                        is_terminal = self._check_termination(synthesis_id, synthesis)
                        
                        # Only continue with non-terminal syntheses
                        if not is_terminal:
                            next_depth_nodes.append((synthesis_id, synthesis))
                        
                elif current_type == "synthesis":
                    # Only generate new antitheses if synthesis isn't terminal
                    if not self.is_terminal(node_id):
                        antitheses = self.generate_antitheses(node_content)
                        for antithesis in antitheses:
                            antithesis_id = self.add_node(
                                summary=antithesis[0],
                                content=antithesis[1],
                                node_type="antithesis",
                                parent_id=node_id
                            )
                            
                            # Check new antithesis for termination conditions
                            is_terminal = self._check_termination(antithesis_id, antithesis)
                            
                            # Only continue with non-terminal antitheses
                            if not is_terminal:
                                next_depth_nodes.append((antithesis_id, antithesis))
                
                # Save state after processing each node
                self._save_state()
            
            # Move to next depth level
            current_depth_nodes = next_depth_nodes