from openai import OpenAI
from typing import Dict, List, Tuple, Optional
import os
import uuid
from datetime import datetime
import json

class DialecticalGraph:
    PROMPT_FILES = {
        "thesis": "thesis_prompt.txt",
        "antithesis": "antithesis_prompt.txt",
        "synthesis": "synthesis_prompt.txt",
        "view_identity": "view_identity_prompt.txt",
        "nonsense": "nonsense_prompt.txt"
    }
    def __init__(self, api_key: str, central_question: str, prompt_dir: str = "./prompts", 
                num_responses: int = 3, max_depth: int = 3, save_dir: str = "./temp"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
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
        
        # Initialize new graph if no save exists or loading failed
        self.graph = {}
        self.edges = []
        self.central_question = central_question
        self.graph[self.central_question] = {
            "summary": "Central Question",
            "content": central_question,
            "node_type": "question",
            "parent_id": None,
            "depth": 0,
            "terminal": False,
            "nonsense": False,
            "identical_to": None
        }
        self.num_responses = num_responses
        self.max_depth = max_depth
        self.prompt_dir = prompt_dir
        self.prompts = self._load_prompts()
        
        # Save initial state
        try:
            self._save_state()
        except Exception as e:
            print(f"Warning: Failed to save initial state: {e}")
            
        self.initialize_graph()
    
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
                model="gpt-4o-mini",
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
        Raises ValueError if formatting is incorrect.
        """
        
        print("LLM Response: ", llm_response)
        
        # Split into individual items
        items = llm_response.split('[START]')
        items = [item.strip() for item in items if item.strip()]
        
        parsed_items = []
        for item in items:
            # Verify [END] tag and remove it
            if not item.endswith('[END]'):
                raise ValueError(f"Item missing [END] tag: {item}")
            item = item[:-5].strip()
            
            # Split into summary and description
            parts = item.split('[BREAK]')
            if len(parts) != 2:
                raise ValueError(f"Item does not have exactly 2 parts: {item}")
            
            summary, description = parts
  
            parsed_items.append((summary.strip(), description.strip()))
        
        return parsed_items



    def generate_theses(self) -> list[tuple[str, str]]:
        """Generate N thesis [summary, description] response pairs to the central question"""
        system_role = "You are a philosophical inquirer generating thesis statements."
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
        system_role = "You are a critical philosopher generating objections to thesis statements."
        prompt = self.prompts["antithesis"].format(
            num_responses=self.num_responses,
            thesis=thesis
        )

        try:
            response = self.generate_completion(prompt, system_role)
            antitheses = self.parse_llm_output(response)
            #print("Antitheses", antitheses)
            return antitheses
        except Exception as e:
            raise Exception(f"Error generating antitheses: {e}")

    def generate_syntheses(self, thesis: tuple[str, str], antithesis: tuple[str,str]) -> list[tuple[str, str]]:
        """Generate N syntheses from a thesis-antithesis pair"""
        system_role = "You are a dialectical philosopher generating synthetic positions."
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

    def generate_view_identity(self, synthesis_id: str) -> str:
        """Generate a view identity analysis for a synthesis"""
        synthesis = self.graph[synthesis_id]["content"]
        ancestral_chain = self.get_ancestral_chain(synthesis_id)
        
        system_role = "You are an analyst identifying the philosophical viewpoint of statements."
        prompt = self.prompts["view_identity"].format(
            synthesis=synthesis,
            ancestral_chain=ancestral_chain
        )

        try:
            return self.generate_completion(prompt, system_role)
        except Exception as e:
            raise Exception(f"Error generating view identity: {e}")

    def generate_nonsense_check(self, synthesis: str) -> str:
        """Check if a synthesis is meaningful or nonsense"""
        system_role = "You are a philosophical critic evaluating statements for meaningfulness."
        prompt = self.prompts["nonsense"].format(
            synthesis=synthesis
        )

        try:
            return self.generate_completion(prompt, system_role)
        except Exception as e:
            raise Exception(f"Error generating nonsense check: {e}")

    def add_node(self, summary: str, content: str, node_type: str, parent_id: str) -> str:
        """Add a new node and save state"""
        node_id = str(uuid.uuid4())
        
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
        
        # Save state after each node addition
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

    def find_identical_ancestor_id(self, node_id: str) -> Optional[str]:
        """Find if any ancestor has identical content to this node"""
        node_content = self.graph[node_id]["content"]
        ancestors = self.get_ancestors(node_id)
        
        for ancestor_id in ancestors:
            if self.graph[ancestor_id]["content"] == node_content:
                return ancestor_id
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

    def _process_dialectical_progression(self, node_id: str, node_content: tuple) -> None:
        """
        Recursively process dialectical progression for a given node
        
        Args:
            node_id: ID of the current node
            node_content: Tuple of (summary, content) for the current node
        """
        # Check depth limit
        if self.graph[node_id]["depth"] >= self.max_depth:
            return
            
        current_type = self.graph[node_id]["node_type"]  # This should now work
        
        if current_type == "thesis":
            # Generate antitheses
            antitheses = self.generate_antitheses(node_content)
            for antithesis in antitheses:
                antithesis_id = self.add_node(
                    summary=antithesis[0],
                    content=antithesis[1],
                    node_type="antithesis",
                    parent_id=node_id
                )
                self._process_dialectical_progression(antithesis_id, antithesis)
                
        elif current_type == "antithesis":
            # Get parent thesis content
            parent_id = self.graph[node_id]["parent_id"]
            parent_content = (self.graph[parent_id]["summary"], 
                            self.graph[parent_id]["content"])
            
            # Generate syntheses
            syntheses = self.generate_syntheses(parent_content, node_content)
            for synthesis in syntheses:
                synthesis_id = self.add_node(
                    summary=synthesis[0],
                    content=synthesis[1],
                    node_type="synthesis",
                    parent_id=node_id
                )
                
                # Check termination conditions
                if self._check_termination(synthesis_id, synthesis):
                    continue
                    
                # If not terminal, process synthesis as new thesis
                self._process_dialectical_progression(synthesis_id, synthesis)
                
        elif current_type == "synthesis":
            # Treat non-terminal synthesis as new thesis
            antitheses = self.generate_antitheses(node_content)
            for antithesis in antitheses:
                antithesis_id = self.add_node(
                    summary=antithesis[0],
                    content=antithesis[1],
                    node_type="antithesis",
                    parent_id=node_id
                )
                self._process_dialectical_progression(antithesis_id, antithesis)

    def _check_termination(self, synthesis_id: str, synthesis: tuple) -> bool:
        """
        Check termination conditions for a synthesis node
        Returns True if the node is terminal, False otherwise
        """
        # Check for nonsense
        nonsense_result = self.generate_nonsense_check(synthesis)
        if nonsense_result.upper() == "NONSENSE":
            self.graph[synthesis_id]["terminal"] = True
            self.graph[synthesis_id]["nonsense"] = True
            return True
        
        # Check for identity with ancestors
        ancestors = self.get_ancestors(synthesis_id)
        if ancestors:
            identical_ancestor_id = self.find_identical_ancestor_id(synthesis_id)
            if identical_ancestor_id:
                self.graph[synthesis_id]["terminal"] = True
                self.graph[synthesis_id]["identical_to"] = identical_ancestor_id
                return True
        
        # Not terminal
        self.graph[synthesis_id]["terminal"] = False
        self.graph[synthesis_id]["nonsense"] = False
        self.graph[synthesis_id]["identical_to"] = None
        return False
    
    def initialize_graph(self) -> None:
        """Initialize the complete dialectical graph with recursive progression"""
        # Initialize central question node if not already done
        if self.central_question not in self.graph:
            self.graph[self.central_question] = {
                "summary": "Central Question",
                "content": self.question,
                "node_type": "question",
                "parent_id": None,
                "depth": 0,
                "terminal": False,
                "nonsense": False,
                "identical_to": None
            }
        
        # Generate initial theses (depth 1)
        theses = self.generate_theses()
        for thesis in theses:
            thesis_id = self.add_node(
                summary=thesis[0], 
                content=thesis[1], 
                node_type="thesis", 
                parent_id=self.central_question
            )
            # Process each thesis through dialectical progression
            self._process_dialectical_progression(thesis_id, thesis)