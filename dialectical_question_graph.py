import random
from openai import OpenAI
from typing import Dict, List, Tuple, Optional
import os
import re

class DialecticalGraph:
    PROMPT_FILES = {
        "thesis": "thesis_prompt.txt",
        "antithesis": "antithesis_prompt.txt",
        "synthesis": "synthesis_prompt.txt",
        "view_identity": "view_identity_prompt.txt",
        "nonsense": "nonsense_prompt.txt"
    }
    def __init__(self, api_key: str, central_question: str, prompt_dir: str = "./prompts", num_responses: int = 3, max_depth: int = 3):
        self.client = OpenAI(api_key=api_key)
        self.graph: Dict[str, Dict[str, any]] = {}
        self.central_question = central_question
        self.graph[self.central_question] = {
            "type": "question",
            "content": central_question,
            "children": [],
            "parent": None,
            "depth": 0
        }
        self.num_responses = num_responses
        self.max_depth = max_depth
        self.prompt_dir = prompt_dir
        self.prompts = self._load_prompts()
        self.initialize_graph()
        
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

    def generate_completion(self, prompt: str, system_role: str, max_tokens: int = 500) -> str:
        """Generate a completion with error handling"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ]
                #max_tokens=max_tokens
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
            response = self.generate_completion(prompt, system_role, max_tokens=300)
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
            return self.generate_completion(prompt, system_role, max_tokens=50)
        except Exception as e:
            raise Exception(f"Error generating view identity: {e}")

    def generate_nonsense_check(self, synthesis: str) -> str:
        """Check if a synthesis is meaningful or nonsense"""
        system_role = "You are a philosophical critic evaluating statements for meaningfulness."
        prompt = self.prompts["nonsense"].format(
            synthesis=synthesis
        )

        try:
            return self.generate_completion(prompt, system_role, max_tokens=50)
        except Exception as e:
            raise Exception(f"Error generating nonsense check: {e}")

    def add_node(self, summary: str, content: str, node_type: str, parent_id: Optional[str] = None) -> str:
        """Add a new node to the graph and return its ID"""
        node_id = f"{node_type}_{len(self.graph)}"
        
        # Calculate depth based on parent
        depth = 0
        if parent_id:
            depth = self.graph[parent_id]["depth"] + 1
            
        self.graph[node_id] = {
            "type": node_type,
            "summary": summary,
            "content": content,
            "children": [],
            "parent": parent_id,
            "identical_to": None,  # Track identity relationships
            "terminal": False,     # Track if this is a terminal node
            "depth": depth        # Track depth in graph
        }
        if parent_id:
            self.graph[parent_id]["children"].append(node_id)
        return node_id

    def get_ancestral_chain(self, node_id: str) -> List[str]:
        """Get the list of thesis and synthesis contents in the ancestral chain"""
        chain = []
        current_id = node_id
        
        while current_id is not None:
            current_node = self.graph[current_id]
            # Only include thesis and synthesis nodes in the chain
            if current_node["type"] in ["thesis", "synthesis"]:
                chain.append(current_node["content"])
            current_id = current_node["parent"]
            
        # Reverse to get chronological order (thesis first)
        return list(reversed(chain))

    def get_ancestors(self, node_id: str) -> List[str]:
        """Get all ancestor nodes of a given node"""
        ancestors = []
        current_id = self.graph[node_id]["parent"]
        while current_id is not None:
            ancestors.append(current_id)
            current_id = self.graph[current_id]["parent"]
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
        """Get the children of a node"""
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} not found in graph")
        return self.graph[node_id]["children"]
    
    
    
    
    
    def initialize_graph(self) -> None:
        """Initialize the complete dialectical graph"""
        try:
            # Generate theses (depth 1)
            theses = self.generate_theses()
            for thesis in theses:
                thesis_id = self.add_node(summary=thesis[0], content=thesis[1], node_type="thesis", parent_id=self.central_question)
                
                if self.graph[thesis_id]["depth"] >= self.max_depth:
                    continue
                    
                # Generate antitheses for each thesis (depth 2)
                antitheses = self.generate_antitheses(thesis)
                for antithesis in antitheses:
                    antithesis_id = self.add_node(summary = antithesis[0], content=antithesis[1], node_type= "antithesis",parent_id=thesis_id)
                    
                    if self.graph[antithesis_id]["depth"] >= self.max_depth:
                        continue
                    
                    # Generate syntheses for each thesis-antithesis pair (depth 3)
                    syntheses = self.generate_syntheses(thesis, antithesis)
                    for synthesis in syntheses:
                        synthesis_id = self.add_node(summary = synthesis[0], content = synthesis[1], node_type= "synthesis", parent_id=antithesis_id)
                        
                        if self.graph[synthesis_id]["depth"] >= self.max_depth:
                            continue
                        
                        # Check for nonsense first
                        nonsense_result = self.generate_nonsense_check(synthesis)
                        if nonsense_result.upper() == "NONSENSE":
                            self.graph[synthesis_id]["terminal"] = True
                            self.graph[synthesis_id]["nonsense"] = True
                            continue
                        
                        # Get ancestors and check for identity
                        ancestors = self.get_ancestors(synthesis_id)
                        identical_ancestor_id = None
                        if ancestors:
                            identical_ancestor_id = self.find_identical_ancestor_id(synthesis_id)
                            
                        if identical_ancestor_id:
                            self.graph[synthesis_id]["terminal"] = True
                            self.graph[synthesis_id]["identical_to"] = identical_ancestor_id
                            continue
                        
                        # If not terminal, mark as valid
                        self.graph[synthesis_id]["nonsense"] = False
                        self.graph[synthesis_id]["identical_to"] = None

        except Exception as e:
            raise Exception(f"Error initializing graph: {e}")