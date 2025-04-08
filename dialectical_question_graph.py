from openai import OpenAI
from typing import Dict, List, Tuple, Optional, Set
import os
import uuid
from datetime import datetime
import time
import json
import logging
import faiss
import numpy as np
import embedding 
from prompt_llm import load_prompts, generate_completion, parse_llm_output, generate_completions_batch, parse_batch_outputs
import pickle
import types  # For adding methods to instance


class DialecticalGraph:
    def __init__(self, 
                central_question: str, 
                nonsense_threshold, 
                view_identity_threshold,  
                num_responses: int, 
                num_reasons: int, 
                max_depth: Optional[int], 
                max_time_seconds: Optional[float] = None, 
                save_dir: str = "./temp", 
                check_termination: bool = False):
        
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
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
        '''
        if self.check_termination:
            self._init_faiss()
        else:
            self.faiss_index = None
            self.node_id_to_index = {}
            logging.info("FAISS index not initialized (check_termination=False)")
        '''
        
        # Track processed nodes by type
        self.processed_theses = set()
        self.processed_antitheses = set()
        self.processed_syntheses = set()
        self.theses_with_reasons = set()
        self.theses_with_antitheses = set()
        
        # Load Prompts
        self.prompts = load_prompts()
        
        # Initialize new graph if not loaded
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
        self.save_state()
        
        # Whether resuming or starting fresh, continue the graph initialization
        self.initialize_graph()
        
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
            theses = self.generate_theses()
            
            if theses is None:
                # Add a blank node if LLM response was None
                self.add_node(
                    summary="[Generation failed]",
                    content="The system was unable to generate a thesis for this question.",
                    node_type="thesis_error",
                    parent_id=self.central_question
                )
                return
            
            for thesis in theses:
                thesis_id = self.add_node(
                    summary=thesis[0],
                    content=thesis[1],
                    node_type="thesis",
                    parent_id=self.central_question
                )

            if self._check_time_exceeded():
                logging.info("Time limit reached after generating initial theses")
                return
        
        # Step 2: Generate reasons for each thesis (if not already done) using batch processing
        thesis_nodes = self.get_children_by_type(self.central_question, "thesis")
        
        # Collect theses that need reasons
        theses_needing_reasons = []
        thesis_ids_needing_reasons = []
        
        for thesis_id in thesis_nodes:
            if thesis_id not in self.theses_with_reasons:
                thesis_data = self.graph[thesis_id]
                thesis_content = (thesis_data["summary"], thesis_data["content"])
                theses_needing_reasons.append(thesis_content)
                thesis_ids_needing_reasons.append(thesis_id)
        
        if theses_needing_reasons:
            logging.info(f"Batch generating reasons for {len(theses_needing_reasons)} theses")
            
            # Generate reasons in batch
            reasons_sets = self.generate_reasons_batch(theses_needing_reasons)
            
            # Process each thesis with its generated reasons
            for i, thesis_id in enumerate(thesis_ids_needing_reasons):
                reasons = reasons_sets[i]
                
                if reasons is None:
                    # Add a blank node if LLM response was None
                    self.add_node(
                        summary="[Reason generation failed]",
                        content="The system was unable to generate supporting reasons for this thesis.",
                        node_type="reason_error",
                        parent_id=thesis_id
                    )
                    continue
                
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
                
                self.theses_with_reasons.add(thesis_id)
                self.save_state()
                
                if self._check_time_exceeded():
                    logging.info("Time limit reached during reason generation")
                    return
        
        # Steps 3-4: Cycle between antitheses and syntheses until termination
        current_depth = 1  # Start with theses at depth 1
        max_iterations = 10  # Safety limit to prevent infinite loops
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            logging.info(f"Starting iteration {iterations} at depth {current_depth}")
            
            # Process all nodes at the current depth
            if current_depth % 2 == 1:  # Odd depths are theses/syntheses
                # Generate antitheses for theses/syntheses at this depth
                has_progress = self._process_odd_level(current_depth)
                logging.info(f"Processed thesis/synthesis nodes at depth {current_depth}, progress: {has_progress}")
            else:  # Even depths are antitheses
                # Generate syntheses and direct replies for antitheses at this depth
                has_progress = self._process_even_level(current_depth)
                logging.info(f"Processed antithesis nodes at depth {current_depth}, progress: {has_progress}")
            
            # Check if we should continue
            if not has_progress:
                # If no nodes were processed at this depth, try the next depth
                current_depth += 1
                logging.info(f"No progress at depth {current_depth-1}, moving to depth {current_depth}")
                
                # If we've reached max_depth, stop
                if self.max_depth is not None and current_depth > self.max_depth-1:
                    logging.info(f"Maximum depth of {self.max_depth} reached")
                    break
            
            # Check time limit
            if self._check_time_exceeded():
                logging.info("Time limit reached during processing")
                return
                
            # Save state after each level is processed
            self.save_state()
        
        logging.info("Graph exploration completed") 
        
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
        
        # Add edge to parent node
        self.edges.append((parent_id, node_id))
        
        #if node is a direct reply, mark node as terminal
        if node_type == "direct_reply":
            self.graph[node_id]["terminal"] = True
        
        #if termination checks are enabled, check for nonsense and view identity
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
                self.save_state()
        
        return node_id
    
            
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

    def generate_theses(self) -> Optional[list[tuple[str, str]]]:
        """Generate N thesis [summary, description] response pairs to the central question
        Returns None if parsing fails"""
        system_role = ""
        prompt = self.prompts["thesis"].format(
            num_responses=self.num_responses,
            central_question=self.central_question_text
        )

        try:
            response = generate_completion(prompt, system_role)
            theses = parse_llm_output(response)
            return theses  # This will be None if parsing failed
        except Exception as e:
            print(f"Error generating theses: {e}")
            return None
            
    def generate_theses_batch(self, num_batches=3) -> List[Optional[list[tuple[str, str]]]]:
        """Generate multiple batches of thesis responses in parallel
        Returns a list of thesis sets, each containing multiple thesis pairs"""
        system_role = ""
        prompts = []
        
        # Create multiple prompts with the same content
        for _ in range(num_batches):
            prompts.append(self.prompts["thesis"].format(
                num_responses=self.num_responses,
                central_question=self.central_question_text
            ))
        
        try:
            # Generate all completions in parallel
            batch_responses = generate_completions_batch(prompts, system_role)
            # Parse all responses
            theses_sets = parse_batch_outputs(batch_responses)
            return theses_sets
        except Exception as e:
            print(f"Error generating batched theses: {e}")
            return [None] * num_batches

    def generate_antitheses(self, thesis: tuple[str, str]) -> Optional[list[tuple[str, str]]]:
        """Generate N antitheses for a given thesis
        Returns None if parsing fails"""
        system_role = ""
        prompt = self.prompts["antithesis"].format(
            num_responses=self.num_responses,
            thesis=thesis
        )

        try:
            response = generate_completion(prompt, system_role)
            antitheses = parse_llm_output(response)
            return antitheses  # This will be None if parsing failed
        except Exception as e:
            print(f"Error generating antitheses: {e}")
            return None
            
    def generate_antitheses_batch(self, theses: List[tuple[str, str]]) -> List[Optional[list[tuple[str, str]]]]:
        """Generate antitheses for multiple theses in parallel
        Returns a list of antithesis sets, each containing multiple antithesis pairs"""
        system_role = ""
        prompts = []
        
        # Create prompts for each thesis
        for thesis in theses:
            prompts.append(self.prompts["antithesis"].format(
                num_responses=self.num_responses,
                thesis=thesis
            ))
        
        try:
            # Generate all completions in parallel
            batch_responses = generate_completions_batch(prompts, system_role)
            # Parse all responses
            antitheses_sets = parse_batch_outputs(batch_responses)
            return antitheses_sets
        except Exception as e:
            print(f"Error generating batched antitheses: {e}")
            return [None] * len(theses)

    def generate_direct_replies(self, thesis: tuple[str, str], antithesis: tuple[str,str]) -> Optional[list[tuple[str, str]]]:
        """Generate N direct replies to a thesis-antithesis pair
        Returns None if parsing fails"""
        system_role = ""
        prompt = self.prompts["direct_reply"].format(
            #num_responses=self.num_responses,
            thesis=thesis,
            antithesis=antithesis
        )

        try:
            response = generate_completion(prompt, system_role)
            syntheses = parse_llm_output(response)
            return syntheses  # This will be None if parsing failed
        except Exception as e:
            print(f"Error generating direct replies: {e}")
            return None
    
    def generate_direct_replies_batch(self, thesis_antithesis_pairs: List[Tuple[tuple[str, str], tuple[str,str]]]) -> List[Optional[list[tuple[str, str]]]]:
        """Generate direct replies for multiple thesis-antithesis pairs in parallel
        Returns a list of direct reply sets"""
        system_role = ""
        prompts = []
        
        # Create prompts for each thesis-antithesis pair
        for thesis, antithesis in thesis_antithesis_pairs:
            prompts.append(self.prompts["direct_reply"].format(
                #num_responses=self.num_responses,
                thesis=thesis,
                antithesis=antithesis
            ))
        
        try:
            # Generate all completions in parallel
            batch_responses = generate_completions_batch(prompts, system_role)
            # Parse all responses
            replies_sets = parse_batch_outputs(batch_responses)
            return replies_sets
        except Exception as e:
            print(f"Error generating batched direct replies: {e}")
            return [None] * len(thesis_antithesis_pairs)    

    def generate_syntheses(self, thesis: tuple[str, str], antithesis: tuple[str,str]) -> Optional[list[tuple[str, str]]]:
        """Generate N syntheses from a thesis-antithesis pair
        Returns None if parsing fails"""
        system_role = ""
        prompt = self.prompts["synthesis"].format(
            num_responses=self.num_responses,
            thesis=thesis,
            antithesis=antithesis
        )

        try:
            response = generate_completion(prompt, system_role)
            syntheses = parse_llm_output(response)
            return syntheses  # This will be None if parsing failed
        except Exception as e:
            print(f"Error generating syntheses: {e}")
            return None
            
    def generate_syntheses_batch(self, thesis_antithesis_pairs: List[Tuple[tuple[str, str], tuple[str,str]]]) -> List[Optional[list[tuple[str, str]]]]:
        """Generate syntheses for multiple thesis-antithesis pairs in parallel
        Returns a list of synthesis sets"""
        system_role = ""
        prompts = []
        
        # Create prompts for each thesis-antithesis pair
        for thesis, antithesis in thesis_antithesis_pairs:
            prompts.append(self.prompts["synthesis"].format(
                num_responses=self.num_responses,
                thesis=thesis,
                antithesis=antithesis
            ))
        
        try:
            # Generate all completions in parallel
            batch_responses = generate_completions_batch(prompts, system_role)
            # Parse all responses
            syntheses_sets = parse_batch_outputs(batch_responses)
            return syntheses_sets
        except Exception as e:
            print(f"Error generating batched syntheses: {e}")
            return [None] * len(thesis_antithesis_pairs)
        
    def generate_reasons(self, thesis: tuple[str, str]) -> Optional[list[tuple[str, str]]]:
        """Generate supporting reasons for a given thesis
        Returns None if parsing fails"""
        system_role = ""
        prompt = self.prompts["reasons"].format(
            num_reasons=self.num_reasons,
            thesis=thesis
        )

        try:
            response = generate_completion(prompt, system_role)
            reasons = parse_llm_output(response)
            return reasons  # This will be None if parsing failed
        except Exception as e:
            print(f"Error generating reasons: {e}")
            return None
            
    def generate_reasons_batch(self, theses: List[tuple[str, str]]) -> List[Optional[list[tuple[str, str]]]]:
        """Generate reasons for multiple theses in parallel
        Returns a list of reason sets, each containing multiple reason pairs"""
        system_role = ""
        prompts = []
        
        # Create prompts for each thesis
        for thesis in theses:
            prompts.append(self.prompts["reasons"].format(
                num_reasons=self.num_reasons,
                thesis=thesis
            ))
        
        try:
            # Generate all completions in parallel
            batch_responses = generate_completions_batch(prompts, system_role)
            # Parse all responses
            reasons_sets = parse_batch_outputs(batch_responses)
            return reasons_sets
        except Exception as e:
            print(f"Error generating batched reasons: {e}")
            return [None] * len(theses)
        
    
    def _process_odd_level(self, depth: int) -> bool:
        """
        Process all thesis/synthesis nodes at the given depth.
        Generate antitheses for each node that hasn't been processed yet.
        Skips direct reply nodes.
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
            
        # Prepare batch processing
        node_contents = []
        node_types = []
        
        for node_id in nodes_to_process:
            node_data = self.graph[node_id]
            node_contents.append((node_data["summary"], node_data["content"]))
            node_types.append(node_data["node_type"])
        
        logging.info(f"Batch generating antitheses for {len(node_contents)} nodes")
        
        # Generate antitheses in batch
        antitheses_sets = self.generate_antitheses_batch(node_contents)
        
        # Process each node with its generated antitheses
        for i, node_id in enumerate(nodes_to_process):
            node_type = node_types[i]
            antitheses = antitheses_sets[i]
            
            if antitheses is None:
                # Add a blank node if LLM response was None
                self.add_node(
                    summary="[Antithesis generation failed]",
                    content=f"The system was unable to generate antitheses for this {node_type}.",
                    node_type="antithesis_error",
                    parent_id=node_id
                )
                
                # Mark this node as processed
                if node_type == "thesis":
                    self.theses_with_antitheses.add(node_id)
                else:  # synthesis
                    self.processed_syntheses.add(node_id)
                    
                # Save state after attempting to process this node
                self.save_state()
                
                # Continue to the next node
                continue
            
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
            self.save_state()
        
        return True  # Successfully processed nodes
   
    def _process_even_level(self, depth: int) -> bool:
        """
        Process all antithesis nodes at the given depth.
        Generate syntheses and direct replies for each node that hasn't been processed yet.
        Uses batch processing for improved performance.
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
            
        # Prepare batch processing
        thesis_antithesis_pairs = []
        
        for node_id in nodes_to_process:
            node_data = self.graph[node_id]
            antithesis_content = (node_data["summary"], node_data["content"])
            
            # Get parent thesis/synthesis content
            parent_id = node_data["parent_id"]
            parent_data = self.graph[parent_id]
            parent_content = (parent_data["summary"], parent_data["content"])
            
            thesis_antithesis_pairs.append((parent_content, antithesis_content))
        
        logging.info(f"Batch generating syntheses for {len(thesis_antithesis_pairs)} thesis-antithesis pairs")
        
        # Generate syntheses in batch
        syntheses_sets = self.generate_syntheses_batch(thesis_antithesis_pairs)
        
        # Generate direct replies in batch
        logging.info(f"Batch generating direct replies for {len(thesis_antithesis_pairs)} thesis-antithesis pairs")
        direct_replies_sets = self.generate_direct_replies_batch(thesis_antithesis_pairs)
        
        # Process each node with its generated syntheses and direct replies
        for i, node_id in enumerate(nodes_to_process):
            # Process syntheses
            syntheses = syntheses_sets[i]
            
            if syntheses is None:
                # Add a blank node if LLM response was None
                self.add_node(
                    summary="[Synthesis generation failed]",
                    content="The system was unable to generate a synthesis from this thesis-antithesis pair.",
                    node_type="synthesis_error",
                    parent_id=node_id
                )
                # Save state after attempting to process this node
                self.save_state()
            else:
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
            
            # Process direct replies
            direct_replies = direct_replies_sets[i]
            
            if direct_replies is None:
                # Add a blank node if LLM response was None
                self.add_node(
                    summary="[Direct Reply generation failed]",
                    content="The system was unable to generate a direct reply from this thesis-antithesis pair.",
                    node_type="direct_reply_error",
                    parent_id=node_id
                )
                # Save state after attempting to process this node
                self.save_state()
            else:
                # Add direct reply nodes
                for direct_reply in direct_replies:
                    reply_id = self.add_node(
                        summary=direct_reply[0],
                        content=direct_reply[1],
                        node_type="direct_reply",
                        parent_id=node_id
                    )
                    
                    if self._check_time_exceeded():
                        # Mark this node as processed before returning
                        self.processed_antitheses.add(node_id)
                        return True
            
            # Mark this node as processed
            self.processed_antitheses.add(node_id)
            
            # Save state after processing each node
            self.save_state()
        
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

    

    def save_state(self):
        """Save the entire object state to disk"""
        # Create a snapshot of the current state
        snapshot = {
            # Core graph data
            'graph': self.graph,
            'edges': self.edges,
            'central_question': self.central_question,
            
            # Configuration 
            'config': {
                'num_responses': self.num_responses,
                'num_reasons': self.num_reasons, 
                'max_depth': self.max_depth,
                'nonsense_threshold': self.nonsense_threshold,
                'view_identity_threshold': self.view_identity_threshold,
                'check_termination': self.check_termination,
            },
            
            # Processing state
            'state': {
                'processed_theses': list(self.processed_theses),
                'processed_antitheses': list(self.processed_antitheses),
                'processed_syntheses': list(self.processed_syntheses),
                'theses_with_reasons': list(self.theses_with_reasons),
                'theses_with_antitheses': list(self.theses_with_antitheses),
            }
        }
        
        # Use atomic write pattern
        temp_file = os.path.join(self.save_dir, 'graph_state.pickle.tmp')
        final_file = os.path.join(self.save_dir, 'graph_state.pickle')
        
        with open(temp_file, 'wb') as f:
            pickle.dump(snapshot, f)
            f.flush()
            os.fsync(f.fileno())
        
        os.replace(temp_file, final_file)


    @classmethod
    def load(cls, save_dir, default_question=None):
        """Factory method to create an instance from a saved state"""
        save_file = os.path.join(save_dir, 'graph_state.pickle')
        
        if os.path.exists(save_file):
            try:
                print(f"Loading graph state from {save_dir}")
                with open(save_file, 'rb') as f:
                    snapshot = pickle.load(f)
                
                # Create a new instance with proper initialization from saved config
                config = snapshot['config']
                
                # Create a new instance with saved configuration
                instance = cls(
                    central_question=default_question or "placeholder",
                    save_dir=save_dir,
                    # Pass the saved configuration values
                    max_depth=config.get('max_depth'),
                    max_time_seconds=config.get('max_time_seconds'),
                    num_responses=config.get('num_responses'),
                    num_reasons=config.get('num_reasons'),
                    nonsense_threshold=config.get('nonsense_threshold'),
                    view_identity_threshold=config.get('view_identity_threshold'),
                    check_termination=config.get('check_termination', False)
                )
                
                # Now overwrite the state with our loaded data
                instance.graph = snapshot['graph']
                instance.edges = snapshot['edges']
                instance.central_question = snapshot['central_question']
                
                # Restore configuration
                for key, value in snapshot['config'].items():
                    setattr(instance, key, value)
                    
                # Restore processing state
                for key, value in snapshot['state'].items():
                    setattr(instance, key, set(value))
                
                # Extract question text
                if instance.central_question in instance.graph:
                    instance.central_question_text = instance.graph[instance.central_question].get("content")
                
                print(f"Successfully loaded graph state from {save_dir}")
                return instance
                    
            except Exception as e:
                print(f"Failed to load saved state: {e}")
                # Fall through to create new instance
        
        # Create new instance if loading failed or no save exists
        if default_question is None:
            raise ValueError("No save file found and no default question provided")
        
        return cls(central_question=default_question, save_dir=save_dir)

    def checkpoint(self):
        """Save the current state as a checkpoint"""
        self.save_state()
        # Save FAISS if needed (currently not in use)
        # if self.check_termination:
        #     self._save_faiss()




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



    def generate_view_identity(self, view_a: str, view_b: str) -> float:
        """Compare two philosophical views and return a matching score from 0-100"""
        # Skip if termination checks are disabled
        if not self.check_termination:
            return 0.0  # Always return "not identical" when checks are disabled
            
        system_role = ""
        prompt = self.prompts["view_identity"].format(
            view_a=view_a,
            view_b=view_b
        )

        try:
            result = generate_completion(prompt, system_role)
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
            
        system_role = ""
        prompt = self.prompts["nonsense"].format(
            view=view
        )

        try:
            result = generate_completion(prompt, system_role)
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