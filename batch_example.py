"""
Example script showing how to use batch API calls for the dialectical graph
"""

import os
import time
from dialectical_question_graph import DialecticalGraph
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Create a graph instance
    graph = DialecticalGraph(
        central_question="What is the nature of freedom?",
        num_responses=5,   # Maximum responses per generation
        num_reasons=3,     # Maximum reasons per thesis
        max_depth=3,       # Maximum depth of the graph
        max_time_seconds=30*60,  # 30 minutes max
        check_termination=False  # Don't check for redundant views for this example
    )
    
    # Example 1: Generate multiple thesis sets in parallel
    logging.info("Generating thesis sets in parallel...")
    start_time = time.time()
    thesis_sets = graph.generate_theses_batch(num_batches=3)
    end_time = time.time()
    logging.info(f"Generated {sum(len(ts) if ts else 0 for ts in thesis_sets)} total theses in {end_time - start_time:.2f} seconds")
    
    # Extract and flatten the theses
    all_theses = []
    for thesis_set in thesis_sets:
        if thesis_set:
            all_theses.extend(thesis_set)
    
    if not all_theses:
        logging.error("No theses were generated!")
        return
    
    # Take just the first 3 theses for demonstration
    selected_theses = all_theses[:3]
    logging.info(f"Selected {len(selected_theses)} theses for further processing")
    
    # Example 2: Generate reasons for multiple theses in parallel
    logging.info("Generating reasons for multiple theses in parallel...")
    start_time = time.time()
    reasons_sets = graph.generate_reasons_batch(selected_theses)
    end_time = time.time()
    
    total_reasons = sum(len(rs) if rs else 0 for rs in reasons_sets)
    logging.info(f"Generated {total_reasons} reasons in {end_time - start_time:.2f} seconds")
    
    # Example 3: Generate antitheses for multiple theses in parallel
    logging.info("Generating antitheses for multiple theses in parallel...")
    start_time = time.time()
    antitheses_sets = graph.generate_antitheses_batch(selected_theses)
    end_time = time.time()
    
    total_antitheses = sum(len(a_set) if a_set else 0 for a_set in antitheses_sets)
    logging.info(f"Generated {total_antitheses} antitheses in {end_time - start_time:.2f} seconds")
    
    # Create thesis-antithesis pairs
    thesis_antithesis_pairs = []
    for i, antithesis_set in enumerate(antitheses_sets):
        if antithesis_set and len(antithesis_set) > 0:
            # Take the first antithesis from each set
            thesis_antithesis_pairs.append((selected_theses[i], antithesis_set[0]))
    
    logging.info(f"Created {len(thesis_antithesis_pairs)} thesis-antithesis pairs")
    
    if not thesis_antithesis_pairs:
        logging.error("No valid thesis-antithesis pairs were created!")
        return
    
    # Example 4: Generate syntheses for multiple thesis-antithesis pairs in parallel
    logging.info("Generating syntheses in parallel...")
    start_time = time.time()
    syntheses_sets = graph.generate_syntheses_batch(thesis_antithesis_pairs)
    end_time = time.time()
    
    total_syntheses = sum(len(s_set) if s_set else 0 for s_set in syntheses_sets)
    logging.info(f"Generated {total_syntheses} syntheses in {end_time - start_time:.2f} seconds")
    
    # Example 5: Generate direct replies for multiple thesis-antithesis pairs in parallel
    logging.info("Generating direct replies in parallel...")
    start_time = time.time()
    direct_replies_sets = graph.generate_direct_replies_batch(thesis_antithesis_pairs)
    end_time = time.time()
    
    total_direct_replies = sum(len(dr_set) if dr_set else 0 for dr_set in direct_replies_sets)
    logging.info(f"Generated {total_direct_replies} direct replies in {end_time - start_time:.2f} seconds")
    
    # Print some examples of the generated content
    if all_theses:
        logging.info(f"Example thesis: {all_theses[0][0]} - {all_theses[0][1][:100]}...")
    
    if thesis_antithesis_pairs and len(thesis_antithesis_pairs) > 0:
        logging.info(f"Example antithesis: {thesis_antithesis_pairs[0][1][0]} - {thesis_antithesis_pairs[0][1][1][:100]}...")
    
    if syntheses_sets and len(syntheses_sets) > 0 and syntheses_sets[0] and len(syntheses_sets[0]) > 0:
        logging.info(f"Example synthesis: {syntheses_sets[0][0][0]} - {syntheses_sets[0][0][1][:100]}...")
    
if __name__ == "__main__":
    main()