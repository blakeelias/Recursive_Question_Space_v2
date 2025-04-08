# Recursive Question Space

## Project Description

Recursive Question Space is a computational philosophy tool that generates dialectical knowledge graphs through recursive exploration of philosophical concepts. The system uses Large Language Models (LLMs) to create a complex web of interconnected philosophical views, objections, and reconciliations, starting from a central philosophical question.

The project automates the philosophical process of thesis → antithesis → synthesis, creating a rich landscape of conceptual exploration that would be difficult to produce manually. The resulting knowledge structures can be exported as interconnected markdown files, facilitating comprehensive study and analysis of philosophical domains.

## How It Works

The project implements a dialectical graph approach to philosophical exploration:

1. **Central Question**: The process begins with a central philosophical question (e.g., "What is knowledge?", "What is freedom?").

2. **Dialectical Exploration**:
   - **Thesis Generation**: The system prompts an LLM to generate multiple distinct philosophical views (theses) that answer the central question.
   - **Reason Generation**: For each thesis, supporting reasons are generated to strengthen the position.
   - **Antithesis Generation**: For each thesis, the system generates objections or counterarguments (antitheses).
   - **Synthesis Generation**: For each antithesis, the system creates "synthesis" views that reconcile the original thesis with the objection.
   - **Direct Replies**: Alternative responses to objections are also generated.

3. **Recursive Exploration**: This process continues recursively, with each synthesis potentially spawning new antitheses, creating a deeply nested graph of philosophical discourse.

4. **Similarity Detection**: The system uses embeddings and similarity detection to identify duplicate or highly similar views, preventing redundancy in the graph.

5. **Termination Conditions**: The exploration can be limited by maximum depth, time constraints, or by detecting when the graph has reached a saturation point.

## Installation Instructions

### Prerequisites
- Python 3.8+
- OpenAI API key (or other compatible LLM provider)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/recursive_question_space.git
   cd recursive_question_space
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage Examples

### Basic Usage

```python
from dialectical_question_graph import DialecticalGraph

# Initialize a new graph
graph = DialecticalGraph(
    central_question="What is knowledge?",
    num_responses=10,  # Maximum number of responses per level
    num_reasons=5,     # Maximum number of reasons per thesis
    max_depth=4,       # Maximum depth of the graph
    max_time_seconds=30*60  # Maximum time in seconds (30 minutes)
)

# The graph generation starts automatically during initialization

# Export the graph to JSON files
graph.save_state("./graph_exports/my_graph")

# Convert the graph to a markdown wiki
from graph_to_markdown import create_markdown_wiki
create_markdown_wiki("./graph_exports/my_graph/graph.json", "wiki_knowledge")
```

### From Command Line

Run the main script to generate a graph with default settings:

```bash
python main.py
```

You can modify the parameters in `main.py` to adjust the central question, exploration depth, and other settings.

## Project Structure

The project consists of the following key files and directories:

- **`dialectical_question_graph.py`**: Core class that implements the dialectical graph structure and exploration algorithm.
- **`main.py`**: Entry point for running graph generation with configurable parameters.
- **`prompt_llm.py`**: Functions for interacting with language models and processing responses.
- **`embedding.py`**: Handles text embeddings for similarity detection and search.
- **`graph_to_markdown.py`**: Converts graph data to interconnected markdown files.
- **`prompts/`**: Directory containing prompt templates:
  - `thesis_prompt.txt`: For generating initial philosophical positions
  - `antithesis_prompt.txt`: For generating objections to positions
  - `synthesis_prompt.txt`: For generating reconciliations of objections
  - `reasons_prompt.txt`: For generating supporting reasons for theses
  - `direct_reply_prompt.txt`: For generating alternative responses to objections
  - `nonsense_prompt.txt`: For detecting irrelevant or nonsensical responses
  - `view_identity_prompt.txt`: For detecting duplicate or highly similar views
- **`Graph_Exports/`**: Default directory for exported graph data.
- **`temp/`**: Directory for temporary storage and state backups during graph generation.

## Configuration Options

The `DialecticalGraph` class supports several configuration parameters:

- **`central_question`**: The philosophical question to explore.
- **`num_responses`**: Maximum number of responses (theses/antitheses/syntheses) to generate at each level.
- **`num_reasons`**: Maximum number of supporting reasons to generate for each thesis.
- **`max_depth`**: Maximum recursion depth for the graph exploration.
- **`max_time_seconds`**: Maximum time limit for graph generation.
- **`nonsense_threshold`**: Similarity threshold for detecting nonsensical responses (0-100).
- **`view_identity_threshold`**: Similarity threshold for detecting duplicate views (0-100).
- **`check_termination`**: Whether to check for graph saturation during generation.
- **`save_dir`**: Directory for saving state backups.

## Advanced Features

### Graph Analysis

Use the `graph-analysis.py` script to analyze the structure and content of generated graphs:

```bash
python graph-analysis.py --input ./graph_exports/my_graph/graph.json
```

### Converting to Wiki Format

Convert a graph to an interconnected wiki of markdown files:

```bash
python graph_to_markdown.py --input ./graph_exports/my_graph/graph.json --output ./wiki_output
```

### Visualization

Visualize the graph structure using the included visualization tools:

```bash
python visualise.py --input ./graph_exports/my_graph/graph.json
```

## License

MIT License

## Acknowledgments

This project was developed as part of research into computational philosophy and automated knowledge generation using large language models.