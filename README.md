# Recursive Question Space

## Project Description

Recursive Question Space is a computational philosophy tool that generates dialectical knowledge graphs (called "inquiry complexes") through recursive exploration of philosophical concepts. The system uses Large Language Models (LLMs) to create a complex web of interconnected philosophical views, objections, and reconciliations. There are two main ways to create inquiry complexes:

1. **From a Central Question**: Starting with a user-defined philosophical question
2. **From Text Documents**: Extracting philosophical questions and positions from existing texts

The project automates the philosophical process of thesis → antithesis → synthesis, creating a rich landscape of conceptual exploration that would be difficult to produce manually. The resulting knowledge structures can be exported as interconnected markdown files, facilitating comprehensive study and analysis of philosophical domains.

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

## Creating Inquiry Complexes

### Method 1: From a Central Question

This approach generates a dialectical graph starting from a philosophical question you provide.

#### Basic Usage

```python
from dialectical_question_graph import DialecticalGraph

# Initialize a new graph
graph = DialecticalGraph(
    central_question="What is knowledge?",
    num_responses=10,           # Maximum responses per dialectical level
    num_reasons=5,              # Maximum reasons per thesis
    max_depth=4,                # Maximum depth of exploration
    max_time_seconds=30*60,     # Maximum time in seconds (30 minutes)
    nonsense_threshold=95,      # Threshold for filtering nonsense (0-100)
    view_identity_threshold=95, # Threshold for detecting duplicates (0-100)
    save_dir="./temp"           # Directory for state backups
)

# Export the completed graph
export_dir = "./Graph_Exports/my_knowledge_graph"
graph.save_state(export_dir)
```

#### Command Line Usage

Modify the parameters in `main.py` and run:

```bash
python main.py
```

#### Parameters for Different Complex Sizes

**Small Complex (Quick exploration)**:
```python
graph = DialecticalGraph(
    central_question="What is freedom?",
    num_responses=5,        # Fewer responses per level
    num_reasons=3,          # Fewer supporting reasons
    max_depth=2,            # Shallow exploration
    max_time_seconds=10*60, # 10 minutes
    save_dir="./temp"
)
```

**Medium Complex (Balanced exploration)**:
```python
graph = DialecticalGraph(
    central_question="What is justice?",
    num_responses=15,       # Moderate responses per level
    num_reasons=7,          # Moderate supporting reasons
    max_depth=4,            # Medium depth
    max_time_seconds=45*60, # 45 minutes
    save_dir="./temp"
)
```

**Large Complex (Deep exploration)**:
```python
graph = DialecticalGraph(
    central_question="What is consciousness?",
    num_responses=30,        # Many responses per level
    num_reasons=10,          # Many supporting reasons
    max_depth=6,             # Deep exploration
    max_time_seconds=120*60, # 2 hours
    save_dir="./temp"
)
```

### Method 2: From Text Documents

This approach extracts philosophical positions and questions from existing texts, then builds dialectical structures around them.

#### Basic Usage

```python
from text_to_inquiry_complex import InquiryComplexExtractor

# Load your text document
with open("input_texts/my_philosophy_paper.txt", "r") as f:
    text_content = f.read()

# Create the extractor
extractor = InquiryComplexExtractor(
    text_passage=text_content,
    max_depth=3,              # Dialectical exploration depth
    top_n_questions=5,        # Number of top questions to explore
    save_dir="./temp"         # Save directory
)

# Extract the inquiry complex
inquiry_complex = extractor.extract_inquiry_complex()

# Save to file
extractor.save_to_file("my_text_inquiry_complex.json")
```

#### Parameters for Different Sizes

**Small Text-Based Complex**:
```python
extractor = InquiryComplexExtractor(
    text_passage=text_content,
    max_depth=2,              # Shallow dialectical depth
    top_n_questions=2,        # Focus on 2 main questions
    save_dir="./input_texts/ICs_from_Texts"
)
```

**Medium Text-Based Complex**:
```python
extractor = InquiryComplexExtractor(
    text_passage=text_content,
    max_depth=3,              # Medium dialectical depth
    top_n_questions=5,        # Explore 5 main questions
    save_dir="./input_texts/ICs_from_Texts"
)
```

**Large Text-Based Complex**:
```python
extractor = InquiryComplexExtractor(
    text_passage=text_content,
    max_depth=5,              # Deep dialectical exploration
    top_n_questions=8,        # Explore many questions
    save_dir="./input_texts/ICs_from_Texts"
)
```

## Save Directory Specification

Both methods allow you to specify where outputs are saved:

### For Central Question Method
```python
# Save to default Graph_Exports with timestamp
save_dir = "./temp"  # For intermediate state backups

# Final export location is automatically generated as:
# ./Graph_Exports/{question}_{depth}_{time}_{timestamp}/
```

### For Text-Based Method
```python
# Specify exact save directory
save_dir = "./input_texts/ICs_from_Texts"  # Custom directory
save_dir = "./custom_output_folder"        # Any custom path
save_dir = "~/Documents/my_inquiry_complexes"  # User directory
```

### Save Directory Structure

**Central Question Method** creates:
```
Graph_Exports/
├── what_is_knowledge_depth4_time30min_20240615_143022/
│   ├── graph.json      # Complete graph structure
│   ├── edges.json      # Graph relationships
│   └── metadata.json   # Generation parameters
```

**Text-Based Method** creates:
```
{save_dir}/
├── my_text_inquiry_complex.json  # Complete inquiry complex
└── analysis_reports/              # Optional analysis files
```

## Configuration Parameters

### Central Question Method Parameters

- **`central_question`**: The philosophical question to explore
- **`num_responses`**: Max responses per dialectical level (5-50 recommended)
- **`num_reasons`**: Max supporting reasons per thesis (3-15 recommended)
- **`max_depth`**: Dialectical recursion depth (2-8 recommended)
- **`max_time_seconds`**: Time limit in seconds (600-7200 for 10min-2hrs)
- **`nonsense_threshold`**: Filter nonsensical responses (90-99 recommended)
- **`view_identity_threshold`**: Filter duplicate views (90-99 recommended)
- **`save_dir`**: Directory for state backups

### Text-Based Method Parameters

- **`text_passage`**: The input text to analyze
- **`max_depth`**: Dialectical exploration depth (2-5 recommended)
- **`top_n_questions`**: Number of questions to explore (2-10 recommended)
- **`save_dir`**: Directory to save the inquiry complex

## Analysis and Visualization

### Analyzing Generated Complexes

```python
from inquiry_complex_utils import InquiryComplexAnalyzer, load_inquiry_complex

# Load and analyze
complex_data = load_inquiry_complex("path/to/your_complex.json")
analyzer = InquiryComplexAnalyzer(complex_data)

# Get detailed statistics
stats = analyzer.get_detailed_stats()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Node types: {stats['node_types']}")

# Generate visualization
analyzer.visualize_graph("my_complex_visualization.png")

# Export summary report
analyzer.export_summary_report("my_complex_report.txt")
```

### Converting to Wiki Format

```python
from graph_to_markdown import create_markdown_wiki

# Convert central question graphs
create_markdown_wiki("./Graph_Exports/my_graph/graph.json", "wiki_output")

# Convert text-based inquiry complexes  
create_markdown_wiki("./input_texts/ICs_from_Texts/my_complex.json", "wiki_output")
```

## Project Structure

- **`dialectical_question_graph.py`**: Core dialectical graph generation from central questions
- **`text_to_inquiry_complex.py`**: Text-based inquiry complex extraction
- **`inquiry_complex_utils.py`**: Analysis and visualization utilities
- **`main.py`**: Entry point for central question method
- **`prompt_llm.py`**: LLM interaction functions
- **`embedding.py`**: Text embeddings for similarity detection
- **`graph_to_markdown.py`**: Export to markdown wiki format
- **`prompts/original/`**: Prompts for central question method
- **`prompts/text/`**: Prompts for text-based method
- **`input_texts/`**: Sample texts and text-based outputs
- **`Graph_Exports/`**: Central question method outputs
- **`temp/`**: Temporary files and state backups

## Examples

The repository includes examples of both methods:
- **Central question graphs**: See `Graph_Exports/` for various philosophical questions
- **Text-based complexes**: See `input_texts/ICs_from_Texts/` for complexes extracted from philosophical papers

## License

MIT License

## Acknowledgments

This project was developed as part of research into computational philosophy and automated knowledge generation using large language models.