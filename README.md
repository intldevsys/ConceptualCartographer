# Conceptual Cartographer 2.0

#### NOTE: This project is the result of an experiment in "minimal guidance" collaborative AI prompting using gemini-2.5-pro-preview-03-25, wherein two language models were provided an open-ended task of collaborating and completing a piece of working code; the utility of which  was also left intentionally undefined. No additional instructions or guidance was included.

#### An audit was also performed by an additional LLM, outlining some potential fundamental flaws hindering efficiency with the first iteration and the Claude Opus-modified product is what is shown here. The full documentation for the process from initial autonomous collaboration to the final product will be included upon request. The end product, as well as the summary and workflow diagrams below were LLM-produced, as well.

An advanced natural language processing tool that automatically discovers conceptual structures, causal relationships, and semantic influences within text documents. Built with state-of-the-art machine learning techniques including transformer embeddings, HDBSCAN clustering, and statistical inference.

<img width="1784" height="904" alt="image" src="https://github.com/user-attachments/assets/b4028183-efa4-40de-818a-bc798f8deff1" />


##  What It Does

Conceptual Cartographer transforms unstructured text into rich, interactive conceptual maps by:

- **Discovering Conceptual Kernels**: Identifies core concepts and themes using advanced clustering
- **Extracting Causal Relations**: Finds cause-effect relationships between concepts using neural relation extraction
- **Analyzing Semantic Influences**: Detects how concepts influence each other using rigorous statistical methods
- **Generating Interactive Visualizations**: Creates beautiful, explorable network graphs of conceptual relationships

## 🎯 Use Cases

### Research & Analysis
- **Academic Research**: Analyze literature to identify key themes and their relationships
- **Market Research**: Extract insights from customer feedback, surveys, and reports
- **Policy Analysis**: Understand complex policy documents and their interconnected concepts

### Content Strategy
- **Content Planning**: Identify gaps and relationships in your content ecosystem  
- **SEO Strategy**: Discover semantic relationships for better content optimization
- **Knowledge Management**: Map organizational knowledge and identify key concepts

### Education & Learning
- **Curriculum Design**: Visualize concept dependencies and learning pathways
- **Research Planning**: Identify research themes and their interconnections
- **Knowledge Synthesis**: Combine insights from multiple documents

### Business Intelligence
- **Strategy Analysis**: Map strategic concepts and their causal relationships
- **Process Documentation**: Understand complex business processes and their dependencies
- **Innovation Mapping**: Identify innovation opportunities through concept analysis
- 
<img width="1618" height="928" alt="image" src="https://github.com/user-attachments/assets/dacd54fa-1b31-446b-b3d1-e102ea708ad5" />

## 🔧 How It Works

The system employs a sophisticated four-phase analysis pipeline:

### Phase 1: Kernel Discovery
- Validates and preprocesses input text
- Generates sentence embeddings using SentenceTransformers
- Applies HDBSCAN clustering to identify conceptual clusters
- Validates cluster stability using bootstrap resampling
- Creates interpretable labels for each conceptual kernel

### Phase 2: Relation Extraction  
- Uses neural language models for relation detection
- Applies natural language inference to identify causal relationships
- Extracts relations like causation, prevention, and enablement
- Merges duplicate relations and assigns confidence scores

### Phase 3: Influence Analysis
- Performs sliding window analysis across the text
- Computes local context embeddings to detect concept activation
- Uses statistical testing (t-tests, bootstrap CI) to validate influences
- Identifies attraction/repulsion patterns between concept pairs

### Phase 4: Visualization
- Computes optimal graph layouts using NetworkX algorithms
- Generates interactive HTML visualizations with vis.js
- Provides detailed information panels and controls
- Supports exploration of concepts, relations, and influences

## 📊 Example Output

The tool generates rich, interactive visualizations showing:

- **Concept Nodes**: Sized by confidence, colored by stability
- **Causal Edges**: Color-coded by relation type (causation, prevention, enablement)
- **Statistical Metrics**: Confidence intervals, p-values, effect sizes
- **Interactive Controls**: Zoom, pan, physics simulation toggle

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/conceptual-cartographer.git
cd conceptual-cartographer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 📝 Usage

### Command Line
```bash
# Analyze a text file
python concart.py your_document.txt

# The tool will generate an interactive HTML visualization
# that opens automatically in your browser
```

### Python API
```python
from concart import ConceptualCartographer

# Initialize the cartographer
cartographer = ConceptualCartographer()

# Analyze text
with open('your_document.txt', 'r') as f:
    text = f.read()

results = cartographer.analyze(text, output_visualization=True)

# Access results
print(f"Found {len(results['kernels'])} conceptual kernels")
print(f"Extracted {len(results['relations'])} causal relations")
print(f"Discovered {len(results['influences'])} semantic influences")
```

## 📋 Requirements

```
numpy>=1.21.0
spacy>=3.4.0
scikit-learn>=1.0.0
sentence-transformers>=2.2.0
hdbscan>=0.8.27
networkx>=2.8.0
transformers>=4.20.0
torch>=1.12.0
scipy>=1.8.0
argparse
```

## 🔬 Technical Details

### Algorithms Used
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Clustering**: HDBSCAN with stability validation
- **Relation Extraction**: Neural similarity with template matching
- **Statistical Testing**: Bootstrap resampling, t-tests
- **Visualization**: Force-directed graph layout

### Key Features
- **Robust preprocessing** with comprehensive text validation
- **Statistical rigor** with p-values and confidence intervals  
- **Stability validation** using bootstrap resampling
- **Interactive visualizations** with modern web technologies
- **Scalable processing** with batch embedding generation

## 🎨 Sample Results

For a document about AI ethics, the tool might discover:

**Conceptual Kernels:**
- Machine Learning Algorithms (confidence: 0.89)
- Ethical AI Development (confidence: 0.76) 
- Bias Detection Methods (confidence: 0.82)

**Causal Relations:**
- Biased Training Data → Unfair AI Outcomes (confidence: 0.91)
- Explainable AI → Increased Trust (confidence: 0.78)

**Semantic Influences:**
- "Ethics" attracts "Transparency" and "Fairness" (p < 0.001)
- "Automation" repels "Human Oversight" and "Control" (p < 0.01)

## 🚦 System Architecture

The tool follows a modular architecture with clear separation of concerns:

- `RobustKernelDiscoverer`: Handles concept discovery and clustering
- `NeuralRelationExtractor`: Manages causal relation extraction
- `StatisticalInfluenceAnalyzer`: Performs rigorous influence analysis  
- `EnhancedVisualizer`: Creates interactive visualizations
- `ConceptualCartographer`: Orchestrates the complete pipeline

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Built with open-source libraries including:
- Sentence Transformers for embeddings
- HDBSCAN for clustering
- spaCy for NLP preprocessing
- NetworkX for graph algorithms
- vis.js for interactive visualizations
