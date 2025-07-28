import numpy as np
import spacy
import argparse
import json
import webbrowser
import os
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import HDBSCAN
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import logging
from scipy import stats
import warnings
from collections import defaultdict
import networkx as nx
from itertools import combinations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
        
class Config:
    """Centralized configuration for the system."""
    # Model configurations
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    RELATION_MODEL = 'microsoft/deberta-v3-base'  # For relation extraction
    
    # Statistical thresholds
    MIN_OBSERVATIONS_FOR_INFLUENCE = 5
    SIGNIFICANCE_LEVEL = 0.05
    BOOTSTRAP_ITERATIONS = 1000
    
    # Clustering parameters
    MIN_CLUSTER_SIZE = 2
    MIN_SAMPLES = 1 
    
    # Analysis parameters
    WINDOW_SIZE = 5
    ACTIVATION_THRESHOLD = 0.6
    
    # Performance optimizations
    BATCH_SIZE = 32
    MAX_TEXT_LENGTH = 100000

# =============================================================================
# Enhanced Data Structures
# =============================================================================

@dataclass
class ConceptualKernel:
    """Represents a discovered concept with enhanced metadata."""
    id: int
    vector: np.ndarray
    label: str
    source_sentences: List[str]
    confidence: float  # Statistical confidence in this cluster
    stability: float   # Cluster stability score
    metadata: Dict = field(default_factory=dict)

@dataclass
class CausalRelation:
    """Represents a causal relationship with confidence scores."""
    cause_kernel_id: int
    effect_kernel_id: int
    confidence: float
    evidence_sentences: List[str]
    relation_type: str  # 'direct_cause', 'enables', 'prevents', etc.
    statistical_support: Dict  # p-values, effect sizes, etc.

@dataclass
class SemanticInfluence:
    """Represents an influence with rigorous statistical backing."""
    influencer_id: int
    affected_pair: Tuple[int, int]
    influence_type: str  # 'attraction', 'repulsion', 'modulation'
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    observations: List[float]
    context_windows: List[str]  # For interpretability

# =============================================================================
# Core Components with Improved Implementations
# =============================================================================

class RobustKernelDiscoverer:
    """Enhanced kernel discovery with validation and stability checks."""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.embedding_model = None
        self.nlp = None
        self._load_models()
    
    def _load_models(self):
        """Lazy loading of models with error handling."""
        try:
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _validate_text(self, text: str) -> List[str]:
        """Validate and preprocess text with comprehensive checks."""
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for meaningful analysis")
        
        if len(text) > self.config.MAX_TEXT_LENGTH:
            warnings.warn(f"Text exceeds maximum length, truncating to {self.config.MAX_TEXT_LENGTH} characters")
            text = text[:self.config.MAX_TEXT_LENGTH]
        
        doc = self.nlp(text)
        sentences = []
        
        for sent in doc.sents:
            cleaned = sent.text.strip()
            # Filter out very short sentences and non-sentences
            if len(cleaned) > 10 and any(char.isalpha() for char in cleaned):
                sentences.append(cleaned)
        
        if len(sentences) < self.config.MIN_CLUSTER_SIZE:
            raise ValueError(f"Not enough valid sentences ({len(sentences)}) for clustering")
        
        return sentences
    
    def _compute_cluster_stability(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
        """Compute stability scores for each cluster using bootstrap resampling."""
        stability_scores = {}
        unique_labels = set(labels) - {-1}
        
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < 3:
                stability_scores[label] = 0.0
                continue
            
            # Bootstrap resampling to test cluster stability
            stabilities = []
            for _ in range(100):  # Reduced iterations for performance
                sample_indices = np.random.choice(cluster_indices, size=len(cluster_indices), replace=True)
                sample_embeddings = embeddings[sample_indices]
                
                # Check if samples remain clustered together
                if len(np.unique(sample_indices)) > 1:
                    pairwise_sims = cosine_similarity(sample_embeddings)
                    avg_sim = np.mean(pairwise_sims[np.triu_indices_from(pairwise_sims, k=1)])
                    stabilities.append(avg_sim)
            
            stability_scores[label] = np.mean(stabilities) if stabilities else 0.0
        
        return stability_scores
    
    def _generate_interpretable_label(self, sentences: List[str], embeddings: np.ndarray) -> str:
        """Generate label using multiple strategies with fallback."""
        # Strategy 1: Find most representative sentence (closest to centroid)
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        representative_idx = np.argmin(distances)
        
        # Truncate if too long
        label = sentences[representative_idx]
        if len(label) > 100:
            label = label[:97] + "..."
        
        return label
    
    def discover(self, text: str) -> List[ConceptualKernel]:
        """Discover conceptual kernels with validation and confidence scoring."""
        logger.info("Starting kernel discovery...")
        
        # Validate and preprocess
        sentences = self._validate_text(text)
        logger.info(f"Processing {len(sentences)} valid sentences")
        
        # Embed sentences in batches for efficiency
        embeddings = []
        for i in range(0, len(sentences), self.config.BATCH_SIZE):
            batch = sentences[i:i + self.config.BATCH_SIZE]
            batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        embeddings = np.array(embeddings)
        
        # Clustering with HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=self.config.MIN_CLUSTER_SIZE,
            min_samples=self.config.MIN_SAMPLES,
            metric='euclidean',  # More stable than cosine for HDBSCAN
            cluster_selection_epsilon=0.3,  # Helps with stability
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Compute cluster stability
        stability_scores = self._compute_cluster_stability(embeddings, labels)
        
        # Build kernels with confidence scores
        kernels = []
        for label in set(labels) - {-1}:
            cluster_indices = np.where(labels == label)[0]
            cluster_sentences = [sentences[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]
            
            # Skip unstable clusters
            if stability_scores.get(label, 0) < 0.5:
                logger.warning(f"Skipping unstable cluster {label}")
                continue
            
            kernel = ConceptualKernel(
                id=label,
                vector=np.mean(cluster_embeddings, axis=0),
                label=self._generate_interpretable_label(cluster_sentences, cluster_embeddings),
                source_sentences=cluster_sentences,
                confidence=1.0,
                stability=stability_scores.get(label, 0),
                metadata={
                    'size': len(cluster_sentences),
                    'coherence': float(np.mean([cosine_similarity([emb1], [emb2])[0][0] for i, emb1 in enumerate(cluster_embeddings) for j, emb2 in enumerate(cluster_embeddings) if i < j]))
                }
            )
            kernels.append(kernel)
        
        logger.info(f"Discovered {len(kernels)} stable conceptual kernels")
        return kernels


class NeuralRelationExtractor:
    """Modern relation extraction using transformer models."""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.relation_model = None
        self.tokenizer = None
        self._relation_templates = {
            'causation': [
                "{} causes {}",
                "{} leads to {}",
                "{} results in {}"
            ],
            'prevention': [
                "{} prevents {}",
                "{} blocks {}",
                "{} inhibits {}"
            ],
            'enablement': [
                "{} enables {}",
                "{} allows {}",
                "{} facilitates {}"
            ]
        }
    
    def _load_relation_model(self):
        """Load a model for natural language inference to detect relations."""
        if self.relation_model is None:
            # Using NLI model as a zero-shot relation classifier
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            self.relation_model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-v3-base"
            )
    
    def _extract_noun_phrases(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract noun phrases with their positions."""
        doc = self.nlp(text)
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            noun_phrases.append((chunk.text, chunk.start_char, chunk.end_char))
        
        # Also include named entities
        for ent in doc.ents:
            noun_phrases.append((ent.text, ent.start_char, ent.end_char))
        
        return noun_phrases
    
    def _score_relation(self, sentence: str, phrase1: str, phrase2: str, relation_type: str) -> float:
        """Score a potential relation using NLI-style inference."""
        # This is a simplified version - in production, you'd use a proper relation extraction model
        
        # Check if both phrases are in the sentence
        if phrase1 not in sentence or phrase2 not in sentence:
            return 0.0
        
        # Use templates to check relation plausibility
        scores = []
        for template in self._relation_templates.get(relation_type, []):
            hypothesis = template.format(phrase1, phrase2)
            
            # Compute semantic similarity between sentence and hypothesis
            sent_emb = self.embedding_model.encode(sentence)
            hyp_emb = self.embedding_model.encode(hypothesis)
            similarity = cosine_similarity([sent_emb], [hyp_emb])[0][0]
            scores.append(similarity)
        
        return max(scores) if scores else 0.0
    
    def _map_phrase_to_kernel(self, phrase: str, kernels: List[ConceptualKernel]) -> Optional[int]:
        """Map a text phrase to the most likely kernel."""
        phrase_embedding = self.embedding_model.encode(phrase)
        
        best_kernel_id = None
        best_similarity = 0.0
        
        for kernel in kernels:
            similarity = cosine_similarity([phrase_embedding], [kernel.vector])[0][0]
            if similarity > best_similarity and similarity > 0.7:  # Threshold for matching
                best_similarity = similarity
                best_kernel_id = kernel.id
        
        return best_kernel_id
    
    def extract_relations(self, text: str, kernels: List[ConceptualKernel]) -> List[CausalRelation]:
        """Extract causal relations with confidence scores."""
        logger.info("Starting relation extraction...")
        
        doc = self.nlp(text)
        relations = []
        
        for sent in doc.sents:
            if len(sent.text.strip()) < 20:
                continue
            
            # Extract noun phrases from sentence
            noun_phrases = self._extract_noun_phrases(sent.text)
            
            # Check all pairs of noun phrases for potential relations
            for i, (phrase1, _, _) in enumerate(noun_phrases):
                for j, (phrase2, _, _) in enumerate(noun_phrases):
                    if i >= j:  # Skip self-relations and duplicates
                        continue
                    
                    # Map phrases to kernels
                    kernel1_id = self._map_phrase_to_kernel(phrase1, kernels)
                    kernel2_id = self._map_phrase_to_kernel(phrase2, kernels)
                    
                    if kernel1_id is None or kernel2_id is None or kernel1_id == kernel2_id:
                        continue
                    
                    # Score different relation types
                    for rel_type in ['causation', 'prevention', 'enablement']:
                        score = self._score_relation(sent.text, phrase1, phrase2, rel_type)
                        
                        if score > 0.75:  # High confidence threshold
                            relation = CausalRelation(
                                cause_kernel_id=kernel1_id,
                                effect_kernel_id=kernel2_id,
                                confidence=score,
                                evidence_sentences=[sent.text],
                                relation_type=rel_type,
                                statistical_support={
                                    'method': 'neural_similarity',
                                    'score': score
                                }
                            )
                            relations.append(relation)
        
        # Merge duplicate relations
        merged_relations = self._merge_duplicate_relations(relations)
        
        logger.info(f"Extracted {len(merged_relations)} high-confidence relations")
        return merged_relations
    
    def _merge_duplicate_relations(self, relations: List[CausalRelation]) -> List[CausalRelation]:
        """Merge relations with same cause-effect pairs."""
        relation_map = defaultdict(list)
        
        for rel in relations:
            key = (rel.cause_kernel_id, rel.effect_kernel_id, rel.relation_type)
            relation_map[key].append(rel)
        
        merged = []
        for key, rel_list in relation_map.items():
            if len(rel_list) == 1:
                merged.append(rel_list[0])
            else:
                # Merge evidence and update confidence
                merged_rel = CausalRelation(
                    cause_kernel_id=key[0],
                    effect_kernel_id=key[1],
                    confidence=np.mean([r.confidence for r in rel_list]),
                    evidence_sentences=list(set(sum([r.evidence_sentences for r in rel_list], []))),
                    relation_type=key[2],
                    statistical_support={
                        'method': 'neural_similarity',
                        'aggregated_score': np.mean([r.confidence for r in rel_list]),
                        'evidence_count': len(rel_list)
                    }
                )
                merged.append(merged_rel)
        
        return merged


class StatisticalInfluenceAnalyzer:
    """Rigorous influence analysis with proper statistical testing."""
    
    def __init__(self, kernels: List[ConceptualKernel], config: Config = Config()):
        self.config = config
        self.kernels = {k.id: k for k in kernels}
        self.kernel_vectors = {k.id: k.vector for k in kernels}
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.nlp = spacy.load("en_core_web_sm")
        self.global_state = self._compute_global_state()
    
    def _compute_global_state(self) -> Dict:
        """Compute baseline statistics for all kernel pairs."""
        state = {
            'distances': {},
            'similarities': {},
            'cooccurrence': defaultdict(int)
        }
        
        for k1_id, k2_id in combinations(self.kernels.keys(), 2):
            vec1, vec2 = self.kernel_vectors[k1_id], self.kernel_vectors[k2_id]
            
            # Euclidean distance
            state['distances'][(k1_id, k2_id)] = np.linalg.norm(vec1 - vec2)
            
            # Cosine similarity
            state['similarities'][(k1_id, k2_id)] = cosine_similarity([vec1], [vec2])[0][0]
        
        return state
    
    def _compute_local_context_embedding(self, sentences: List[str]) -> np.ndarray:
        """Compute a robust context embedding from multiple sentences."""
        embeddings = self.embedding_model.encode(sentences)
        
        # Use weighted average based on sentence position (more recent = higher weight)
        weights = np.linspace(0.5, 1.0, len(sentences))
        weights = weights / weights.sum()
        
        weighted_embedding = np.average(embeddings, axis=0, weights=weights)
        return weighted_embedding
    
    def _statistical_test_influence(self, observations: List[float], null_hypothesis: float = 0.0) -> Dict:
        """Perform rigorous statistical testing on influence observations."""
        if len(observations) < self.config.MIN_OBSERVATIONS_FOR_INFLUENCE:
            return {
                'significant': False,
                'p_value': 1.0,
                'effect_size': 0.0,
                'confidence_interval': (0.0, 0.0)
            }
        
        observations = np.array(observations)
        
        # T-test against null hypothesis
        t_stat, p_value = stats.ttest_1samp(observations, null_hypothesis)
        
        # Effect size (Cohen's d)
        effect_size = (np.mean(observations) - null_hypothesis) / (np.std(observations) + 1e-8)
        
        # Bootstrap confidence interval
        bootstrap_means = []
        for _ in range(self.config.BOOTSTRAP_ITERATIONS):
            bootstrap_sample = np.random.choice(observations, size=len(observations), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        confidence_interval = (
            np.percentile(bootstrap_means, 2.5),
            np.percentile(bootstrap_means, 97.5)
        )
        
        return {
            'significant': p_value < self.config.SIGNIFICANCE_LEVEL,
            'p_value': p_value,
            'effect_size': effect_size,
            'confidence_interval': confidence_interval,
            't_statistic': t_stat,
            'mean': np.mean(observations),
            'std': np.std(observations)
        }
    
    def analyze_influences(self, text: str) -> List[SemanticInfluence]:
        """Detect semantic influences with statistical rigor."""
        logger.info("Starting influence analysis with statistical validation...")
        
        doc = self.nlp(text)
        sentences = [s.text for s in doc.sents if len(s.text.strip()) > 10]
        
        # Track observations for each potential influence
        influence_observations = defaultdict(lambda: {
            'deltas': [],
            'contexts': []
        })
        
        # Sliding window analysis
        for i in range(len(sentences) - self.config.WINDOW_SIZE + 1):
            window_sentences = sentences[i:i + self.config.WINDOW_SIZE]
            context_embedding = self._compute_local_context_embedding(window_sentences)
            
            # Determine active kernels in this window
            active_kernels = []
            for kid, kvec in self.kernel_vectors.items():
                activation = cosine_similarity([context_embedding], [kvec])[0][0]
                if activation > self.config.ACTIVATION_THRESHOLD:
                    active_kernels.append(kid)
            
            # Need at least 3 kernels for influence analysis
            if len(active_kernels) < 3:
                continue
            
            # Analyze all possible influence patterns
            for influencer in active_kernels:
                for k1, k2 in combinations([k for k in active_kernels if k != influencer], 2):
                    pair = tuple(sorted([k1, k2]))
                    
                    # Skip if we don't have global baseline
                    if pair not in self.global_state['distances']:
                        continue
                    
                    # Compute local distance in context
                    vec1_local = self.kernel_vectors[k1]
                    vec2_local = self.kernel_vectors[k2]
                    
                    # Project onto context to get local distance
                    proj1 = np.dot(vec1_local, context_embedding)
                    proj2 = np.dot(vec2_local, context_embedding)
                    local_distance = abs(proj1 - proj2)
                    
                    # Normalize by global distance
                    global_distance = self.global_state['distances'][pair]
                    delta = (global_distance - local_distance) / (global_distance + 1e-8)
                    
                    # Record observation
                    influence_observations[(influencer, pair)]['deltas'].append(delta)
                    influence_observations[(influencer, pair)]['contexts'].append(
                        " ".join(window_sentences[:2])  # Store first 2 sentences for context
                    )
        
        # Statistical validation and influence creation
        validated_influences = []
        
        for (influencer, pair), data in influence_observations.items():
            test_results = self._statistical_test_influence(data['deltas'])
            
            if test_results['significant']:
                influence_type = 'attraction' if test_results['mean'] > 0 else 'repulsion'
                
                influence = SemanticInfluence(
                    influencer_id=influencer,
                    affected_pair=pair,
                    influence_type=influence_type,
                    effect_size=abs(test_results['effect_size']),
                    confidence_interval=test_results['confidence_interval'],
                    p_value=test_results['p_value'],
                    observations=data['deltas'],
                    context_windows=data['contexts'][:5]  # Keep top 5 for interpretability
                )
                validated_influences.append(influence)
        
        # Sort by effect size
        validated_influences.sort(key=lambda x: x.effect_size, reverse=True)
        
        logger.info(f"Found {len(validated_influences)} statistically significant influences")
        return validated_influences


class EnhancedVisualizer:
    """Improved visualization with better layouts and interactivity."""
    
    def __init__(self, kernels: List[ConceptualKernel], relations: List[CausalRelation],
                 influences: List[SemanticInfluence], text: str):
        self.kernels = kernels
        self.relations = relations
        self.influences = influences
        self.text = text
        self.nlp = spacy.load("en_core_web_sm")
    
    def _compute_graph_layout(self) -> Dict[int, Tuple[float, float]]:
        """Compute optimal node positions using graph algorithms."""
        G = nx.DiGraph()
        
        # Add nodes
        for kernel in self.kernels:
            G.add_node(kernel.id, weight=kernel.confidence)
        
        # Add edges from relations
        for relation in self.relations:
            G.add_edge(relation.cause_kernel_id, relation.effect_kernel_id, 
                      weight=relation.confidence)
        
        # Use spring layout with custom parameters
        pos = nx.spring_layout(G, k=1, iterations=50, weight='weight')
        
        # Scale and center positions properly
        positions = {}
        for node, (x, y) in pos.items():
            # Scale to reasonable canvas coordinates and center
            positions[node] = (x * 300 + 400, y * 300 + 300)  # Smaller scale, centered
        
        return positions
    
    def generate_interactive_visualization(self, output_file: str = "conceptual_map_v2.html"):
        """Generate an enhanced interactive visualization."""
        positions = self._compute_graph_layout()
        
        # Prepare data for visualization
        nodes_data = []
        for kernel in self.kernels:
            x, y = positions.get(kernel.id, (0, 0))
            nodes_data.append({
                'id': kernel.id,
                'label': kernel.label[:50] + '...' if len(kernel.label) > 50 else kernel.label,
                'x': x,
                'y': y,
                'value': kernel.confidence * 30,  # Size based on confidence
                'title': f"Confidence: {kernel.confidence:.2f}<br>Stability: {kernel.stability:.2f}",
                'color': {
                    'background': f'rgba(150, 150, 250, {kernel.stability})',
                    'border': '#4040A0'
                }
            })
        
        edges_data = []
        for relation in self.relations:
            color = {
                'causation': '#FF6B6B',
                'prevention': '#FFA06B',
                'enablement': '#6BCF7F'
            }.get(relation.relation_type, '#999999')
            
            edges_data.append({
                'from': relation.cause_kernel_id,
                'to': relation.effect_kernel_id,
                'value': relation.confidence * 5,
                'color': color,
                'title': f"{relation.relation_type}<br>Confidence: {relation.confidence:.2f}",
                'arrows': 'to'
            })
        
        # Debug logging as suggested in READ.md
        print(f"DEBUG: nodes_data sample: {nodes_data[:2] if nodes_data else 'EMPTY'}")
        print(f"DEBUG: edges_data sample: {edges_data[:2] if edges_data else 'EMPTY'}")  
        print(f"DEBUG: Node positions: {[(n['x'], n['y']) for n in nodes_data[:3]]}")
        
        # Enhanced HTML template with better styling and controls
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conceptual Cartographer 2.0</title>
            <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style>
                body { 
                    margin: 0; 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #1a1a1a;
                    color: #e0e0e0;
                }
                #container {
                    display: grid;
                    grid-template-columns: 1fr 350px;
                    height: 100vh;
                }
                #network {
                    background: #0a0a0a;
                    border-right: 1px solid #333;
                    width: 100%;
                    height: 100vh;
                    min-height: 500px;
                }
                #sidebar {
                    padding: 20px;
                    overflow-y: auto;
                    background: #1a1a1a;
                }
                h1 {
                    font-size: 24px;
                    margin-bottom: 20px;
                    color: #6B8FFF;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 15px;
                    background: #252525;
                    border-radius: 8px;
                }
                .stat {
                    display: flex;
                    justify-content: space-between;
                    margin: 10px 0;
                    font-size: 14px;
                }
                .stat-value {
                    color: #6BCF7F;
                    font-weight: bold;
                }
                #selected-info {
                    margin-top: 20px;
                    padding: 15px;
                    background: #2a2a3a;
                    border-radius: 8px;
                    min-height: 100px;
                }
                .legend {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-top: 10px;
                }
                .legend-item {
                    display: flex;
                    align-items: center;
                    gap: 5px;
                    font-size: 12px;
                }
                .legend-color {
                    width: 20px;
                    height: 3px;
                }
                button {
                    background: #6B8FFF;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-right: 10px;
                }
                button:hover {
                    background: #5a7ee8;
                }
            </style>
        </head>
        <body>
            <div id="container">
                <div id="network"></div>
                <div id="sidebar">
                    <h1>Conceptual Map Analysis</h1>
                    
                    <div class="section">
                        <h3>Overview</h3>
                        <div class="stat">
                            <span>Concepts Discovered:</span>
                            <span class="stat-value">__CONCEPT_COUNT__</span>
                        </div>
                        <div class="stat">
                            <span>Causal Relations:</span>
                            <span class="stat-value">__RELATION_COUNT__</span>
                        </div>
                        <div class="stat">
                            <span>Semantic Influences:</span>
                            <span class="stat-value">__INFLUENCE_COUNT__</span>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h3>Relation Types</h3>
                        <div class="legend">
                            <div class="legend-item">
                                <div class="legend-color" style="background: #FF6B6B;"></div>
                                <span>Causation</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background: #FFA06B;"></div>
                                <span>Prevention</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background: #6BCF7F;"></div>
                                <span>Enablement</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h3>Controls</h3>
                        <button onclick="resetView()">Reset View</button>
                        <button onclick="togglePhysics()">Toggle Physics</button>
                    </div>
                    
                    <div id="selected-info">
                        <h3>Selected Element</h3>
                        <p>Click on a node or edge to see details</p>
                    </div>
                </div>
            </div>
            
            <script>
                // Data
                const nodesData = __NODES_DATA__;
                const edgesData = __EDGES_DATA__;
                
                // Create network
                const container = document.getElementById('network');
                const data = {
                    nodes: new vis.DataSet(nodesData),
                    edges: new vis.DataSet(edgesData)
                };
                
                const options = {
                    nodes: {
                        shape: 'dot',
                        font: {
                            size: 12,
                            color: '#e0e0e0'
                        },
                        borderWidth: 2
                    },
                    edges: {
                        width: 2,
                        smooth: {
                            type: 'continuous',
                            roundness: 0.5
                        }
                    },
                    physics: {
                        enabled: true,
                        stabilization: {
                            iterations: 200
                        },
                        barnesHut: {
                            gravitationalConstant: -2000,
                            centralGravity: 0.3,
                            springLength: 100,
                            springConstant: 0.04
                        }
                    },
                    interaction: {
                        hover: true,
                        tooltipDelay: 200
                    }
                };
                
                const network = new vis.Network(container, data, options);
                
                // Event handlers
                
                network.once('stabilizationIterationsDone', function() {
                    network.fit(); // Fit all nodes in viewport
                    network.setOptions({ physics: false });
                    physicsEnabled = false;
                });
                
                // Also add an immediate fit in case stabilization doesn't work:
                setTimeout(function() {
                    network.fit();
                }, 1000);
                
                network.on('click', function(params) {
                    const selectedInfo = document.getElementById('selected-info');
                    
                    if (params.nodes.length > 0) {
                        const nodeId = params.nodes[0];
                        const node = nodesData.find(n => n.id === nodeId);
                        selectedInfo.innerHTML = `
                            <h3>Concept: ${node.label}</h3>
                            <div>${node.    .replace('<br>', '<br/>')}</div>
                        `;
                    } else if (params.edges.length > 0) {
                        const edgeId = params.edges[0];
                        const edge = edgesData.find(e => e.id === edgeId);
                        if (edge) {
                            selectedInfo.innerHTML = `
                                <h3>Relation</h3>
                                <div>${edge.title.replace('<br>', '<br/>')}</div>
                            `;
                        }
                    }
                });
                
                // Control functions
                let physicsEnabled = true;
                
                function togglePhysics() {
                    physicsEnabled = !physicsEnabled;
                    network.setOptions({ physics: { enabled: physicsEnabled } });
                }
                
                function resetView() {
                    network.fit();
                }
                
            </script>
        </body>
        </html>
        """
        
        # Fill in the template
        html = html_template.replace('__NODES_DATA__', json.dumps(nodes_data, cls=NumpyEncoder))
        html = html.replace('__EDGES_DATA__', json.dumps(edges_data, cls=NumpyEncoder))
        html = html.replace('__CONCEPT_COUNT__', str(len(self.kernels)))
        html = html.replace('__RELATION_COUNT__', str(len(self.relations)))
        html = html.replace('__INFLUENCE_COUNT__', str(len(self.influences)))
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Visualization saved to {output_file}")
        
        # Open in browser
        webbrowser.open('file://' + os.path.realpath(output_file))


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

class ConceptualCartographer:
    """Main class orchestrating the entire analysis pipeline."""
    
    def __init__(self, config: Config = Config()):
        self.config = config
        self.kernels = None
        self.relations = None
        self.influences = None
    
    def analyze(self, text: str, output_visualization: bool = True) -> Dict:
        """Run complete analysis pipeline with error handling."""
        results = {
            'success': False,
            'kernels': [],
            'relations': [],
            'influences': [],
            'errors': []
        }
        
        try:
            # Step 1: Kernel Discovery
            logger.info("=" * 50)
            logger.info("PHASE 1: Kernel Discovery")
            logger.info("=" * 50)
            
            discoverer = RobustKernelDiscoverer(self.config)
            self.kernels = discoverer.discover(text)
            results['kernels'] = self.kernels
            
            if not self.kernels:
                raise ValueError("No conceptual kernels discovered. Text may be too short or lack coherent themes.")
            
            # Print discovered kernels
            print("\nDiscovered Conceptual Kernels:")
            for kernel in self.kernels:
                print(f"  [{kernel.id}] {kernel.label}")
                print(f"      Confidence: {kernel.confidence:.2f}, Stability: {kernel.stability:.2f}")
            
            # Step 2: Relation Extraction
            logger.info("\n" + "=" * 50)
            logger.info("PHASE 2: Relation Extraction")
            logger.info("=" * 50)
            
            extractor = NeuralRelationExtractor(self.config)
            self.relations = extractor.extract_relations(text, self.kernels)
            results['relations'] = self.relations
            
            # Print extracted relations
            if self.relations:
                print("\nExtracted Causal Relations:")
                kernel_map = {k.id: k.label for k in self.kernels}
                for rel in self.relations[:10]:  # Show top 10
                    cause = kernel_map.get(rel.cause_kernel_id, "Unknown")
                    effect = kernel_map.get(rel.effect_kernel_id, "Unknown")
                    print(f"  {cause} --[{rel.relation_type}]--> {effect}")
                    print(f"      Confidence: {rel.confidence:.2f}")
            
            # Step 3: Influence Analysis
            logger.info("\n" + "=" * 50)
            logger.info("PHASE 3: Semantic Influence Analysis")
            logger.info("=" * 50)
            
            analyzer = StatisticalInfluenceAnalyzer(self.kernels, self.config)
            self.influences = analyzer.analyze_influences(text)
            results['influences'] = self.influences
            
            # Print discovered influences
            if self.influences:
                print("\nStatistically Significant Semantic Influences:")
                kernel_map = {k.id: k.label for k in self.kernels}
                for inf in self.influences[:5]:  # Show top 5
                    influencer = kernel_map.get(inf.influencer_id, "Unknown")
                    p1 = kernel_map.get(inf.affected_pair[0], "Unknown")
                    p2 = kernel_map.get(inf.affected_pair[1], "Unknown")
                    print(f"  {influencer} causes {inf.influence_type} between ({p1}, {p2})")
                    print(f"      Effect size: {inf.effect_size:.2f}, p-value: {inf.p_value:.4f}")
                    print(f"      95% CI: [{inf.confidence_interval[0]:.3f}, {inf.confidence_interval[1]:.3f}]")
            
            # Step 4: Visualization
            if output_visualization:
                logger.info("\n" + "=" * 50)
                logger.info("PHASE 4: Generating Visualization")
                logger.info("=" * 50)
                
                visualizer = EnhancedVisualizer(self.kernels, self.relations, self.influences, text)
                visualizer.generate_interactive_visualization()
            
            results['success'] = True
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            results['errors'].append(str(e))
            raise
        
        return results
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics about the analysis."""
        if not self.kernels:
            return {}
        
        stats = {
            'num_kernels': len(self.kernels),
            'avg_kernel_confidence': np.mean([k.confidence for k in self.kernels]),
            'avg_kernel_stability': np.mean([k.stability for k in self.kernels]),
            'num_relations': len(self.relations) if self.relations else 0,
            'num_influences': len(self.influences) if self.influences else 0
        }
        
        if self.relations:
            relation_types = defaultdict(int)
            for rel in self.relations:
                relation_types[rel.relation_type] += 1
            stats['relation_types'] = dict(relation_types)
        
        if self.influences:
            stats['avg_effect_size'] = np.mean([inf.effect_size for inf in self.influences])
            stats['num_significant_influences'] = sum(1 for inf in self.influences if inf.p_value < 0.01)
        
        return stats


# =============================================================================
# Example Usage and Testing
# =============================================================================

def load_test_text(filename):
    """Load text from various file types"""
    try:
        if filename.endswith('.txt'):
            with open(filename, 'r', encoding='utf-8') as file:
                return file.read()
        elif filename.endswith('.pdf'):
            # You'd need: pip install PyPDF2
            import PyPDF2
            with open(filename, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def test_with_file():
    text = load_test_text('test_document.txt')  # or .pdf
    if text:
        cartographer = ConceptualCartographer()
        results = cartographer.analyze(text, output_visualization=True)
        return results
        
def test_with_example_text():
    """Test the system with a complex example text."""
    
    example_text = """
    The rise of artificial intelligence has fundamentally transformed how we approach problem-solving
    in the modern era. Machine learning algorithms, particularly deep neural networks, have demonstrated
    remarkable capabilities in pattern recognition and prediction tasks. This technological advancement
    has led to breakthroughs in various fields, from medical diagnosis to autonomous vehicles.
    
    However, the rapid development of AI systems has also raised significant ethical concerns. The
    lack of transparency in deep learning models, often referred to as the "black box" problem,
    makes it difficult to understand how these systems arrive at their decisions. This opacity
    can lead to biased outcomes, especially when the training data reflects historical prejudices.
    
    To address these challenges, researchers have developed explainable AI techniques. These methods
    aim to provide insights into the decision-making process of complex models. By making AI systems
    more interpretable, we can better identify and correct biases, ensuring fairer outcomes for all
    users. This transparency also helps build trust between humans and AI systems.
    
    The implementation of explainable AI has shown promising results in critical applications. In
    healthcare, interpretable models allow doctors to understand why a particular diagnosis was
    suggested, enabling them to make more informed decisions. Similarly, in financial services,
    explainable AI helps ensure that loan approval processes are fair and non-discriminatory.
    
    Despite these advances, challenges remain. The trade-off between model complexity and
    interpretability continues to be a fundamental issue. More complex models often achieve
    better performance but at the cost of transparency. Researchers are actively working on
    methods that can maintain high performance while providing meaningful explanations.
    
    The future of AI likely lies in finding the right balance between capability and interpretability.
    As AI systems become more integrated into society, the demand for transparency will only grow.
    This will drive further innovation in explainable AI techniques, ultimately leading to systems
    that are both powerful and trustworthy.
    """
    
    # Run analysis
    cartographer = ConceptualCartographer()
    results = cartographer.analyze(example_text, output_visualization=True)
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    stats = cartographer.get_summary_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze text for conceptual kernels')
    parser.add_argument('filename', nargs='?', help='Text file to analyze')
    parser.add_argument('--output', '-o', help='Output visualization file')
    
    args = parser.parse_args()
    
    if args.filename:
        print(f"Loading text from: {args.filename}")
        
        try:
            with open(args.filename, 'r', encoding='utf-8') as file:
                text = file.read()
            
            cartographer = ConceptualCartographer()
            results = cartographer.analyze(text, output_visualization=True)
            print("Analysis completed successfully!")
            
        except FileNotFoundError:
            print(f"Error: File '{args.filename}' not found.")
        except Exception as e:
            print(f"Analysis failed: {e}")
    else:
        print("Usage: python3 concart.py <filename>")
        print("Example: python3 concart.py test_text.txt")