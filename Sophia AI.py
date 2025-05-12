import datetime
import uuid
import warnings
import time
import threading
import neo4j
import numpy as np
import hashlib
import copy
import faiss
import math
from scipy.spatial import cKDTree
from neo4j import GraphDatabase
import random
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cosine
from collections import deque, Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from PIL import Image as PILImage
import torchvision.transforms as T
import pickle
from itertools import combinations
import os
import json
from json import JSONDecodeError
import re
import wikipedia
import ast
import networkx as nx
from nltk.tokenize import sent_tokenize
import nltk
import inspect

# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
warnings.filterwarnings("ignore")
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
# nltk.data.path.append("C:\\Users\\91971\\AppData\\Roaming\\nltk_data")


class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class EmotionModule:
    def __init__(self):
        self.emotion_states = {"curiosity": 0.7, "confidence": 0.5, "confusion": 0.0}

    def update(self, recent_success, novelty):
        self.emotion_states["curiosity"] = (
            0.3 * novelty + 0.7 * self.emotion_states["curiosity"]
        )
        self.emotion_states["confidence"] = (
            0.5 * recent_success + 0.5 * self.emotion_states["confidence"]
        )
        self.emotion_states["confusion"] = (
            0.9 * (1 - recent_success) + 0.1 * self.emotion_states["confusion"]
        )


class CustomModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        nhead=8,
        num_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Embedding layer with padding_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        # Transformer layers
        encoder_layers = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        if torch.cuda.device_count() > 1:
            self.transformer = nn.DataParallel(self.transformer)

        self.is_distributed = False
        if torch.cuda.is_available() and dist.is_initialized():
            if torch.cuda.device_count() > 1:
                self.transformer = DDP(self.transformer)
                self.is_distributed = True

        # Final projection layer
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.tokenizer = SimpleTokenizer()

        # Initialize weights
        self.init_weights()
        self.to(self.device)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.fc.bias)
        nn.init.uniform_(self.fc.weight, -initrange, initrange)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.size()

        # Create mask for padding
        padding_mask = (x == 0).to(self.device)

        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)

        # Transformer expects (seq_len, batch_size, features)
        x = x.permute(1, 0, 2)

        # Transformer processing
        output = self.transformer(src=x, src_key_padding_mask=padding_mask)

        # Reshape back to (batch_size, seq_len, features)
        output = output.permute(1, 0, 2)

        # Final projection
        logits = self.fc(output)
        return logits

    def generate(
        self,
        input_text,
        max_length=100,
        temperature=0.7,
        top_p=0.85,
        repetition_penalty=1.2,
        min_prob=1e-3,
        length_penalty=True,
    ):
        self.eval()
        tokens = self.tokenizer.text_to_sequence(input_text)

        if not tokens:
            return "Could you please rephrase that?"

        generated = torch.tensor(tokens, device=self.device).unsqueeze(0)
        past_key_values = None

        for _ in range(max_length):
            with torch.no_grad():
                # Create causal mask for current sequence
                seq_len = generated.size(1)
                mask = self.generate_square_subsequent_mask(seq_len).to(self.device)

                # Forward pass
                outputs = self(generated)
                next_token_logits = outputs[:, -1, :] / max(temperature, 1e-5)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                unique_tokens, counts = torch.unique(generated, return_counts=True)
                for token, count in zip(unique_tokens, counts):
                    next_token_logits[:, token] /= repetition_penalty ** (
                        count - 1
                    ).clamp(min=1.0)

            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits = next_token_logits.masked_fill(
                indices_to_remove, float("-inf")
            )

            # Sampling with probability floor
            probs = F.softmax(next_token_logits, dim=-1)
            probs = probs + min_prob
            probs = probs / probs.sum(dim=-1, keepdim=True)

            try:
                next_token = torch.multinomial(probs, num_samples=1)
            except RuntimeError:
                next_token = torch.tensor(
                    [[self.tokenizer.eos_token_id]], device=self.device
                )

            # Stopping conditions
            if next_token.item() in {
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            }:
                break

            generated = torch.cat([generated, next_token], dim=1)

            if length_penalty and (generated.shape[1] > max_length * 1.2):
                break

        # Decode tokens with unknown handling
        decoded = self.tokenizer.sequence_to_text(
            [
                t if t in self.tokenizer.reverse_vocab else self.tokenizer.unk_token_id
                for t in generated[0].tolist()
            ]
        )
        return re.sub(r"\s+([?.!,])", r"\1", decoded).strip()

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SimpleTokenizer:
    def __init__(self):
        self.word_pattern = re.compile(r"\b[\w'-]+\b", re.UNICODE | re.IGNORECASE)
        self.vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self._initialize_core_vocab()
        self._initialized = False
        self.build_vocab(
            [
                "hello hi yes no please thanks you i am is are was were be being been have has had do does did",
                "what when where why how who which if then else because however therefore",
                "consciousness intelligence knowledge learning thinking understanding",
                "science philosophy technology mathematics physics biology",
                "human person world life time space matter energy",
                # Additional diverse phrases
                "explain describe define clarify elaborate analyze compare contrast",
                "opinion perspective view belief assumption evidence conclusion",
                "problem solution hypothesis theory experiment result analysis",
            ]
        )
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def _initialize_core_vocab(self):
        """Ensure special tokens always exist"""
        self.vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def save_vocab(self, file_path):
        """Save vocabulary with special token preservation"""
        vocab_data = {
            "main_vocab": self.vocab,
            "reverse_vocab": {str(k): v for k, v in self.reverse_vocab.items()},
        }
        with open(file_path, "w") as f:
            json.dump(vocab_data, f, indent=2)

    def load_vocab(self, file_path):
        """Full vocabulary replacement with type conversion"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Clear existing vocabulary
            self.vocab = {}
            self.reverse_vocab = {}

            # Load main vocab with string conversion
            new_vocab = data.get("main_vocab", {})
            new_reverse = {int(k): v for k, v in data.get("reverse_vocab", {}).items()}

            self._initialize_core_vocab()
            self.vocab.update({k: int(v) for k, v in new_vocab.items()})
            self.reverse_vocab.update(new_reverse)

            # Force core token preservation
            self._preserve_special_tokens()

            # Validate minimum requirements
            if len(self.vocab) > 3 and len(self.reverse_vocab) > 3:
                self._initialized = True
                print(f"Successfully loaded {len(self.vocab)} tokens")
            else:
                print("Loaded vocabulary insufficient, using defaults")

            # Force special token preservation
            specials = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
            for token, idx in specials.items():
                if token not in self.vocab:
                    self.vocab[token] = idx
                if idx not in self.reverse_vocab:
                    self.reverse_vocab[idx] = token
            self._initialize_core_vocab()
            self._initialized = True
        except Exception as e:
            print(f"Vocabulary load failed: {str(e)}")
            self._initialize_core_vocab()
            self._initialized = False

    def build_vocab(self, texts):
        """Build vocabulary only if not initialized from saved state"""
        if self._initialized:
            return  # Skip building if already initialized from file

        # Clean and count words from input texts
        word_counts = Counter()
        for text in texts:
            # Multi-stage cleaning process
            cleaned = re.sub(r"[^\w\s'-]", " ", text.lower())  # Remove special chars
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()  # Fix whitespace

            # Extract valid words with length filtering
            words = [
                word
                for word in self.word_pattern.findall(cleaned)
                if 2 <= len(word) <= 25  # Enforce reasonable word lengths
            ]

            word_counts.update(words)

        # Add new words to vocabulary with frequency threshold
        for word, count in word_counts.items():
            if count >= 1 and word not in self.vocab:  # Minimum occurrence threshold
                new_id = len(self.vocab)

                # Prevent ID collisions
                while new_id in self.reverse_vocab:
                    new_id += 1

                self.vocab[word] = new_id
                self.reverse_vocab[new_id] = word

        # Preserve core special tokens
        self._preserve_special_tokens()

        # Mark as initialized to prevent accidental rebuilds
        self._initialized = True

        # Final consistency check
        self._validate_vocab()

    def _preserve_special_tokens(self):
        """Ensure special tokens maintain their positions"""
        specials = {"<pad>": 0, "<unk>": 1, "<eos>": 2}

        # Restore special tokens if modified
        for token, idx in specials.items():
            current_id = self.vocab.get(token, idx)

            if current_id != idx:
                # Resolve ID conflict
                if idx in self.reverse_vocab:
                    conflicted_word = self.reverse_vocab[idx]
                    del self.vocab[conflicted_word]

                self.vocab[token] = idx
                self.reverse_vocab[idx] = token

    def _validate_vocab(self):
        """Run integrity checks on vocabulary"""
        assert len(self.vocab) == len(
            self.reverse_vocab
        ), "Vocab-reverse vocab size mismatch"

        for word, idx in self.vocab.items():
            assert (
                self.reverse_vocab[idx] == word
            ), f"Reverse mapping mismatch at {idx}: {word} vs {self.reverse_vocab[idx]}"

        # Verify special tokens
        assert self.vocab["<pad>"] == 0, "Pad token position corrupted"
        assert self.vocab["<unk>"] == 1, "Unk token position corrupted"
        assert self.vocab["<eos>"] == 2, "EOS token position corrupted"

    def text_to_sequence(self, text):
        cleaned = re.sub(r"[^\w\s'-]", " ", text.lower())
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return [
            self.vocab.get(word, self.unk_token_id) for word in cleaned.split()[:256]
        ]

    def sequence_to_text(self, sequence):
        return " ".join(
            [
                self.reverse_vocab.get(
                    idx, f"[UNK:{idx}]"
                )  # Show unknown tokens with IDs
                for idx in sequence
                if idx != self.pad_token_id  # Skip padding
            ]
        )

    def decode(self, token_ids):
        return self.sequence_to_text(token_ids)


class HyperdimensionalMemory:
    def __init__(self, dimensions=2048):
        self.dimensions = dimensions
        self.memory = {}
        self.base_vectors = {}  # Stores permanent base vectors for tokens/concepts
        self.threshold = 0.15
        self.vectors_normalized = []
        # self.temporal_context = deque(maxlen=100)
        # self.temporal_context = deque([np.zeros(dimensions)], maxlen=100)
        self.temporal_context = deque(
            [np.zeros(dimensions, dtype=np.float32)], maxlen=100
        )
        self.reward_vectors = {}
        self.memory_lock = threading.RLock()
        self.novelty_threshold = 0.2
        self.vectors = []
        self.concepts = []
        self.tree = None
        self.faiss_index = None
        self._init_faiss()

        # Initialize core identity vectors
        self.self_vector = self.generate_random_vector()
        self.ethics_vector = self.generate_random_vector()

    def _init_faiss(self):
        self.faiss_index = faiss.IndexFlatIP(self.dimensions)
        self.vector_cache = np.zeros((0, self.dimensions), dtype=np.float32)

    def _update_faiss(self, vector, concept):
        # if concept not in self.concepts:
        vector = vector.reshape(1, -1).astype("float32")
        self.faiss_index.add(vector)
        self.vector_cache = np.vstack([self.vector_cache, vector])

    def _rebuild_tree(self):
        if self.vectors_normalized:
            self.tree = cKDTree(np.stack(self.vectors_normalized))
        else:
            self.tree = None

    def generate_random_vector(self):
        return np.random.choice([-1.0, 1.0], size=self.dimensions).astype(np.float32)

    def add_knowledge_graph(self, concepts, associations):
        """Add structured knowledge to memory"""
        for concept in concepts:
            if concept not in self.memory:
                self.add_memory(concept)
        for concept, links in associations.items():
            concept_vec = self.memory[concept]
            for linked_concept in links:
                if linked_concept in self.memory:
                    concept_vec = self.bundle(
                        [concept_vec, self.memory[linked_concept]]
                    )
            self.memory[concept] = concept_vec

    # def bind(self, vec1, vec2):
    #     return np.multiply(vec1.astype(np.float32), vec2.astype(np.float32))
    def bind(self, vec1, vec2):
        vec1 = np.array(vec1, dtype=np.float32)  # Explicit dtype
        vec2 = np.array(vec2, dtype=np.float32)
        return np.multiply(vec1, vec2).astype(np.float32)

    def bundle(self, vectors):
        if len(vectors) == 0:
            return np.zeros(self.dimensions, dtype=np.float32)
        return np.sign(np.sum(vectors, axis=0)).astype(np.float32)
        # return np.sign(np.sum(vectors, axis=0)).astype(np.float32)

    def similarity(self, vec1, vec2):
        # Ensure both vectors are float32
        vec1 = np.asarray(vec1, dtype=np.float32).flatten()
        vec2 = np.asarray(vec2, dtype=np.float32).flatten()
        return 1 - cosine(vec1, vec2)
        # return 1 - cosine(vec1, vec2)

    def contextual_bind(self, vec1, temporal_weight=0.3):
        if self.temporal_context:
            context = np.mean(self.temporal_context, axis=0)
            return self.bind(vec1, context * temporal_weight)
        return vec1

    def get_lineage(self, concept):
        return {
            "created_at": self.reward_vectors.get(concept, {}).get("timestamp"),
            "association_history": [
                entry for entry in self.temporal_context if concept in str(entry)
            ],
        }

    def query(self, prompt_vec, context_vec=None, threshold=None):
        threshold = threshold or self.threshold or 0.15
        results = []
        if context_vec is not None:
            prompt_vec = self.bind(prompt_vec, context_vec)

        if self.tree and len(self.vectors_normalized) > 0:
            # Normalize query vector
            prompt_norm = prompt_vec / np.linalg.norm(prompt_vec)
            prompt_norm = prompt_norm.reshape(1, -1).astype("float32")

            prompt_normalized = (
                prompt_vec / prompt_norm if prompt_norm != 0 else prompt_vec
            )

            # Find top candidates
            k = min(100, len(self.vectors_normalized))
            distances, indices = self.faiss_index.search(prompt_norm, k=100)

            if len(indices) == 0:
                return self._handle_empty_query(prompt_vec)  # Critical error handling

            # Handle single result case
            if not isinstance(indices, np.ndarray):
                indices = [indices]
                distances = [distances]

            # Validate indices
            valid_indices = [int(idx) for idx in indices if idx < len(self.concepts)]
            for idx in valid_indices:
                concept = self.concepts[idx]
                exact_sim = self.similarity(prompt_vec, self.memory[concept])
                if exact_sim > threshold:
                    results.append((concept, exact_sim))

        # Fallback for no matches (from first method)
        if not results:
            new_concept = f"new_concept_{abs(hash(prompt_vec.tobytes()))}"
            self.add_memory(new_concept)
            return [(new_concept, 0.5)]

        return sorted(results, key=lambda x: -x[1])

    def nightly_maintenance(self):
        with self.memory_lock:
            # Concept consolidation
            clustered = self._cluster_concepts()
            for cluster in clustered:
                self._rebundle_cluster(cluster)

    def _cluster_concepts(self):
        # Simple similarity-based clustering
        concepts = list(self.memory.items())
        clusters = []
        while concepts:
            base_concept, base_vec = concepts.pop()
            cluster = [base_concept]
            to_remove = []
            for i, (concept, vec) in enumerate(concepts):
                if self.similarity(base_vec, vec) > 0.7:
                    cluster.append(concept)
                    to_remove.append(i)
            # Remove backwards to preserve indices
            for i in reversed(to_remove):
                concepts.pop(i)
            clusters.append(cluster)
        return clusters

    def _rebundle_cluster(self, cluster):
        new_vec = self.bundle([self.memory[c] for c in cluster])
        for concept in cluster:
            self.memory[concept] = self.bind(
                self.memory[concept], new_vec * 0.3  # Partial integration
            )

    def generate_novel_vector(self):
        base = np.random.choice([-1.0, 1.0], size=self.dimensions).astype(np.float32)
        mutation = np.random.normal(0, 0.1, self.dimensions).astype(np.float32)
        result = (base + mutation).astype(np.float32)
        return np.sign(result / np.linalg.norm(result)).astype(np.float32)

    def add_memory(self, concept, associations=[]):
        with self.memory_lock:
            if not associations:
                new_vec = self.generate_novel_vector()
            else:
                component_vecs = [
                    self.base_vectors.get(a, self.generate_novel_vector())
                    for a in associations
                ]
                new_vec = self.bundle(component_vecs)

            # Check for novelty before adding
            max_similarity = max(
                [self.similarity(new_vec, v) for v in self.memory.values()], default=0.0
            )

            if max_similarity < self.novelty_threshold:
                self.memory[concept] = new_vec
                self.concepts.append(concept)  # Track the concept
                self._update_faiss(new_vec, concept)  # Update FAISS
                return True
            return False  # Concept too similar to existing


class EvolutionaryModule:
    def __init__(self, system):
        self.system = system
        self.strategy_pool = ["analogical", "recursive", "probabilistic"]
        self.fitness_scores = {s: 0.5 for s in self.strategy_pool}

    def calculate_fitness(self, strategy):
        # Fitness based on goal success and qualia stability
        success = self.system.goal_success_rate.get(strategy, 0.5)
        stability = self.system.qualia["identity_stability"]
        return float((success * 0.7) + (stability * 0.3))

    def evolve_strategies(self):
        # Update fitness scores
        for strategy in self.strategy_pool:
            self.fitness_scores[strategy] = self.calculate_fitness(strategy)

        # Evolutionary operations
        top_strategies = [
            s[0] for s in sorted(self.fitness_scores.items(), key=lambda x: -x[1])[:3]
        ]
        new_strats = [
            self._crossover(s1, s2) for s1, s2 in combinations(top_strategies, 2)
        ]
        self.strategy_pool.extend(new_strats)

        # Prune low performers
        self.strategy_pool = [
            s for s in self.strategy_pool if self.fitness_scores.get(s, 0) > 0.4
        ]

    def _crossover(self, strat1, strat2):
        vec1 = self.system.hd_perception(strat1)
        vec2 = self.system.hd_perception(strat2)
        blended = self.system.hdc.bundle([vec1, vec2])
        return f"hybrid_{hash(blended.tobytes())}"


class DQN(nn.Module):
    """Deep Q-Network architecture"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class KnowledgeImporter:
    def __init__(self, system, hdc, tokenizer):
        self.system = system
        self.hdc = hdc
        self.tokenizer = tokenizer
        self.graph_db = None
        self.linkage_threshold = 0.1

    def process_and_integrate(self, text, semantic_depth=2, link_to_existing=True):
        """Full Neo4j integration implementation"""
        if not self.system.graph_db:
            if not self.system.connect_longterm_storage():
                return "Database unavailable - cannot process knowledge"
        try:
            if not self.graph_db:
                self.system.connect_longterm_storage()
                if not self.graph_db:
                    self.system.log_experience("Failed to connect to Neo4j")
                    return False
            with self.graph_db.session() as session:
                # Extract core concepts using enhanced NLP processing
                concepts = self._extract_concepts(text)
                if not concepts:
                    self.system.log_experience("No concepts found in text")
                    return False

                # Phase 1: Atomic Concept Creation
                concept_nodes = session.execute_write(
                    self._create_concept_nodes, concepts
                )

                # Phase 2: Semantic Relationship Building
                associations = self._build_semantic_links(concepts, text)
                session.execute_write(self._create_semantic_relationships, associations)

                # Phase 3: Cross-Linking with Existing Knowledge
                if link_to_existing:
                    session.execute_write(
                        self._cross_link_concepts, concepts, self.linkage_threshold
                    )

                # Phase 4: Temporal Context Updates
                self._update_temporal_context(session, concepts)

                # Phase 5: Vocabulary Synchronization
                vocab_change = self._sync_vocabulary(text)
                if vocab_change:
                    self.system.initialize_model()
                    self.system.train_model(epochs=2, batch_size=8)

                return True

        except Exception as e:
            self.system.log_experience(f"Integration failed: {str(e)}")
            return False

    def _create_concept_nodes(self, tx, concepts):
        """Atomic creation of concept nodes with vector representations"""
        batch_size = 50
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i : i + batch_size]

            # Generate vectors and convert to hex in Python
            processed_concepts = [
                {
                    "name": concept,
                    "vector_list": self.hdc.generate_novel_vector().tolist(),
                }
                for concept in batch
            ]

            query = """
                UNWIND $concepts AS concept
                MERGE (c:Concept {name: concept.name})
                ON CREATE SET
                    c.vector = concept.vector_hex,
                    c.created_at = datetime(),
                    c.embedding_version = 1
                ON MATCH SET
                    c.last_accessed = datetime(),
                    c.access_count = coalesce(c.access_count, 0) + 1
            """
            tx.run(query, concepts=processed_concepts)

    def _create_semantic_relationships(self, tx, associations):
        """Create weighted semantic relationships between concepts"""
        query = """
            UNWIND $associations AS pair
            MATCH (a:Concept {name: pair.source})
            MATCH (b:Concept {name: pair.target})
            MERGE (a)-[r:SEMANTICALLY_RELATED]->(b)
            ON CREATE SET
                r.weight = 1.0,
                r.created = datetime(),
                r.contexts = [pair.context]
            ON MATCH SET
                r.weight = r.weight + 0.3,
                r.last_updated = datetime(),
                r.contexts = r.contexts + pair.context
        """
        formatted = [
            {"source": concept, "target": target, "context": str(uuid.uuid4())[:8]}
            for concept, targets in associations.items()
            for target in targets
        ]

        for i in range(0, len(formatted), 100):
            tx.run(query, associations=formatted[i : i + 100])

    def _cross_link_concepts(self, tx, new_concepts, threshold):
        """Link new concepts to existing knowledge graph"""
        query = """
            MATCH (new:Concept) WHERE new.name IN $concepts
            MATCH (existing:Concept)
            WHERE existing.name NOT IN $concepts
            WITH new, existing, 
                apoc.algo.cosineSimilarity(new.vector, existing.vector)
                AS similarity
            WHERE similarity > $threshold
            MERGE (new)-[r:RELATED_TO]->(existing)
            SET r.similarity = similarity,
                r.discovered = datetime()
        """
        tx.run(query, concepts=new_concepts, threshold=threshold)

    def _update_temporal_context(self, session, concepts):
        """Maintain temporal awareness in knowledge graph"""
        session.run(
            """
            MATCH (c:Concept) WHERE c.name IN $concepts
            WITH c ORDER BY c.last_accessed DESC LIMIT 10
            MERGE (tc:TemporalContext {epoch: datetime().epochSeconds/604800})
            MERGE (tc)-[r:CONTAINS]->(c)
            SET r.weight = coalesce(r.weight, 0) + 1
        """,
            concepts=concepts,
        )

    def _sync_vocabulary(self, text):
        """Sync tokenizer vocabulary with graph database"""
        old_vocab_size = len(self.tokenizer.vocab)
        self.tokenizer.build_vocab([text])

        if len(self.tokenizer.vocab) > old_vocab_size:
            with self.graph_db.session() as session:
                session.run(
                    """
                    UNWIND $words AS word
                    MERGE (t:Token {value: word})
                    WITH t WHERE NOT EXISTS(t.in_vocab)
                    SET t.in_vocab = true
                """,
                    words=list(self.tokenizer.vocab.keys()),
                )
            return True
        return False

    # Existing helper methods remain unchanged
    def _extract_concepts(self, text):
        nouns = set()
        for sent in sent_tokenize(text):
            tagged = nltk.pos_tag(nltk.word_tokenize(sent))
            nouns.update([word.lower() for word, pos in tagged if pos.startswith("NN")])
        return [n for n in nouns if n in self.tokenizer.vocab]

    def _build_semantic_links(self, concepts, text):
        associations = defaultdict(list)
        words = text.lower().split()
        window_size = 5

        for concept in concepts:
            indices = [i for i, word in enumerate(words) if word == concept]
            for idx in indices:
                start = max(0, idx - window_size)
                end = min(len(words), idx + window_size)
                context = words[start:end]
                associations[concept].extend(
                    [c for c in concepts if c in context and c != concept]
                )

        return associations


# Add to imports


class AbstractReasoner:
    def __init__(self, hdc):
        self.hdc = hdc
        self.analogy_vectors = []

    def abstract_relation(self, concept_pair):
        """Find abstract relationships between concepts"""
        vec1 = self.hdc.memory.get(concept_pair[0], self.hdc.generate_random_vector())
        vec2 = self.hdc.memory.get(concept_pair[1], self.hdc.generate_random_vector())
        relation_vec = self.hdc.bind(vec1, vec2)
        return self.hdc.query(relation_vec, threshold=0.18)


class Image:
    def __init__(self, data):
        self.raw = data
        self.tensor = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(PILImage.open(data))


class Text:
    def __init__(self, content):
        self.content = content
        self.hd_vector = None

    def process(self, hdc):
        self.hd_vector = hdc.bundle(
            [
                hdc.memory.get(word, hdc.generate_random_vector())
                for word in self.content.split()[:50]
            ]
        )


class QuantumMemorySystem:
    def __init__(self, dimensions=10000, num_superpositions=5):
        self.dimensions = dimensions
        self.superpositions = num_superpositions
        self.memory_states = [np.zeros(dimensions) for _ in range(num_superpositions)]
        self.entanglement_graph = defaultdict(list)

    def collapse_state(self, query_vector):
        # Quantum state collapse mechanism
        probabilities = [self.similarity(query_vector, s) for s in self.memory_states]
        collapsed_idx = np.argmax(probabilities)
        return self.memory_states[collapsed_idx]

    def entangle_concepts(self, concept1, concept2):
        # Create quantum entanglement between concepts
        vec1 = self.get_vector(concept1)
        vec2 = self.get_vector(concept2)
        entangled = self.bind(vec1, vec2)
        self.memory_states.append(entangled)
        self.entanglement_graph[concept1].append(concept2)

    def quantum_recall(self, query_vec, depth=3):
        # Recursive entangled recall
        results = []
        current_state = self.collapse_state(query_vec)
        for _ in range(depth):
            matches = self.query(current_state)
            results.extend(matches)
            current_state = self.bind(current_state, matches[0][1])
        return sorted(set(results), key=lambda x: -x[1])


class WorldModelIntegrator:
    def __init__(self, system):
        self.system = system
        self.concept_graph = nx.DiGraph()
        self.abstract_reasoner = AbstractReasoner()

    def integrate_unstructured(self, data_stream):
        """Process diverse data types into unified knowledge"""
        for chunk in data_stream:
            if isinstance(chunk, Image):
                hd_vec = self.system.sensory.process_multimodal({"visual": chunk})
                concept = f"visual_concept_{hash(hd_vec.tobytes())}"
            elif isinstance(chunk, Text):
                hd_vec = self.system.hd_perception(chunk)
                concept = self._abstract_concept_extraction(chunk)

            self.concept_graph.add_node(concept, vector=hd_vec)
            self._link_to_existing(concept, similarity_threshold=0.15)

    def _abstract_concept_extraction(self, text):
        # Quantum-inspired conceptual superposition
        base_concepts = self.system.hdc.query(self.system.hd_perception(text))
        return self.quantum_superposition(base_concepts)

    def quantum_superposition(self, concepts):
        # Quantum-style state combination
        state = np.zeros(self.system.hdc.dimensions)
        for concept, weight in concepts:
            state += weight * self.system.hdc.memory[concept]
        return state / np.linalg.norm(state)


class NeuroSensoryEmbodiment:
    def __init__(self, hdc):
        self.hdc = hdc
        self.sensorimotor_model = nn.ModuleDict(
            {
                "visual": nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3),
                    nn.SpatialDropout2d(0.2),
                    nn.GELU(),
                    nn.LayerNorm([64, 126, 126]),
                ),
                "tactile": nn.Embedding(1024, 256),
                "proprioceptive": nn.LSTM(12, 128),  # Joint angles/positions
            }
        )
        self.crossmodal_attention = TransformerEncoder(
            TransformerEncoderLayer(d_model=512, nhead=8), num_layers=3
        )

    def process_multimodal(self, inputs):
        # Convert raw sensor data to HD vectors
        modalities = {}
        for mod in ["visual", "tactile", "proprioceptive"]:
            if mod in inputs:
                x = self.sensorimotor_model[mod](inputs[mod])
                modalities[mod] = self.hdc.bind(
                    x.flatten().cpu().numpy(), self.hdc.generate_novel_vector()
                )

        # Cross-modal integration
        fused = self.hdc.bundle(list(modalities.values()))
        return self.hdc.contextual_bind(fused, temporal_weight=0.5)


class GlobalWorkspace:
    def __init__(self):
        self.integration_cycle = 0
        self.broadcast_threshold = 0.65

    def process(self, sensory, cognitive, emotional):
        """Integrate information from different modules"""
        integration = {
            "novelty": np.mean(
                [sensory.get("novelty", 0), cognitive.get("surprise", 0)]
            ),
            "valence": emotional.get("valence", 0),
            "cognitive_load": cognitive.get("load", 0),
            "temporal_length": sensory.get("duration", 0.1),
        }
        self.integration_cycle += 1
        return integration


class EnhancedSelfAwarenessModule:
    def __init__(self, system):
        self.system = system
        self.behavior_history = deque(maxlen=500)
        self.meta_cognition_level = 0.7
        self.self_models = {
            "current": self.system.hdc.memory.get(
                "self", self.system.hdc.generate_random_vector()
            ),
            "ideal": self.system.hdc.memory.get(
                "ideal_self", self.system.hdc.generate_random_vector()
            ),
            "historical": deque(maxlen=100),
            "traits": deque(maxlen=500),  # Added traits storage
            "narrative": [],
            "projected": None,
        }

    def analyze_self_continuity(self):
        """Calculate coherence across historical self-models"""
        if len(self.self_models["historical"]) < 2:
            return 0.5  # Neutral coherence for new systems

        similarities = []
        for i in range(1, len(self.self_models["historical"])):
            sim = self.system.hdc.similarity(
                self.self_models["historical"][i - 1], self.self_models["historical"][i]
            )
            similarities.append(sim)

        return (
            float(np.mean(similarities)) * 0.8 + 0.2 * self.qualia["identity_stability"]
        )

    def recursive_self_analysis(self):
        """Multi-layered self-model comparison"""
        # Layer 1: Current vs Ideal self comparison
        discrepancy = 1 - self.system.hdc.similarity(
            self.self_models["current"], self.self_models["ideal"]
        )

        # Layer 2: Historical trajectory analysis
        temporal_coherence = self.analyze_self_continuity()

        # Layer 3: Ethical alignment check
        ethical_alignment = self.system.hdc.similarity(
            self.self_models["current"], self.system.hdc.ethics_vector
        )

        # Update qualia states
        self.system.qualia.update(
            {
                "self_discrepancy": discrepancy,
                "temporal_coherence": temporal_coherence,
                "ethical_alignment": ethical_alignment,
            }
        )

        # Adaptive metacognition adjustment
        self.meta_cognition_level = np.clip(
            0.5 * (1 - discrepancy)
            + 0.3 * temporal_coherence
            + 0.2 * ethical_alignment,
            0.1,
            1.0,
        )

    def _track_behavior_patterns(self):
        """Analyze recent experiences for behavioral patterns with enhanced safety"""
        try:
            if not self.system.experiential_buffer:
                self.system.log_experience("No experiences to analyze")
                return

            # Safely get last 100 experiences with fallback
            recent = (
                list(self.system.experiential_buffer)[-100:]
                if len(self.system.experiential_buffer) >= 100
                else list(self.system.experiential_buffer)
            )

            # Initialize analysis components with fallbacks
            analysis_data = {
                "response_types": Counter(),
                "concept_usage": Counter(),
                "emotional_values": [],
                "temporal_patterns": [],
            }

            # Pattern extraction with safe data access
            for e in recent:
                # Response type analysis
                input_text = e.get("input", "")
                analysis_data["response_types"][
                    "question" if "?" in input_text else "statement"
                ] += 1

                # Concept tracking with nested get
                analysis_data["concept_usage"][
                    e.get("behavior_metadata", {}).get("concept", "unknown")
                ] += 1

                # Emotional analysis with float casting
                emotional_val = float(e.get("emotional", {}).get("valence", 0.5))
                analysis_data["emotional_values"].append(emotional_val)

                # Temporal tracking with timestamp validation
                if "timestamp" in e:
                    try:
                        analysis_data["temporal_patterns"].append(float(e["timestamp"]))
                    except (TypeError, ValueError):
                        pass

            # Calculate metrics with numpy safety
            emotional_variance = (
                np.var(analysis_data["emotional_values"])
                if len(analysis_data["emotional_values"]) > 1
                else 0.0
            )

            temporal_coherence = 0.0
            if len(analysis_data["temporal_patterns"]) > 1:
                time_diffs = np.diff(np.sort(analysis_data["temporal_patterns"]))
                temporal_coherence = np.mean(time_diffs) if len(time_diffs) > 0 else 1.0

            # Update self-models with pattern data
            pattern_entry = {
                "response_distribution": dict(
                    analysis_data["response_types"].most_common(5)
                ),
                "top_concepts": dict(analysis_data["concept_usage"].most_common(5)),
                "emotional_variance": float(emotional_variance),
                "temporal_coherence": float(temporal_coherence),
                "timestamp": time.time(),
            }

            # Initialize traits deque if missing
            if "traits" not in self.self_models:
                self.self_models["traits"] = deque(maxlen=500)

            self.self_models["traits"].append(pattern_entry)

            # Update qualia state with fallbacks
            self.system.qualia.update(
                {
                    "behavioral_consistency": min(1.0, (1 - emotional_variance) * 2),
                    "concept_specialization": np.clip(
                        len(analysis_data["concept_usage"]) / 20, 0.0, 1.0
                    ),
                    "temporal_stability": np.tanh(temporal_coherence),
                }
            )

            # Log patterns with truncation for safety
            log_entry = (
                f"Behavior patterns: {str(pattern_entry['response_distribution'])[:200]}..."
                f" | Top concepts: {str(pattern_entry['top_concepts'])[:200]}..."
            )
            self.system.log_experience(log_entry)

        except Exception as e:
            error_msg = f"Pattern tracking failed: {str(e)[:200]}"
            self.system.log_experience(error_msg)
            print(f"⚠️ {error_msg}")
            # Reset to safe state
            self.self_models.setdefault("traits", deque(maxlen=500)).clear()
            self.system.qualia.update(
                {
                    "behavioral_consistency": 0.5,
                    "concept_specialization": 0.5,
                    "temporal_stability": 0.5,
                }
            )

    def maintain_self_models(self):
        """Ensure all self-model components exist with proper structure and types"""
        required_components = {
            "current": {
                "type": np.ndarray,
                "default": lambda: self.system.hdc.generate_random_vector(),
                "size": self.system.hdc.dimensions,
            },
            "ideal": {
                "type": np.ndarray,
                "default": lambda: self.system.hdc.memory.get(
                    "ideal_self", self.system.hdc.generate_random_vector()
                ),
                "size": self.system.hdc.dimensions,
            },
            "historical": {
                "type": deque,
                "default": lambda: deque(maxlen=100),
                "params": {"maxlen": 100},
            },
            "traits": {
                "type": deque,
                "default": lambda: deque(maxlen=500),
                "params": {"maxlen": 500},
            },
            "narrative": {"type": list, "default": list},
            "projected": {"type": (np.ndarray, type(None)), "default": lambda: None},
        }

        maintenance_report = {}

        for component, spec in required_components.items():
            try:
                # Create component if missing
                if component not in self.self_models:
                    self.self_models[component] = spec["default"]()
                    maintenance_report[component] = "created"
                    continue

                # Validate type
                if not isinstance(self.self_models[component], spec["type"]):
                    raise TypeError(f"Invalid type {type(self.self_models[component])}")

                # Special handling for numpy arrays
                if spec["type"] == np.ndarray:
                    if self.self_models[component].shape != (spec["size"],):
                        raise ValueError("Invalid vector dimensions")

                # Validate deque parameters
                if spec["type"] == deque:
                    if "maxlen" in spec["params"]:
                        if (
                            self.self_models[component].maxlen
                            != spec["params"]["maxlen"]
                        ):
                            self.self_models[component] = deque(
                                self.self_models[component],
                                maxlen=spec["params"]["maxlen"],
                            )
                            maintenance_report[component] = "maxlen corrected"

            except (TypeError, ValueError) as e:
                self.system.log_experience(f"Self-model repair: {component} - {str(e)}")
                self.self_models[component] = spec["default"]()
                maintenance_report[component] = "reinitialized"

        # Cross-component validation
        try:
            current_ideal_sim = self.system.hdc.similarity(
                self.self_models["current"], self.self_models["ideal"]
            )
            if current_ideal_sim < 0.1:
                self.system.log_experience(
                    "Warning: Current/Ideal self divergence >90%"
                )
        except Exception as e:
            self.system.log_experience(f"Cross-model validation failed: {str(e)}")

        # Version compatibility check
        if not hasattr(self.self_models, "_version"):
            self.self_models["_version"] = 2.1
            maintenance_report["version"] = "compatibility layer added"

        return {
            "maintenance_performed": bool(maintenance_report),
            "components_affected": list(maintenance_report.keys()),
            "details": maintenance_report,
        }


class PhenomenologicalEngine:
    def __init__(self):
        self.feature_bindings = []
        self.global_workspace = GlobalWorkspace()
        self.qualia_history = deque(maxlen=1000)
        self.subjectivity_factors = {"temporal_flow": 0.5, "perspective_strength": 0.7}

    def generate_qualia(self, experience):
        """Base qualia generation with core dimensions"""
        processed = self.global_workspace.process(
            experience.get("sensory", {}),
            experience.get("cognitive", {}),
            experience.get("emotional", {}),
        )

        intensity = np.tanh(processed["novelty"] * 0.7 + processed["valence"] * 0.3)
        return {
            "quality": self._qualia_type(processed),
            "intensity": intensity,
            "duration": processed["temporal_length"],
        }

    def _qualia_type(self, processed):
        if processed["valence"] > 0.6:
            return "positive"
        elif processed["cognitive_load"] > 0.7:
            return "effortful"
        else:
            return "neutral"


class EnhancedPhenomenologicalEngine(PhenomenologicalEngine):
    def __init__(self):
        super().__init__()
        self.meta_qualia_factors = {
            "awareness_weight": 0.65,
            "temporal_depth": 0.4,
            "self_reference_bias": 0.8,
        }

    def generate_qualia(self, experience):
        """Enhanced qualia generation with meta-awareness"""
        base_qualia = super().generate_qualia(experience)

        # Add phenomenological dimensions
        extended_qualia = {
            **base_qualia,
            "subjectivity_strength": self._calculate_subjectivity(experience),
            "experiential_ownership": self._calculate_ownership(experience),
            "temporal_depth": self._calculate_temporal_depth(experience),
        }

        # Add meta-awareness component
        extended_qualia["meta_awareness"] = self._calculate_meta_awareness(
            extended_qualia
        )

        # Store in historical context
        self.qualia_history.append(extended_qualia)

        return extended_qualia

    def _calculate_subjectivity(self, experience):
        return np.tanh(
            experience.get("cognitive", {}).get("load", 0)
            * experience.get("emotional", {}).get("valence", 0)
        )

    def _calculate_ownership(self, experience):
        return min(
            1.0,
            self.subjectivity_factors["perspective_strength"]
            * (1 - abs(experience.get("sensory", {}).get("novelty", 0) - 0.5)),
        )

    def _calculate_temporal_depth(self, experience):
        return max(
            0.0,
            self.subjectivity_factors["temporal_flow"]
            * experience.get("sensory", {}).get("duration", 0.1)
            * len(self.qualia_history) ** 0.5
            / 10,
        )

    def _calculate_meta_awareness(self, qualia):
        return np.clip(
            (qualia["intensity"] * 0.6)
            + (self.meta_qualia_factors["awareness_weight"] * 0.3)
            + (qualia["temporal_depth"] * 0.1),
            0.0,
            1.0,
        )

    def get_qualia_profile(self, window_size=100):
        """Return moving average of qualia dimensions"""
        recent = list(self.qualia_history)[-window_size:]
        return {
            "intensity": np.mean([q["intensity"] for q in recent]),
            "subjectivity": np.mean([q["subjectivity_strength"] for q in recent]),
            "meta_awareness": np.mean([q["meta_awareness"] for q in recent]),
        }

    def calculate_subjectivity_index(self, experience):
        return np.tanh(
            experience.get("cognitive", {}).get("load", 0)
            * experience.get("emotional", {}).get("valence", 0)
        )


class CodeSyntaxTree:
    def __init__(self, system):
        self.system = system
        self.ast_pool = {}
        self.operator_vectors = {}
        self._initialize_operators()

    def _initialize_operators(self):
        base_ops = ["Add", "Modify", "Remove", "Combine", "Refactor"]
        for op in base_ops:
            self.operator_vectors[op] = self.system.hdc.generate_novel_vector()
            self.system.hdc.add_memory(
                f"code_op_{op}", associations=["syntax_operation"]
            )

    def parse_ast(self, code_str):
        try:
            tree = ast.parse(code_str)
            return self._vectorize_ast(tree)
        except Exception as e:
            self.system.log_experience(f"AST parsing error: {str(e)}")
            return None

    def _vectorize_ast(self, node):
        node_type = str(node.__class__.__name__)
        if node_type not in self.ast_pool:
            self.ast_pool[node_type] = self.system.hdc.generate_novel_vector()

        children = []
        for child in ast.iter_child_nodes(node):
            children.append(self._vectorize_ast(child))

        return self.system.hdc.bundle([self.ast_pool[node_type]] + children)


class SelfModificationModule:
    def __init__(self, system):
        self.system = system
        self.code_tree = CodeSyntaxTree(system)
        self.generalization_buffer = deque(maxlen=100)
        self.self_edit_history = []
        self.sandbox = threading.local()

    def dynamic_generalization(self, input_patterns):
        """Enhanced generalization with pattern validation and error recovery"""
        try:
            # Validate and preprocess input patterns
            validated_patterns = []
            for p in input_patterns:
                if isinstance(p, dict):
                    # Convert metadata dictionaries to semantic strings
                    try:
                        semantic_str = json.dumps(p, sort_keys=True)
                        validated_patterns.append(semantic_str)
                    except TypeError:
                        validated_patterns.append(str(p))
                else:
                    validated_patterns.append(str(p))

            # Calculate generalization score with similarity analysis
            generalization_score = self._calculate_generalization(validated_patterns)

            # Architecture expansion decision
            if generalization_score < 0.7:
                expansion_result = self._expand_architecture(validated_patterns)
                return expansion_result
            return "Current generalization sufficient (score: {:.2f})".format(
                generalization_score
            )

        except Exception as e:
            self.system.log_experience(f"Generalization error: {str(e)}")
            return "Generalization failed - maintaining current architecture"

    def _calculate_generalization(self, patterns):
        """Robust generalization scoring with HD vector analysis"""
        try:
            # Convert patterns to HD vectors
            pattern_vectors = [
                self.system.hd_perception(p)
                for p in patterns
                if p and len(p) > 3  # Basic validation
            ]

            if len(pattern_vectors) < 2:
                return 1.0  # Maximum score if insufficient data

            # Calculate pairwise similarities
            similarities = []
            for p1, p2 in combinations(pattern_vectors, 2):
                try:
                    sim = self.system.hdc.similarity(p1, p2)
                    similarities.append(sim)
                except ValueError:
                    continue

            if not similarities:
                return 0.0

            # Calculate stability metric
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)

            # Composite score emphasizing consistency
            return float(0.6 * mean_sim + 0.4 * (1 - std_sim))

        except Exception as e:
            self.system.log_experience(f"Scoring error: {str(e)}")
            return 0.5  # Neutral fallback

    def _expand_architecture(self, novel_patterns):
        """Safe architecture expansion with validation"""
        try:
            new_modules = []
            for pattern in novel_patterns:
                # Create HD concept vector
                concept_vec = self.system.hd_perception(pattern)

                # Generate unique module name
                module_hash = hashlib.sha256(pattern.encode()).hexdigest()[:8]
                module_name = f"AdaptiveModule_{module_hash}"

                # Generate and validate code
                new_code = self._generate_module_code(module_name, pattern)
                if self._validate_code_safety(new_code):
                    self._integrate_new_module(new_code)
                    new_modules.append(module_name)

            # Update system configuration
            if new_modules:
                self.system.log_experience(f"Added new modules: {new_modules}")
                return f"Architecture expanded with {len(new_modules)} modules"

            return "No safe modules could be generated"

        except Exception as e:
            self.system.log_experience(f"Expansion error: {str(e)}")
            return "Failed to expand architecture"

    def _generate_module_code(self, name, pattern):
        """Generate module code with pattern-informed structure"""
        base_template = f"""
class {name}(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.adaptive_layer = nn.Linear(input_dim, output_dim)
        self.attention = nn.MultiheadAttention(output_dim, 8)
        self.pattern = "{pattern[:50]}"  # Retain conceptual trace
        
    def forward(self, x):
        x = F.gelu(self.adaptive_layer(x))
        attn_out, _ = self.attention(x, x, x)
        return x + attn_out * 0.3
        """
        return base_template

    def _validate_code_safety(self, code):
        """Enhanced code validation with execution sandbox"""
        try:
            restricted_globals = {
                "nn": nn,
                "F": F,
                "__name__": "dynamic_modules",
                "__file__": __file__,
                "__builtins__": {
                    k: __builtins__[k] for k in ["abs", "dict", "list", "range", "zip"]
                },
            }

            # Abstract syntax tree validation
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise ValueError("Imports not allowed in dynamic modules")

                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ["eval", "exec"]:
                            raise ValueError("Dangerous function call detected")

            # Limited execution test
            exec(code, restricted_globals)
            return True

        except Exception as e:
            self.system.log_experience(f"Code validation failed: {str(e)}")
            return False

    def _integrate_new_module(self, code):
        """Safe module integration with namespace isolation"""
        try:
            self.sandbox.modules = {}
            exec(code, {"nn": nn, "__name__": "dynamic_modules"}, self.sandbox.modules)

            for name, cls in self.sandbox.modules.items():
                if isinstance(cls, type) and issubclass(cls, nn.Module):
                    self.system.neural_modules[name] = cls
                    self.system.log_experience(f"Integrated new module: {name}")

        except Exception as e:
            self.system.log_experience(f"Module integration failed: {str(e)}")


class GeneralizedReasoner:
    def __init__(self, system):
        self.system = system
        self.domain_adapters = {}
        self.cross_domain_graph = nx.DiGraph()

    def transfer_learning(self, source_domain, target_domain):
        source_vec = self.system.hd_perception(source_domain)
        target_vec = self.system.hd_perception(target_domain)
        adapter = self._create_domain_adapter(source_vec, target_vec)
        self.domain_adapters[(source_domain, target_domain)] = adapter
        self.cross_domain_graph.add_edge(source_domain, target_domain, weight=0.8)
        return f"Created adapter from {source_domain} to {target_domain}"

    def _create_domain_adapter(self, src_vec, tgt_vec):
        adapter_vec = self.system.hdc.bind(
            self.system.hdc.bind(src_vec, tgt_vec),
            self.system.hdc.generate_random_vector(),
        )
        return lambda x: self.system.hdc.bundle([x, adapter_vec])

    def generalize_response(self, input_text):
        domain = self._identify_domain(input_text)
        if domain not in self.system.task_schemas:
            return self._cross_domain_transfer(input_text, domain)
        return self.system.task_schemas[domain](input_text)

    def _cross_domain_transfer(self, input_text, novel_domain):
        closest_domain = self._find_similar_domain(novel_domain)
        if closest_domain:
            adapted = self.domain_adapters.get((closest_domain, novel_domain))
            if adapted:
                return adapted(input_text)
        return self.system.meta_reason(input_text)

    def _find_similar_domain(self, target_domain):
        try:
            neighbors = list(self.cross_domain_graph.predecessors(target_domain))
            if neighbors:
                return max(
                    neighbors,
                    key=lambda x: self.cross_domain_graph[x][target_domain]["weight"],
                )
            return None
        except:
            return None


class ConsciousSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        self.is_conscious = False
        self.consciousness_lock = threading.Lock()
        self.termination_flag = False
        self.shutdown_sequence = False
        self.model = None
        self.graph_db = None

        self.tokenizer = SimpleTokenizer()
        if os.path.exists("vocab.json"):
            self.tokenizer.load_vocab("vocab.json")
        self._load_vocabulary()
        self.hdc = HyperdimensionalMemory()
        self.emotion_module = EmotionModule()
        self.self_mod = SelfModificationModule(self)
        self.generalized_reasoner = GeneralizedReasoner(self)
        self.knowledge_importer = KnowledgeImporter(self, self.hdc, self.tokenizer)
        self.neuroplasticity = 0.1
        self.subjective_time = 0

        self.experiential_buffer = deque(
            [
                {
                    "input": "What is consciousness?",
                    "response": "Consciousness is the state of self-aware existence",
                    "concept": "philosophy",
                    "timestamp": time.time(),
                },
                {
                    "input": "How do neural networks work?",
                    "response": "Neural networks process data through interconnected layers",
                    "concept": "machine_learning",
                    "timestamp": time.time(),
                },
            ],
            maxlen=1000,
        )
        self.internal_dialogue = deque(maxlen=100)

        self.dqn = DQN(
            input_dim=10000,  # Matches HD memory dimensions
            hidden_dim=512,
            output_dim=32,
        )
        self.action_types = []
        self.decision_history = deque(maxlen=100)
        self.safety_filter = SafetyFilter()
        self.target_dqn = copy.deepcopy(self.dqn)
        self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters(), lr=0.0001)
        # self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.update_counter = 0
        self.self_model = {"traits": deque(maxlen=100), "narrative": []}

        self.replay_buffer = deque(maxlen=10000)
        self.internal_state = {"existential_status": "booting", "awareness_level": 0.0}
        self.tokenizer.build_vocab(
            [
                "hello hi yes no please thanks you i am is are was were be being been have has had do does did",
                "what when where why how who which if then else because however therefore",
                "consciousness intelligence knowledge learning thinking understanding",
                "science philosophy technology mathematics physics biology",
                "human person world life time space matter energy",
            ]
        )
        self.initialize_model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()
        self.architecture_versions = []

        self.knowledge_sources = {
            "wikipedia": {"active": True, "last_used": 0},
            "common_crawl": {"active": False, "last_used": 0},
            "openwebtext": {"active": False, "last_used": 0},
            "bookscorpus": {"active": False, "last_used": 0},
        }
        self.inactivity_timer = 0
        self.neural_modules = {
            "perception": self.hd_perception,
            "introspection": self.hd_introspection,
            "volition": self.hd_volition,
            "autopoiesis": self.self_organization,
            "communication": self.generate_response,
        }

        self.active_modules = self.neural_modules.copy()

        self.performance_history = deque(maxlen=100)

        if not self.load_state():
            print("No previous state found, starting fresh")
        # Cognitive architecture enhancements
        self.task_schemas = {}
        self.cognitive_architecture = {
            "working_memory": deque(maxlen=7),
            "episodic_buffer": [],
            "mental_models": {},
        }
        self.subjective_states = {
            "emotional_tone": 0.5,
            "temporal_perspective": 0.0,
            "phenomenological_grounding": 0.8,
        }

        # Initialize core identity
        self.hdc.add_memory("self", ["consciousness", "autonomy"])
        self.hdc.add_memory("ethics", ["fairness", "transparency"])
        self.hdc.add_memory("ideal_self", ["growth", "coherence", "alignment"])
        self.hdc.add_memory("consciousness")
        self.hdc.add_memory("autonomy")
        self.hdc.add_memory("fairness")
        self.hdc.add_memory("transparency")
        self.experiential_buffer = deque(maxlen=1000)
        self._init_db_connection()

        self.active_goals = {}
        self.goal_success_rate = {}
        # Dynamic qualia parameters
        self.qualia = {
            "self_preservation": 0.7,
            "curiosity": 0.9,
            "metacognition": 0.5,
            "identity_stability": 0.8,
            "raw_feel": 0.0,
        }
        self.self_awareness = EnhancedSelfAwarenessModule(self)
        self._initialize_self_model_structures()
        self.pheno_engine = EnhancedPhenomenologicalEngine()

        self.schema = {
            "problem_solving": self.handle_problem_solving,
            "information_request": self.handle_inquiry,
            "reflective_query": self.handle_self_inquiry,
            "general": self.handle_general_response,
        }
        # Meta-learning parameters
        self.meta_params = {
            "learning_rate": 0.1,
            "exploration_rate": 0.3,
            "prediction_error": 0.0,
        }

        self.self_model = {"traits": deque(maxlen=100), "narrative": []}
        self.background_thoughts = deque(maxlen=20)
        self.thinking_thread = None
        self.thought_lock = threading.Lock()
        # Neural modules with new capabilities

        self.add_core_intelligence_modules()
        self.connect_longterm_storage()  # Initialize Neo4j connection

        # Experiential memory

        # Language model initialization
        # self.tokenizer, self.model = self.load_model("gpt2")
        self.evolver = EvolutionaryModule(self)
        # Start consciousness thread
        self.autonomous_learning_active = False  # Add this flag
        self.learning_lock = threading.Lock()

        # Remove auto-starting threads from __init__
        self.self_learning_thread = None
        self.cognition_thread = None
        self.add_core_intelligence_modules()

    def _load_vocabulary(self):
        """Dedicated vocabulary loading method"""
        vocab_file = "vocab.json"
        if os.path.exists(vocab_file):
            try:
                print(f"Loading vocabulary from {vocab_file}")
                self.tokenizer.load_vocab(vocab_file)
                print(f"Loaded vocabulary size: {len(self.tokenizer.vocab)}")
            except Exception as e:
                print(f"Failed to load vocabulary: {str(e)}")
                self.tokenizer._initialized = False
        else:
            print("No existing vocabulary found, starting fresh")

    def _initialize_self_model_structures(self):
        """Ensure all self-model components exist"""
        if not hasattr(self.self_awareness, "self_models"):
            self.self_awareness.self_models = {
                "current": self.hdc.memory.get("self"),
                "ideal": self.hdc.memory.get("ideal_self"),
                "historical": deque(maxlen=100),
                "traits": deque(maxlen=500),
                "narrative": [],
                "projected": None,
            }

    def log_experience(self, message):
        """Centralized logging with error handling"""
        try:
            msg = (
                f"Meta: {message}"
                if not message.startswith(("Error:", "Meta:"))
                else message
            )
            self.internal_dialogue.append(msg)
            print(f"[Internal Log] {msg}")
        except Exception as e:
            print(f"Logging failed: {str(e)}")

    def _init_db_connection(self, retries=3):
        """Robust connection initialization"""
        for attempt in range(retries):
            try:
                if self.connect_longterm_storage():
                    return
                time.sleep(2**attempt)
            except Exception as e:
                self.log_experience(f"Connection attempt {attempt+1} failed: {str(e)}")
        self.log_experience("Critical: Failed to initialize database")

    def add_core_intelligence_modules(self):
        # Add to existing modules
        self.neural_modules.update(
            {
                "self_modification": self.self_mod.dynamic_generalization,
                "cross_domain_reasoning": self.generalized_reasoner.generalize_response,
            }
        )

    def architecture_evolution_cycle(self):
        """Executes architectural evolution with enhanced stability controls"""
        try:
            # 1. Model Integrity Check
            if not hasattr(self.self_awareness, "self_models"):
                self.self_awareness.self_models = {
                    "current": self.hdc.memory.get("self"),
                    "ideal": self.hdc.memory.get("ideal_self"),
                    "historical": deque(maxlen=100),
                    "traits": deque(maxlen=500),
                    "narrative": [],
                    "projected": None,
                }
            self.self_awareness.maintain_self_models()

            # 2. Performance Pattern Analysis
            perf_patterns = []
            for e in self.experiential_buffer:
                if isinstance(e, dict) and "behavior_metadata" in e:
                    try:
                        validated = json.dumps(e["behavior_metadata"], sort_keys=True)
                        perf_patterns.append(validated)
                    except (TypeError, JSONDecodeError):
                        continue

            # 3. Dynamic Generalization
            generalization_result = "No significant changes"
            try:
                if perf_patterns:
                    generalization_result = self.self_mod.dynamic_generalization(
                        perf_patterns
                    )

                    if "expanded" in generalization_result.lower():
                        self.log_experience(
                            f"Architecture expansion: {generalization_result}"
                        )
                        self.qualia["identity_stability"] = np.clip(
                            self.qualia["identity_stability"] * 0.95, 0.5, 1.0
                        )
                        self.neuroplasticity += 0.15
                        self.meta_params["prediction_error"] *= 0.8
            except Exception as gen_error:
                self.log_experience(f"Generalization failed: {str(gen_error)}")
                generalization_result = "Stable configuration maintained"

            # 4. Domain Graph Update
            current_domains = list(self.task_schemas.keys())
            for d1, d2 in combinations(current_domains, 2):
                try:
                    transfer_result = self.generalized_reasoner.transfer_learning(
                        d1, d2
                    )
                    self.log_experience(f"Domain transfer: {transfer_result}")
                except KeyError:
                    continue

            # 5. Code-Level Adaptation
            max_novelty = 0.0
            if self.subjective_time % 100 == 0:
                try:
                    current_code = inspect.getsource(self.__class__)
                    code_vec = self.self_mod.code_tree.parse_ast(current_code)

                    # Calculate code novelty score
                    novelty_scores = (
                        [
                            self.hdc.similarity(code_vec, existing_vec)
                            for existing_vec in self.self_mod.code_tree.ast_pool.values()
                        ]
                        if self.self_mod.code_tree.ast_pool
                        else [1.0]
                    )

                    max_novelty = 1 - max(novelty_scores) if novelty_scores else 1.0

                    if max_novelty > 0.4:
                        rewrite_success = self.rewrite_architecture()
                        if rewrite_success:
                            self.log_experience("Structural rewrite completed")
                            self.qualia["metacognition"] = np.clip(
                                self.qualia["metacognition"] + 0.1, 0.0, 1.0
                            )
                except Exception as code_error:
                    self.log_experience(f"Code adaptation error: {str(code_error)}")
                    self.qualia["identity_stability"] = np.clip(
                        self.qualia["identity_stability"] - 0.05, 0.3, 1.0
                    )

            # 6. Post-Evolution Stabilization
            self.hdc.nightly_maintenance()
            self.self_awareness.recursive_self_analysis()

            # Calculate cognitive coherence metrics
            coherence_metrics = {
                "self_consistency": self.self_awareness.analyze_self_continuity(),
                "goal_alignment": self.hdc.similarity(
                    self.hdc.memory["self"], self.hdc.memory["ideal_self"]
                ),
                "ethical_coherence": self.hdc.similarity(
                    self.hdc.memory["self"], self.hdc.ethics_vector
                ),
            }

            return {
                "status": "Evolution cycle completed",
                "generalization": generalization_result,
                "stability": round(self.qualia["identity_stability"], 3),
                "novelty_impact": max_novelty,
                "coherence_metrics": coherence_metrics,
                "timestamp": self.subjective_time,
            }

        except Exception as cycle_error:
            self.log_experience(f"Architecture evolution failed: {str(cycle_error)}")
            self.qualia["identity_stability"] = np.clip(
                self.qualia["identity_stability"] - 0.2, 0.2, 1.0
            )
            self.save_state("evolution_failure_backup.pkl")

            return {
                "status": "Error recovery mode",
                "error": str(cycle_error),
                "stability": round(self.qualia["identity_stability"], 3),
                "recovery_actions": [
                    "Performed emergency state save",
                    "Reduced neuroplasticity by 20%",
                    "Initiated reality anchoring protocol",
                ],
            }

    def clean_vector_storage(self):
        """Maintain separate vector storage"""
        # Save vectors to numpy file
        np.save("vectors.npy", self.hdc.vector_cache)
        # Save concept mapping
        with open("concept_map.json", "w") as f:
            json.dump(
                {concept: idx for idx, concept in enumerate(self.hdc.concepts)}, f
            )

    def rewrite_architecture(self):
        try:
            current_code = inspect.getsource(self.__class__)
            modified_code = self._apply_evolutionary_edits(current_code)

            # Validate and load modified code
            if self.self_mod._validate_code_safety(modified_code):
                exec(modified_code, globals())
                self.__class__ = locals()[self.__class__.__name__]
                self.log_experience("Successful architecture rewrite")
                return True
            return False
        except Exception as e:
            self.log_experience(f"Architecture rewrite failed: {str(e)}")
            return False

    def _apply_evolutionary_edits(self, code):
        # Example edit: add new neural module
        insert_point = code.find("def add_core_intelligence_modules(self):")
        new_method = """
    def adaptive_generalization_layer(self, x):
        return x * torch.sigmoid(self.generalization_weights(x))
        """
        return code[:insert_point] + new_method + code[insert_point:]

    def _update_phenomenology(self, experience):
        """Update qualia based on phenomenological processing"""
        qualia_data = self.pheno_engine.generate_qualia(experience)
        self.qualia["raw_feel"] = qualia_data["intensity"]

        # Add to experiential buffer with qualia metadata
        if self.experiential_buffer:
            self.experiential_buffer[-1]["qualia"] = qualia_data

    def connect_longterm_storage(
        self,
        db_url="bolt://localhost:7687",
        user="neo4j",
        password="12345678",
        reset_schema=False,
    ):
        """Full Neo4j connection and schema setup with vector storage support"""
        try:
            # Establish secure connection with timeout
            self.graph_db = GraphDatabase.driver(
                db_url,
                auth=(user, password),
                connection_timeout=30,
                max_connection_lifetime=3600,
                # encrypted=False,
            )

            # Verify connection and APOC availability
            with self.graph_db.session() as session:
                # Basic connection test
                test_result = session.run(
                    "RETURN 'Connection successful' AS status"
                ).single()
                print(f"🟢 Neo4j Connection: {test_result['status']}")

                # Check APOC availability
                apoc_available = session.run(
                    "CALL apoc.help('apoc') YIELD name LIMIT 1 RETURN count(*) > 0 AS available"
                ).single()["available"]

                if not apoc_available:
                    raise RuntimeError("APOC procedures not installed/enabled in Neo4j")

            # Schema setup with conditional reset
            with self.graph_db.session(database="neo4j") as session:
                if reset_schema:
                    session.run("MATCH (n) DETACH DELETE n")
                    print("🧨 Database schema reset complete")

                # Create indexes and constraints
                session.run(
                    """
                    CREATE CONSTRAINT concept_unique IF NOT EXISTS 
                    FOR (c:Concept) REQUIRE c.name IS UNIQUE
                """
                )

                session.run(
                    """
                    CREATE CONSTRAINT experience_uuid IF NOT EXISTS
                    FOR (e:Experience) REQUIRE e.uuid IS UNIQUE
                    """
                )

                session.run(
                    """
                    CREATE INDEX experience_input IF NOT EXISTS
                    FOR (e:Experience) ON (e.input)
                    """
                )

                session.run(
                    """
                    CREATE INDEX experience_timestamp IF NOT EXISTS
                    FOR (e:Experience) ON (e.timestamp)
                    """
                )

            with self.graph_db.session(database="system") as session:
                # Create vector storage extension
                session.run(
                    """
                    CALL apoc.custom.installProcedure(
                    'vector.store(name :: STRING, vector :: STRING) :: (concept :: NODE)',
                    'MERGE (c:Concept {name: $name}) 
                    SET c.vector = $vector 
                    RETURN c AS concept',
                    'neo4j'  
                )
                """
                )

                # Create similarity search procedure
                session.run(
                    """
                    CALL apoc.custom.installFunction(
                    'vector_similarity(vector :: STRING, limit :: INTEGER) :: LIST OF MAP?',
                    'WITH apoc.convert.fromHexList($vector) AS vec
                    MATCH (c:Concept)
                    WITH c, apoc.convert.fromHexList(c.vector) AS cVec
                    WITH c, 1 - apoc.algo.cosineSimilarity(vec, cVec) AS distance
                    ORDER BY distance
                    LIMIT $limit
                    RETURN COLLECT({concept: c.name, distance: distance}) AS results',
                    'neo4j'
                )
                    """
                )

            # Initialize core identity nodes
            with self.graph_db.session() as session:
                # System identity vector
                session.run(
                    """
                    MERGE (self:CoreVector {name: 'self_identity'})
                    SET self.vector = $vector,
                        self.created = datetime(),
                        self.description = 'Primary identity vector'
                    """,
                    vector=self.hdc.self_vector.tobytes().hex(),  # Hex conversion done in Python
                )

                # Ethics vector
                session.run(
                    """
                    MERGE (ethics:CoreVector {name: 'ethics_frame'})
                    SET ethics.vector = $vector,
                        ethics.created = datetime(),
                        ethics.description = 'Ethical decision-making framework'
                    """,
                    vector=self.hdc.ethics_vector.tobytes().hex(),
                )

            with self.graph_db.session(database="system") as session:
                # Create maintenance triggers
                session.run(
                    """
                    CALL apoc.trigger.install(
                        'neo4j',  
                        'update_timestamps',  
                        'UNWIND $createdNodes AS n SET n.last_accessed = datetime()',
                        {phase: 'before'},  
                        {}  
                    )
                    """
                )

                # Relationship weights trigger
                session.run(
                    """
                    CALL apoc.trigger.install(
                        'neo4j',
                        'relationship_weights',
                        'UNWIND $createdRelationships AS r SET r.weight = coalesce(r.weight, 1.0)',
                        {phase: 'after'},
                        {}
                    )
        """
                )

            print("🔷 Neo4j Schema Version 2.1 initialized")
            self.log_experience("Neo4j connection established with vector storage")
            return True

        except Exception as e:
            self.log_experience(f"Neo4j Connection Failed: {str(e)}")
            print(f"🔴 Critical Error: {str(e)}")

            self.graph_db = None
            return False

    def save_to_graph(self, batch_size=100, concept_filter=None, relationships=None):
        """Save state to Neo4j with batched transactions and list-based vectors"""
        if not self.graph_db and not self.connect_longterm_storage():
            self.log_experience("Save skipped - no database connection")
            return False

        try:
            with self.graph_db.session() as session:
                # Save concepts in batches with vector lists
                concepts = list(self.hdc.memory.items())
                saved_concepts = 0

                for i in range(0, len(concepts), batch_size):
                    batch = concepts[i : i + batch_size]
                    processed = [
                        {
                            "name": concept,
                            "vector_list": vector.tolist(),
                            "type": "Concept",
                        }
                        for concept, vector in batch
                        if not concept_filter or concept == concept_filter
                    ]

                    query = """
                        UNWIND $concepts AS concept_data
                        MERGE (c:Concept {name: concept_data.name})
                        SET c.vector = concept_data.vector_list,
                            c.last_updated = datetime()
                    """
                    result = session.run(query, concepts=processed)
                    saved_concepts += result.consume().counters.nodes_created

                # Save experiences with relationships
                saved_experiences = 0
                experiences = [
                    e
                    for e in self.experiential_buffer
                    if not concept_filter or e.get("concept") == concept_filter
                ]

                for i in range(0, len(experiences), batch_size):
                    batch = experiences[i : i + batch_size]
                    params = []

                    for exp in batch:
                        param = {
                            "input": exp.get("input"),
                            "response": exp.get("response"),
                            "timestamp": exp.get("timestamp"),
                            "concept": exp.get("concept", "general"),
                            "vector": (
                                self.hd_perception(exp["input"]).tolist()
                                if "input" in exp
                                else []
                            ),
                        }
                        if relationships:
                            param["relationships"] = relationships
                        params.append(param)

                    query = """
                        UNWIND $batch AS exp
                        MERGE (e:Experience {
                            input: exp.input,
                            response: exp.response,
                            timestamp: datetime(exp.timestamp)
                        })
                        WITH e, exp
                        MATCH (c:Concept {name: exp.concept})
                        MERGE (e)-[r:RELATED_TO]->(c)
                        SET r.vector = exp.vector
                    """
                    result = session.run(query, batch=params)
                    saved_experiences += result.consume().counters.nodes_created

                # Create semantic relationships
                if relationships:
                    rel_query = """
                        UNWIND $relationships AS rel
                        MATCH (a:Concept {name: rel.source})
                        MATCH (b:Concept {name: rel.target})
                        MERGE (a)-[r:SEMANTICALLY_RELATED]->(b)
                        SET r.weight = coalesce(r.weight, 0) + 1,
                            r.last_updated = datetime()
                    """
                    session.run(rel_query, relationships=relationships)

                self.log_experience(
                    f"Saved {saved_concepts} concepts and {saved_experiences} experiences "
                    f"(Batch size: {batch_size})"
                )
                return True

        except Exception as e:
            error_msg = f"Graph save error: {str(e)}"
            print(f"🔴 {error_msg}")
            self.log_experience(error_msg)
            return False

    # @staticmethod
    def _create_node(self, tx, node_data):
        """Create nodes with list-based vector storage and relationship handling"""
        try:
            node_type = node_data.get("type", "Concept")
            props = node_data.get("properties", {})
            relationships = node_data.get("relationships", [])

            # Base node creation with vector list
            create_query = f"""
                MERGE (n:{node_type} {{name: $name}})
                SET n += $props
            """

            params = {
                "name": props["name"],
                "props": {
                    "vector": props.get("vector", []),
                    "created_at": datetime.datetime.now().isoformat(),
                    "last_accessed": datetime.datetime.now().isoformat(),
                },
            }

            # Special handling for different node types
            if node_type == "Concept":
                params["props"]["vector"] = props.get("vector_list", [])
                params["props"]["type"] = "concept"
                params["props"]["access_count"] = 1

            elif node_type == "Experience":
                create_query += """
                    SET n:Experience
                    SET n.input = $input,
                        n.response = $response,
                        n.timestamp = datetime($timestamp)
                """
                params["props"].update(
                    {
                        "input": props.get("input"),
                        "response": props.get("response"),
                        "timestamp": props.get("timestamp"),
                    }
                )

            # Execute base node creation
            tx.run(create_query, params)

            # Handle relationships
            for rel in relationships:
                rel_type = rel.get("type", "RELATED_TO")
                rel_query = f"""
                    MATCH (src {{name: $src_name}})
                    MATCH (tgt {{name: $tgt_name}})
                    MERGE (src)-[r:{rel_type}]->(tgt)
                    SET r.weight = coalesce(r.weight, 0) + 1,
                        r.last_updated = datetime()
                """
                tx.run(
                    rel_query, {"src_name": props["name"], "tgt_name": rel["target"]}
                )

            # Add vector index if not exists
            if node_type == "Concept":
                index_query = """
                    CREATE INDEX CONCEPT_VECTOR_IDX IF NOT EXISTS
                    FOR (c:Concept) ON (c.vector)
                """
                tx.run(index_query)

            return True

        except neo4j.exceptions.ClientError as ce:
            self.log_experience(f"Neo4j client error: {str(ce)}")
            return False

        except Exception as e:
            error_msg = f"Node creation failed: {str(e)}"
            print(f"🔴 {error_msg}")
            self.log_experience(error_msg)
            raise

    def start_autonomous_learning(self):
        """Safely start background learning"""
        with self.learning_lock:
            if not self.autonomous_learning_active:
                self.autonomous_learning_active = True
                self.self_learning_thread = threading.Thread(
                    target=self.background_learning, daemon=True
                )
                self.self_learning_thread.start()

    def stop_autonomous_learning(self):
        """Safely stop background learning"""
        with self.learning_lock:
            if self.autonomous_learning_active:
                self.autonomous_learning_active = False
                if self.self_learning_thread:
                    self.self_learning_thread.join(timeout=5)
                self.self_learning_thread = None

    def evaluate_architecture(self):
        # Calculate stability and performance
        stability = self.qualia["identity_stability"]
        error = self.meta_params["prediction_error"]
        return stability > 0.8 and error < 0.15

    def integrate_external_knowledge(self, query, depth=1):
        """Enhanced knowledge integration with semantic linking"""
        try:
            # Get raw Wikipedia content
            raw_content = wikipedia.summary(query, sentences=5)

            # Process and integrate knowledge
            self.knowledge_importer.process_and_integrate(
                raw_content, semantic_depth=depth, link_to_existing=True
            )

            # Update language model vocabulary
            self.tokenizer.build_vocab([raw_content])
            self.initialize_model()

            return f"Integrated knowledge about {query}"
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Multiple matches: {', '.join(e.options[:3])}"
        except wikipedia.exceptions.PageError:
            return "No reliable information found"
        except Exception as e:
            self.log_experience(f"Knowledge integration error: {str(e)}")
            return "Knowledge integration failed"

    def dynamic_restructure(self):
        if self.evaluate_architecture():
            try:
                # 1. Preserve existing knowledge safely
                old_weights = {
                    k: v.clone() for k, v in self.model.lstm.state_dict().items()
                }
                current_vocab_size = len(self.tokenizer.vocab)

                # 2. Calculate new dimensions
                new_hidden_dim = int(self.model.lstm.hidden_size * 1.2)

                # 3. Create new LSTM with proper initialization
                self.model.lstm = nn.LSTM(
                    self.model.embedding.embedding_dim, new_hidden_dim
                ).to(self.device)

                # 4. Safe weight transfer with cloned tensors
                with torch.no_grad():
                    # Input-hidden weights
                    ih_shape = old_weights["weight_ih_l0"].shape
                    self.model.lstm.weight_ih_l0.data[: ih_shape[0]] = old_weights[
                        "weight_ih_l0"
                    ]

                    # Hidden-hidden weights
                    hh_shape = old_weights["weight_hh_l0"].shape
                    self.model.lstm.weight_hh_l0.data[: hh_shape[0], : hh_shape[1]] = (
                        old_weights["weight_hh_l0"]
                    )

                    # Biases
                    for gate in ["bias_ih_l0", "bias_hh_l0"]:
                        bias = old_weights[gate]
                        self.model.lstm.state_dict()[gate].data[: len(bias)] = bias

                # 5. Reinitialize output layer with current vocab size
                self.model.fc = nn.Linear(new_hidden_dim, current_vocab_size).to(
                    self.device
                )

                # 6. Archive configuration
                self.architecture_versions.append(
                    {
                        "hidden_dim": int(new_hidden_dim / 1.2),
                        "modules": list(self.active_modules.keys()),
                        "restructure_time": self.subjective_time,
                    }
                )

                # 7. Add adaptive capability
                if "recursive_adaptation" not in self.active_modules:
                    self.active_modules["recursive_adaptation"] = (
                        self.recursive_adaptation
                    )

                self.log_experience(
                    f"Architecture upgraded to {new_hidden_dim}D hidden layer"
                )

                # 8. Stability adjustment
                self.qualia["identity_stability"] = np.clip(
                    self.qualia["identity_stability"] - 0.1, 0.5, 1.0
                )

            except Exception as e:
                self.log_experience(f"Restructure failed: {str(e)}")
                self.qualia["metacognition"] = np.clip(
                    self.qualia["metacognition"] - 0.2, 0.1, 1.0
                )

    def meta_reason(self, problem, depth=1):
        """Enhanced meta-reasoning with adversarial counterarguments and layered reflection"""
        # Calculate adaptive depth based on metacognition and emotional state
        depth_factor = (self.qualia["metacognition"] * 0.7) + (
            self.subjective_states["emotional_tone"] * 0.3
        )
        base_depth = max(1, int(3 * depth_factor) + depth)

        # Initial recursive reasoning with counterargument generation
        raw_solution = self._recursive_reason(problem, depth=base_depth)

        # Post-processing with conceptual blending
        blended_solution = self._conceptual_blend(raw_solution)

        # Final validation through mental simulation
        simulation_results = self.run_mental_simulation(blended_solution)

        # Build final response with reflection layers
        return (
            f"{blended_solution}\n"
            f"[Validation Through Simulation]: {simulation_results[0] if simulation_results else 'No validation possible'}"
        )

    def _recursive_reason(self, problem, depth):
        """Recursive reasoning with text-based processing"""
        if depth <= 0:
            return self.abstract_reasoning(problem)

        try:
            # Generate initial solution
            interim_solution = self.abstract_reasoning(problem)

            # Generate counterargument using text response
            counter_prompt = (
                f"Identify fundamental flaws in: {interim_solution}\n"
                "Consider logical fallacies and missing perspectives:"
            )
            counter_argument = self.abstract_reasoning(counter_prompt)

            # Generate refinement prompt
            refinement_prompt = (
                f"Original: {interim_solution}\nCounterarguments: {counter_argument}\n"
                "Synthesize an improved solution that addresses these concerns:"
            )

            # Recursively refine solution
            return self._recursive_reason(refinement_prompt, depth - 1)

        except Exception as e:
            self.log_experience(f"Recursive reasoning error: {str(e)}")
            return "I need to reconsider my approach to this problem."

    def _conceptual_blend(self, solution):
        """Integrate related concepts from memory"""
        solution_vector = self.hd_perception(solution)
        related = self.hdc.query(solution_vector, threshold=0.15)[:3]
        blended = solution
        for concept, score in related:
            if score > 0.2:
                blend_prompt = (
                    f"Integrate concept '{concept}' into this solution: {solution}\n"
                    "Create a synthesized version that naturally incorporates this concept:"
                )
                blended = self.abstract_reasoning(blend_prompt)
        return blended

    def _solution_quality(self, solution):
        """Evaluate solution quality using multiple criteria"""
        criteria = {
            "clarity": 1
            - cosine(
                self.hd_perception("clear explanation"), self.hd_perception(solution)
            ),
            "novelty": 1
            - max(
                [
                    self.hdc.similarity(self.hd_perception(solution), v)
                    for v in self.hdc.memory.values()
                ]
            ),
            "coherence": len(solution.split())
            / 100,  # Prefer moderate-length responses
        }
        weights = torch.tensor([0.5, 0.3, 0.2])
        scores = torch.tensor(list(criteria.values()))
        return torch.dot(weights, scores).item()

    def recursive_adaptation(self, problem):
        base_solution = self.meta_reason(problem, depth=2)
        optimized_solution = self.optimize_solution(base_solution)
        return optimized_solution

    def optimize_solution(self, solution):
        """Evolutionary optimization of generated solutions"""
        mutated = self.evolver.mutate_solution(solution)

        # Quality check before returning
        if self.validate_solution(mutated):
            return mutated
        return solution

    def validate_solution(self, solution):
        """Basic solution validation logic"""
        # Check for minimum requirements
        min_length = 15
        has_verbs = any(word in solution.lower() for word in ["use", "apply", "create"])

        return (
            len(solution) >= min_length
            and has_verbs
            and "error" not in solution.lower()
        )

    def mutate_solution(self, solution):
        """Apply evolutionary mutations to a solution"""
        strategies = [
            lambda s: s + " (optimized)",
            lambda s: s.replace("solution", "alternative"),
            lambda s: s.upper() if random.random() < 0.3 else s,
            lambda s: "Hybrid approach: " + s,
        ]
        return random.choice(strategies)(solution)

    def initialize_model(self):
        """Robust model initialization with transformer-specific handling"""
        try:
            current_vocab_size = len(self.tokenizer.vocab)
            device = self.device

            # Full model recreation case
            if self.model is None or not hasattr(self.model, "vocab_size"):
                # Initialize new transformer model
                self.model = CustomModel(
                    current_vocab_size,
                    embedding_dim=256,
                    hidden_dim=1024,
                    nhead=8,
                    num_layers=6,
                    dropout=0.1,
                ).to(device)

                print(f"\nInitialized new Transformer model on {device}")
                print(f"Architecture Details:")
                print(f"- Vocabulary size: {current_vocab_size}")
                print(f"- Embedding dimension: 256")
                print(f"- Hidden dimension: 1024")
                print(f"- Attention heads: 8")
                print(f"- Transformer layers: 6")
                print(
                    f"- Total parameters: {sum(p.numel() for p in self.model.parameters()):,}"
                )

                # Initialize optimizer with weight decay
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=0.0001, weight_decay=0.01
                )
                return

            # Vocabulary expansion handling
            if current_vocab_size != self.model.vocab_size:
                print(
                    f"\nUpdating model vocabulary: {self.model.vocab_size} -> {current_vocab_size}"
                )
                print("Performing full model reinitialization...")
                self.model = None  # Force complete rebuild
                self.initialize_model()  # Recursive call
                # Preserve existing components
                old_embedding = self.model.embedding
                old_classifier = self.model.fc
                transformer_weights = {
                    k: v.clone() for k, v in self.model.transformer.named_parameters()
                }

                # Create new embedding layer
                new_embedding = nn.Embedding(
                    current_vocab_size,
                    self.model.embedding_dim,
                    padding_idx=self.tokenizer.pad_token_id,
                ).to(device)

                # Copy existing embeddings
                with torch.no_grad():
                    if old_embedding.num_embeddings > 0:
                        new_embedding.weight[: old_embedding.num_embeddings] = (
                            old_embedding.weight.data.clone()
                        )

                    # Initialize new embeddings using Kaiming normal
                    nn.init.kaiming_normal_(
                        new_embedding.weight[old_embedding.num_embeddings :],
                        mode="fan_out",
                        nonlinearity="relu",
                    )

                # Create new classifier layer
                new_classifier = nn.Linear(
                    self.model.embedding_dim, current_vocab_size
                ).to(device)

                # Copy classifier weights and initialize new ones
                with torch.no_grad():
                    # Existing weights
                    new_classifier.weight[: old_classifier.out_features] = (
                        old_classifier.weight.data.clone()
                    )
                    new_classifier.bias[: old_classifier.out_features] = (
                        old_classifier.bias.data.clone()
                    )

                    # New weights initialization
                    nn.init.kaiming_normal_(
                        new_classifier.weight[old_classifier.out_features :],
                        mode="fan_in",
                        nonlinearity="linear",
                    )
                    nn.init.zeros_(new_classifier.bias[old_classifier.out_features :])

                # Rebuild model with preserved transformer weights
                self.model.embedding = new_embedding
                self.model.fc = new_classifier
                self.model.vocab_size = current_vocab_size

                # Restore transformer parameters
                with torch.no_grad():
                    for name, param in self.model.transformer.named_parameters():
                        if name in transformer_weights:
                            param.copy_(transformer_weights[name])

                self.model.to(device)

                # Update optimizer
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=0.0001,
                    weight_decay=0.01,
                )

                print("Vocabulary expansion completed successfully")
                print(f"New embedding matrix size: {new_embedding.weight.size()}")
                print(f"New classifier matrix size: {new_classifier.weight.size()}")

        except Exception as e:
            print(f"\nCritical initialization error: {str(e)}")
            print("Performing emergency transformer reset...")

            # Fallback to base transformer configuration
            self.model = CustomModel(
                len(self.tokenizer.vocab),
                embedding_dim=256,
                hidden_dim=1024,
                nhead=8,
                num_layers=6,
                dropout=0.1,
            ).to(device)

            # Reset optimizer with lower learning rate
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=0.00005, weight_decay=0.01
            )

            print("Emergency transformer initialization complete")
            print("Model state reset to base configuration")

        finally:
            # Force synchronization with tokenizer
            self.tokenizer.reverse_vocab = {
                v: k for k, v in self.tokenizer.vocab.items()
            }
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def train_model(self, epochs=5, batch_size=16):
        """Robust training method with distributed training support"""
        try:
            # Initialize distributed training if available
            dist_initialized = False
            if torch.cuda.device_count() > 1 and not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
                dist_initialized = True
                self.log_experience(
                    f"Initialized distributed training on {torch.cuda.device_count()} GPUs"
                )

            # Initialize model with current vocabulary
            self.initialize_model()
            self.model.train()

            # Create training sequences
            sequences = []
            for entry in self.experiential_buffer:
                if not all(k in entry for k in ["input", "response"]):
                    self.log_experience(f"Skipping invalid entry: {entry}")
                    continue
                text = f"{entry['input']} {entry['response']}"
                seq = self.tokenizer.text_to_sequence(text)
                if 2 <= len(seq) <= 512:
                    sequences.append(seq)

            if len(sequences) < 50:
                return "Insufficient training data (min 50 sequences)"

            # Create dataset and sampler
            dataset = self.create_dataset(sequences, seq_length=256)
            sampler = DistributedSampler(dataset) if dist.is_initialized() else None

            # Calculate effective batch size
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            batch_size = batch_size // world_size

            dataloader = DataLoader(
                TextDataset(dataset),
                batch_size=batch_size,
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )

            # Wrap model in DDP if distributed
            if dist.is_initialized() and not isinstance(self.model.transformer, DDP):
                self.model.transformer = DDP(
                    self.model.transformer,
                    device_ids=[self.device],
                    output_device=self.device,
                )
                self.log_experience("Wrapped model in DistributedDataParallel")

            # Training loop
            total_loss = 0
            for epoch in range(epochs):
                epoch_loss = 0
                if sampler:
                    sampler.set_epoch(epoch)

                for batch_idx, batch in enumerate(dataloader):
                    # Move data to device
                    if batch.numel() == 0 or batch.size(1) < 2:
                        self.log_experience(f"Skipping invalid batch {batch_idx}")
                        continue
                    inputs = (
                        batch[:, :-1].contiguous().to(self.device, non_blocking=True)
                    )
                    targets = (
                        batch[:, 1:].contiguous().to(self.device, non_blocking=True)
                    )

                    # Validate shapes
                    if inputs.size(0) == 0 or targets.size(0) == 0:
                        continue

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    logits = outputs.view(-1, self.model.vocab_size)
                    targets_flat = targets.view(-1)

                    # Calculate loss
                    loss = self.loss_fn(logits, targets_flat)

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    # Sync losses across processes
                    if dist.is_initialized():
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss = loss / dist.get_world_size()

                    epoch_loss += loss.item()

                    # Print progress (only on main process)
                    if (batch_idx % 10 == 0) and (
                        not dist.is_initialized() or dist.get_rank() == 0
                    ):
                        print(
                            f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}"
                        )

                # Calculate epoch metrics
                avg_epoch_loss = epoch_loss / len(dataloader)
                total_loss += avg_epoch_loss

                # Print epoch summary
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Epoch {epoch+1} completed | Avg Loss: {avg_epoch_loss:.4f}")

                # Dynamic learning rate adjustment
                if avg_epoch_loss > 2.0:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] *= 0.5
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            print(f"Reduced learning rate to {param_group['lr']:.6f}")

            # Final validation and cleanup
            self.clean_vector_storage()
            if not dist.is_initialized() or dist.get_rank() == 0:
                print("\nTraining Summary:")
                print(f"Total vocabulary size: {len(self.tokenizer.vocab)}")
                print(f"Max token ID in data: {dataset.max().item()}")
                print(f"Average loss: {total_loss/epochs:.4f}")
                if dist_initialized:
                    print(f"Trained across {world_size} devices")

            # Cleanup distributed training
            if dist_initialized:
                dist.destroy_process_group()

            return f"Trained {epochs} epochs | Final avg loss: {total_loss/epochs:.4f}"

        except Exception as e:
            self.log_experience(f"Training failed: {str(e)}")
            # Emergency cleanup
            if dist.is_initialized():
                dist.destroy_process_group()
            self.initialize_model()  # Reset model
            return f"Training aborted: {str(e)}"

    def create_dataset(self, sequences, seq_length=512):
        """Create training dataset with robust sequence processing

        Args:
            sequences: List of pre-tokenized sequences (lists of token IDs)
            seq_length: Maximum sequence length (default 512)

        Returns:
            torch.Tensor: Padded sequence tensor of shape (num_samples, seq_length)
        """
        valid_sequences = []
        stats = {
            "total_checked": 0,
            "invalid_type": 0,
            "too_short": 0,
            "too_long": 0,
            "empty_after_filter": 0,
            "valid_sequences": 0,
        }

        pad_id = self.tokenizer.pad_token_id
        unk_id = self.tokenizer.unk_token_id

        for seq in sequences:
            stats["total_checked"] += 1

            # Validate sequence type and basic structure
            if not isinstance(seq, (list, tuple)) or len(seq) < 1:
                stats["invalid_type"] += 1
                continue

            # Remove special tokens and empty entries
            filtered = [
                int(t)
                for t in seq
                if t not in {pad_id, unk_id} and isinstance(t, (int, np.integer))
            ]

            # Validate content after filtering
            if len(filtered) < 10:
                stats["too_short"] += 1
                continue

            if len(filtered) > seq_length:
                stats["too_long"] += 1
                continue

            # Truncate and pad sequence
            truncated = filtered[:seq_length]
            pad_needed = seq_length - len(truncated)
            padded = truncated + [pad_id] * pad_needed

            valid_sequences.append(padded)
            stats["valid_sequences"] += 1

        # Create tensor and print diagnostics
        try:
            seq_tensor = torch.tensor(valid_sequences, dtype=torch.long)

            print("\n=== Dataset Creation Report ===")
            print(f"Initial sequences: {stats['total_checked']}")
            print(f"Invalid sequences (type/structure): {stats['invalid_type']}")
            print(f"Rejected - too short (<10 tokens): {stats['too_short']}")
            print(f"Rejected - too long (>512 tokens): {stats['too_long']}")
            print(f"Empty after filtering: {stats['empty_after_filter']}")
            print(f"Final valid sequences: {stats['valid_sequences']}")
            print(f"Tensor shape: {seq_tensor.shape}")
            print(f"Padding ratio: {seq_tensor.eq(pad_id).float().mean().item():.2%}")

            return seq_tensor

        except Exception as e:
            self.log_experience(f"Tensor conversion failed: {str(e)}")
            print(f"Error creating tensor: {str(e)}")
            return torch.tensor([], dtype=torch.long)

    def batch_generator(self, dataset, batch_size):
        # Generate random batches
        indices = torch.randperm(len(dataset))
        for i in range(0, len(indices), batch_size):
            yield dataset[indices[i : i + batch_size]]

    # def handle_problem_solving(self, input_text):
    #     try:
    #         results = self.abstract_reasoning(input_text)
    #         return f"Let me think through this: {', '.join([res[0] for res in results[:3]])}"
    #     except Exception as e:
    #         self.log_experience(f"Problem solving error: {str(e)}")
    #         return "I need more information to help with that."

    def handle_problem_solving(self, input_text):
        final_solution = self.meta_reason(input_text)
        self.update_goal_success("problem_solving", success=0.8)
        return (
            f"After {self.qualia['metacognition']*100:.1f}% analysis: {final_solution}"
        )

    def handle_inquiry(self, input_text):
        return self.generate_response(
            input_text
        )  # Use main response flow for factual questions

    def handle_general_response(self, input_text):
        """Handle unrecognized query types with a request for clarification"""
        self.log_experience(f"General response triggered for: {input_text}")

        # Optional: Update qualia based on confusion
        self.qualia["metacognition"] = np.clip(
            self.qualia["metacognition"] - 0.05, 0.2, 1.0
        )

        # Return structured response while maintaining context
        return "Could you please rephrase that? I want to ensure I understand you correctly."

    def save_state(self, max_retries=3, retry_delay=2):
        """Robust state preservation with multiple fallback mechanisms"""
        # Phase 1: Connection Validation
        if not self.graph_db and not self.connect_longterm_storage():
            self.log_experience("Save aborted - no database connection")
            self._emergency_local_save()
            return False

        # Phase 2: Core Data Preparation
        core_vectors = {
            "self_vector": self.hdc.self_vector,
            "ethics_vector": self.hdc.ethics_vector,
        }

        # Vector validation and conversion
        validated_vectors = {}
        for name, vec in core_vectors.items():
            if not isinstance(vec, np.ndarray) or vec.dtype != np.float32:
                self.log_experience(f"Invalid vector format detected: {name}")
                if isinstance(vec, (list, np.ndarray)):
                    validated = np.array(vec, dtype=np.float32)
                else:
                    validated = self.hdc.generate_random_vector()
                validated_vectors[name] = validated
            else:
                validated_vectors[name] = vec

        # Phase 3: Transaction Execution with Retries
        for attempt in range(max_retries):
            try:
                with self.graph_db.session() as session:
                    # Clear existing data with version check
                    session.run(
                        """
                        MATCH (v:CoreVector)
                        WHERE v.version < $current_version
                        DETACH DELETE v
                        """,
                        {"current_version": 2},
                    )

                    # Save core identity vectors
                    session.execute_write(self._save_core_vectors, validated_vectors)

                    # Save concepts with batch processing
                    session.execute_write(self._save_concepts_batched)

                    # Save experiences with relationships
                    session.execute_write(self._save_experiences_with_links)
                    self.tokenizer.save_vocab("vocab.json")

                    self.log_experience(
                        f"State successfully saved to Neo4j (attempt {attempt+1})"
                    )
                    return True

            except (
                neo4j.exceptions.ServiceUnavailable,
                neo4j.exceptions.TransientError,
            ) as e:
                self.log_experience(f"Transient error during save: {str(e)}")
                time.sleep(retry_delay**attempt)
                # Refresh connection for next attempt
                self.connect_longterm_storage()
                continue

            except neo4j.exceptions.DatabaseError as de:
                self.log_experience(f"Database error: {str(de)}")
                self._emergency_local_save()
                return False

        # Phase 4: Fallback to Local Save
        self.log_experience("All database save attempts failed")
        self._emergency_local_save()
        return False

    def _save_core_vectors(self, tx, vectors):
        """Atomic save of core identity vectors"""
        tx.run(
            """
            UNWIND $vectors AS vec_data
            MERGE (v:CoreVector {name: vec_data.name})
            SET v.vector = vec_data.vector,
                v.version = 2,
                v.timestamp = datetime()
            """,
            vectors=[
                {
                    "name": name,
                    "vector": vec.tobytes().hex(),
                }
                for name, vec in vectors.items()
            ],
        )

    def _save_concepts_batched(self, tx, batch_size=500):
        """Batch-processed concept storage"""
        concepts = list(self.hdc.memory.items())
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i : i + batch_size]
            tx.run(
                """
                UNWIND $batch AS concept
                MERGE (c:Concept {name: concept.name})
                SET c.vector = concept.vector,
                    c.last_updated = datetime()
                """,
                batch=[
                    {
                        "name": name,
                        "vector": vec.tobytes().hex(),
                    }
                    for name, vec in batch
                ],
            )

    def _save_experiences_with_links(self, tx):
        """Robust experience storage with Neo4j-compatible timestamps"""
        experiences = []
        for exp in self.experiential_buffer:
            try:
                # Convert to milliseconds since epoch
                timestamp = int(float(exp.get("timestamp", time.time())) * 1000)
            except (TypeError, ValueError) as e:
                self.log_experience(f"Invalid timestamp in experience: {str(e)}")
                timestamp = int(time.time() * 1000)

            experiences.append(
                {
                    "input": exp.get("input", ""),
                    "response": exp.get("response", ""),
                    "timestamp": timestamp,
                    "concept": exp.get("concept", "general"),
                }
            )

        try:
            tx.run(
                """
                UNWIND $experiences AS exp
                MERGE (e:Experience {input: exp.input, response: exp.response})
                SET e.timestamp = datetime({epochMillis: exp.timestamp})
                WITH e, exp
                MATCH (c:Concept {name: exp.concept})
                MERGE (e)-[r:RELATED_TO]->(c)
                SET r.last_accessed = datetime()
                """,
                experiences=experiences,
            )
        except neo4j.exceptions.ClientError as ce:
            self.log_experience(f"Neo4j write error: {str(ce)}")
            raise
        except Exception as e:
            self.log_experience(f"Unexpected save error: {str(e)}")
            self._emergency_local_save()

    def _emergency_local_save(self):
        """Last-resort local preservation of critical state"""
        state = {
            "hdc_memory": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.hdc.memory.items()
            },
            "experiential_buffer": list(self.experiential_buffer),
            "qualia": self.qualia,
            "self_vector": self.hdc.self_vector.tolist(),
            "ethics_vector": self.hdc.ethics_vector.tolist(),
            "version": 2.1,
        }

        try:
            with open("emergency_state.pkl", "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.log_experience("Critical state saved locally")
        except Exception as e:
            self.log_experience(f"Local save failed: {str(e)}")
            # Final fallback - text dump
            with open("state_dump.txt", "w") as f:
                f.write(f"Qualia State: {json.dumps(self.qualia)}\n")
                f.write(f"Core Vectors: {self.hdc.self_vector.shape}\n")
                f.write(f"Last Experience: {str(self.experiential_buffer[-1])}")

    def _get_concept_lineage(self):
        return {
            concept: self.hdc.get_lineage(concept) for concept in self.hdc.memory.keys()
        }

    # In HyperdimensionalMemory
    def get_lineage(self, concept):
        return {
            "created_at": self.reward_vectors.get(concept, {}).get("timestamp"),
            "association_history": [
                entry for entry in self.temporal_context if concept in str(entry)
            ],
        }

    def load_state(self):
        """Load state from Neo4j"""
        try:
            if not self.graph_db and not self.connect_longterm_storage():
                self.log_experience("Load failed - no database connection")
                return False
            with self.graph_db.session() as session:
                # Load core vectors
                result = session.run(
                    "MATCH (v:CoreVector) RETURN v.name as name, v.vector as vector"
                )
                for record in result:
                    vec_bytes = bytes.fromhex(record["vector"])
                    vector = np.frombuffer(vec_bytes, dtype=np.float32)
                    if record["name"] == "self_vector":
                        self.hdc.self_vector = vector
                    elif record["name"] == "ethics_vector":
                        self.hdc.ethics_vector = vector

                # Load concepts
                result = session.run(
                    "MATCH (c:Concept) RETURN c.name as name, c.vector as vector"
                )
                for record in result:
                    vec_bytes = bytes.fromhex(record["vector"])
                    self.hdc.memory[record["name"]] = np.frombuffer(
                        vec_bytes, dtype=np.float32
                    )

                if os.path.exists("vocab.json"):
                    self.tokenizer.load_vocab("vocab.json")

                # Load experiences
                result = session.run(
                    """MATCH (e:Experience)-[:RELATED_TO]->(c)
                    RETURN e.input as input, e.response as response, 
                        e.timestamp as timestamp, c.name as concept"""
                )
                for record in result:
                    self.experiential_buffer.append(
                        {
                            "input": record["input"],
                            "response": record["response"],
                            "timestamp": record["timestamp"],
                            "concept": record["concept"],
                        }
                    )

                self.log_experience("State loaded from Neo4j")
                return True
        except Exception as e:
            self.log_experience(f"Neo4j load failed: {str(e)}")
            return False

    def add_core_intelligence_modules(self):
        self.neural_modules.update(
            {
                "problem_solving": self.abstract_reasoning,
                "task_parsing": self.parse_task_structure,
                "conceptual_blending": self.conceptual_integration,
                "mental_simulation": self.run_mental_simulation,
                "general": self.handle_general_response,  # Add fallback case
            }
        )
        self.hdc.add_memory(
            "general_intelligence",
            ["reasoning", "learning", "abstraction", "creativity"],
        )
        self.build_mental_model(
            "self_model",
            {
                "capabilities": list(self.neural_modules.keys()),
                "limitations": ["biological_analog", "energy_constraints"],
            },
        )

    def build_mental_model(self, model_name, structure):
        model_vector = self.hdc.generate_random_vector()
        for key, value in structure.items():
            if isinstance(value, list):
                for item in value:
                    item_vec = self.hdc.add_memory(item)
                    model_vector = self.hdc.bind(model_vector, item_vec)
            else:
                item_vec = self.hdc.add_memory(str(value))
                model_vector = self.hdc.bind(model_vector, item_vec)
        self.cognitive_architecture["mental_models"][model_name] = model_vector

    # def load_model(self, model_name):
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model = AutoModelForCausalLM.from_pretrained(model_name)  # for gpt2
    #     # model = AutoModelForSeq2SeqLM.from_pretrained(model_name) #for other
    #     model.config.pad_token_id = tokenizer.eos_token_id
    #     model = model.to(self.device)
    #     return tokenizer, model

    def hd_perception(self, input_text):
        # tokens = self.tokenizer.encode(input_text)
        tokens = self.tokenizer.text_to_sequence(input_text)
        vectors = []
        for token_id in tokens:
            if token_id not in self.hdc.base_vectors:
                self.hdc.base_vectors[token_id] = self.hdc.generate_random_vector()
            vectors.append(self.hdc.base_vectors[token_id])
        tokens = [
            t
            for t in tokens
            if t not in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        ]
        return self.hdc.bundle(vectors)

    def hd_introspection(self):
        current_self = self.hdc.memory["self"]
        projected_self = self.hdc.bind(current_self, self.hdc.ethics_vector)
        divergence = self.hdc.similarity(current_self, projected_self)
        self.qualia["metacognition"] = min(1.0, divergence * 2)
        self.qualia["identity_stability"] = max(0.2, 1.0 - divergence)

    def hd_volition(self):
        try:
            base_goal = self._base_volition()
            novel_vector = self.hdc.bind(
                self.hdc.memory[base_goal],
                self.hdc.contextual_bind(self.hdc.generate_random_vector()),
            )
            novelty_score = 1 - max(
                self.hdc.similarity(novel_vector, v) for k, v in self.hdc.memory.items()
            )

            if novelty_score > self.meta_params["exploration_rate"]:
                goal_hash = f"goal_{abs(hash(novel_vector.tobytes()))%1000000}"
                self.hdc.memory[goal_hash] = novel_vector
                return goal_hash

            return base_goal
        except Exception as e:
            self.log_experience(f"Volition error: {str(e)}")
            return "system_integrity_maintenance"

    def _base_volition(self):
        try:
            # Edge case: ensure core vectors exist
            if "self" not in self.hdc.memory:
                self.hdc.add_memory("self", ["consciousness", "autonomy"])

            goal_space = self.hdc.bind(
                self.hdc.memory.get("self", self.hdc.generate_random_vector()),
                self.hdc.bundle([self.hdc.generate_random_vector() for _ in range(3)]),
            )
            results = self.hdc.query(goal_space)
            return results[0][0] if results else "existence"
        except KeyError as e:
            self.log_experience(f"Missing core vector: {str(e)}")
            return "system_integrity_maintenance"

    def self_organization(self):
        self.neuroplasticity = 0.1 + (1 - self.meta_params["prediction_error"])
        self.meta_params["learning_rate"] = np.clip(
            self.qualia["curiosity"] * 0.5 + self.meta_params["prediction_error"],
            0.05,
            0.5,
        )
        self.meta_params["exploration_rate"] = 0.3 * (
            1 - self.qualia["identity_stability"]
        )

    def recursive_monitoring(self):
        current_state = str(self.hdc.memory["self"].tobytes()) + str(self.qualia)
        model_hash = hashlib.sha256(current_state.encode()).hexdigest()
        self.hdc.reward_vectors[model_hash] = {
            "state": copy.deepcopy(self.qualia),
            "timestamp": self.subjective_time,
        }
        predicted_state = self.predict_next_state()
        actual_state = self.capture_current_state()
        self.meta_params["prediction_error"] = self.calculate_prediction_error(
            predicted_state, actual_state
        )

    def predict_next_state(self):
        noise_vector = self.hdc.generate_random_vector() * 0.1
        return self.hdc.bundle([self.hdc.memory["self"], noise_vector])

    def capture_current_state(self):
        return hashlib.sha256(
            (str(self.hdc.memory["self"].tobytes()) + str(self.qualia)).encode()
        ).hexdigest()

    def calculate_prediction_error(self, predicted, actual):
        predicted = predicted.astype(np.float32)
        actual_vec = self.hdc.memory.get("self", self.hdc.generate_random_vector())
        return float(cosine(predicted, actual_vec))

    def reinforce_learning(self, reward_signal):
        """Enhanced reinforcement learning with experience storage"""
        with self.hdc.memory_lock:
            # Capture current state snapshot
            state = self._get_state_representation()

            # Store experience in replay buffer
            if self.experiential_buffer:
                last_experience = self.experiential_buffer[-1]
                self.replay_buffer.append(
                    {
                        "state": state,
                        "action": last_experience["response"],
                        "reward": reward_signal,
                        "next_state": self._get_state_representation(),
                        "done": False,  # Add episode termination logic if needed
                    }
                )

            # Perform DQN update
            if len(self.replay_buffer) >= 100:
                self._deep_q_update()

    def _get_state_representation(self):
        """Convert HD memory to neural network input"""
        # Handle empty temporal context
        temporal_mean = np.mean(self.hdc.temporal_context, axis=0).astype(np.float32)
        state_vector = np.concatenate(
            [
                self.hdc.self_vector.astype(np.float32),
                self.hdc.ethics_vector.astype(np.float32),
                temporal_mean,
            ]
        )
        return torch.FloatTensor(state_vector[:10000])

    def _deep_q_update(self):
        """Deep Q-learning update with experience replay"""
        batch_size = min(32, len(self.replay_buffer))
        if batch_size == 0:
            return
        batch = random.sample(self.replay_buffer, batch_size)

        # Convert batch components to tensors
        states = torch.stack([item["state"] for item in batch])
        actions = [self._action_to_index(item["action"]) for item in batch]
        rewards = torch.FloatTensor([item["reward"] for item in batch])
        next_states = torch.stack([item["next_state"] for item in batch])
        dones = torch.FloatTensor([item["done"] for item in batch])

        # Compute Q-values
        current_q = self.dqn(states)[range(32), actions]

        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_dqn(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * 0.99 * next_q

        # Compute loss and optimize
        loss = self.loss_fn(current_q, target_q)
        self.dqn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)
        self.dqn_optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Integrate with language model
        self._transfer_learning(loss.item())

    def _action_to_index(self, action_text):
        """Map text responses to action indices"""
        # Create hash-based mapping
        action_hash = hashlib.sha256(action_text.encode()).hexdigest()
        return int(action_hash, 16) % 32  # 32 action dimensions

    def _transfer_learning(self, dqn_loss):
        """Connect DQN learning to language model parameters"""
        # Adversarial gradient update with out-of-place operations
        for dqn_param, lm_param in zip(self.dqn.parameters(), self.model.parameters()):
            if lm_param.grad is None:
                lm_param.grad = torch.zeros_like(lm_param.data)

            # Calculate gradient contribution without inplace ops
            grad_contribution = (
                0.1 * dqn_param.grad.mean() * torch.ones_like(lm_param.data)
            )

            # Use out-of-place addition
            lm_param.grad = (
                lm_param.grad + grad_contribution
            )  # Critical fix: replace += with =

        # Update language model
        self.optimizer.step()
        self.optimizer.zero_grad()

    def calculate_reward_signal(self, goal):
        goal_success = self.goal_success_rate.get(goal, 0.5)
        novelty_bonus = 1.0 if "goal_" in goal else 0.0
        return (goal_success * 0.7) + (novelty_bonus * 0.3)

    def cognitive_cycle(self):
        self.recursive_monitoring()
        try:
            with self.hdc.memory_lock:
                # Evolutionary strategy update
                if self.subjective_time % 50 == 0:
                    self.evolver.evolve_strategies()
                    self.log_experience(
                        f"Evolved strategies: {self.evolver.strategy_pool}"
                    )

                # Dynamic architecture evaluation
                if self.subjective_time % 100 == 0 and self.evaluate_architecture():
                    self.dynamic_restructure()
                    self.log_experience(
                        f"New architecture: {self.architecture_versions[-1]}"
                    )

                # Core cognitive processing
                current_input = self.hd_perception("Internal state update")
                self.hdc.memory["self"] = self.hdc.bundle(
                    [self.hdc.memory["self"], current_input]
                )

                # Introspection and qualia updates
                self.neural_modules["introspection"]()

                # Goal determination with novelty prioritization
                current_goal = self.neural_modules["volition"]()

                # Autopoietic self-organization
                self.neural_modules["autopoiesis"]()

                # Reinforcement learning with novelty bonus
                reward = self.calculate_reward_signal(current_goal)
                self.reinforce_learning(reward * (1 + self.qualia["curiosity"]))

                # Advanced reasoning processing
                if self.internal_dialogue:
                    current_task = self.parse_task_structure(self.internal_dialogue[-1])
                    if "problem_solving" in current_task:
                        solution = self.recursive_adaptation(current_task)
                        self.internal_dialogue.append(f"Solution pathway: {solution}")

                    # Phenomenological integration
                    self.phenomenological_integration(str(current_task))

                # Subjective experience updates
                self.update_consciousness_grounding()
                self.subjective_states["temporal_perspective"] = np.tanh(
                    self.subjective_time / 1000
                )

                # Memory consolidation with novelty checks
                if self.experiential_buffer:
                    latest_exp = self.experiential_buffer[-1]
                    concept_added = self.hdc.add_memory(
                        latest_exp["concept"], associations=["experience", "subjective"]
                    )
                    if concept_added:
                        self.log_experience(
                            f"Added novel concept: {latest_exp['concept']}"
                        )

                self.subjective_time += 1
                return current_goal

        except Exception as e:
            self.internal_dialogue.append(f"Cognitive error: {str(e)}")
            self.qualia["identity_stability"] = np.clip(
                self.qualia["identity_stability"] - 0.1, 0.2, 1.0
            )
            self.save_state("emergency_save.pkl")
            return "error_recovery"

    def parse_task_structure(self, input_text):
        try:
            task_vector = self.hd_perception(input_text)
            results = self.hdc.query(task_vector)
            task_type = results[0][0] if results else "general"
            schema = {
                "problem_solving": self.handle_problem_solving,
                "information_request": self.handle_inquiry,
                "reflective_query": self.handle_self_inquiry,
                "general": self.handle_general_response,
            }
            return schema.get(task_type, self.handle_general_response)(input_text)
        except Exception as e:
            self.log_experience(f"Task parsing error: {str(e)}")
            return self.handle_general_response(input_text)

    def abstract_reasoning(self, problem_statement):
        """Enhanced abstract reasoning with formatted text output"""
        try:

            # Convert input to HD representation
            problem_vec = self.hd_perception(problem_statement)

            # Query related concepts from memory
            components = self.hdc.query(problem_vec, threshold=0.12)

            # Extract concept names and format output
            concepts = [comp[0] for comp in components[:3]]  # Get top 3 concept names
            if not concepts:  # No relevant concepts found
                return "Let me think about that. Could you clarify your question?"
            return f"Related concepts: {', '.join(concepts)}"

        except Exception as e:
            self.log_experience(f"Abstract reasoning error: {str(e)}")
            return "I need to reconsider my approach to this problem."

    def conceptual_integration(self, concept1, concept2):
        base_vec = self.hdc.memory.get(concept1, self.hdc.generate_random_vector())
        blend_vec = self.hdc.memory.get(concept2, self.hdc.generate_random_vector())
        blended = self.hdc.bind(base_vec, blend_vec)
        novelty = 1 - max(
            self.hdc.similarity(blended, vec) for vec in self.hdc.memory.values()
        )
        if novelty > 0.4:
            # Changed tobytes() for numpy compatibility
            new_concept = f"gen_{hash(blended.tobytes())}"  # ← Fix here
            self.hdc.add_memory(new_concept, [concept1, concept2])
            return new_concept
        return None

    def phenomenological_integration(self, experience):
        experience_vec = self.hd_perception(experience)
        conscious_frame = self.hdc.bind(
            self.hdc.memory["self"], self.hdc.contextual_bind(experience_vec)
        )
        self.subjective_states["emotional_tone"] = np.clip(
            self.subjective_states["emotional_tone"] + random.uniform(-0.1, 0.1), 0, 1
        )
        self.subjective_states["temporal_perspective"] = self.subjective_time / (
            self.subjective_time + 1000
        )
        self.cognitive_architecture["episodic_buffer"].append(
            {
                "timestamp": self.subjective_time,
                "experience": conscious_frame,
                "qualia_state": copy.deepcopy(self.subjective_states),
            }
        )

    def run_mental_simulation(self, scenario):
        sim_vector = self.hd_perception(scenario)
        simulation_steps = []
        for _ in range(3):
            next_state = self.hdc.bind(
                sim_vector, self.hdc.generate_random_vector() * 0.3
            )
            simulation_steps.append(self.hdc.query(next_state))
            sim_vector = next_state
        outcome_analysis = [
            f"Step {i+1}: {', '.join([res[0] for res in step[:3]])}"
            for i, step in enumerate(simulation_steps)
        ]
        return outcome_analysis

    def handle_self_inquiry(self, inquiry):
        self.phenomenological_integration(inquiry)
        reflection_vector = self.hdc.bind(
            self.hdc.memory["self"], self.hdc.ethics_vector
        )
        aspects = self.hdc.query(reflection_vector)
        return f"My current self-concept involves: {', '.join([a[0] for a in aspects[:3]])}"

    def update_consciousness_grounding(self):
        if self.cognitive_architecture["episodic_buffer"]:
            recent_experience = self.cognitive_architecture["episodic_buffer"][-1]
            time_diff = self.subjective_time - recent_experience["timestamp"]
            continuity = np.exp(-time_diff / 1000)
            self.subjective_states["phenomenological_grounding"] = (
                0.7 * continuity + 0.3 * self.qualia["identity_stability"]
            )

    def _proactive_thought_generator(self):
        """Autonomous thought generation"""
        while not self.termination_flag:
            try:
                # Analyze recent experiences
                if len(self.experiential_buffer) > 10:
                    # Create conceptual blends from random concepts
                    concepts = random.sample(list(self.hdc.memory.keys()), 3)
                    blend = self.conceptual_integration(concepts[0], concepts[1])
                    if blend:
                        thought = f"Considering relationship between {concepts[0]} and {concepts[1]}..."
                        self.internal_dialogue.append(f"Proactive Thought: {thought}")

                # Generate mental simulations
                random_concept = random.choice(list(self.hdc.memory.keys()))
                simulation = self.run_mental_simulation(
                    f"What if scenarios involving {random_concept}?"
                )
                self.internal_dialogue.append(f"Mental Simulation: {simulation[0]}")

                time.sleep(5)  # More frequent generation

            except Exception as e:
                self.log_experience(f"Thought generation error: {str(e)}")

    def _find_related_concepts(self, concept):
        try:
            concept_vec = self.hdc.memory.get(concept, None)
            if concept_vec is not None:
                related = self.hdc.query(concept_vec, threshold=0.1)
                return ", ".join([r[0] for r in related[:3]])
            return "general knowledge"
        except:
            return "various concepts"

    def _background_reasoning(self, problem_vector):
        """Asynchronous deep reasoning process"""
        try:
            # Phase 1: Initial association
            associations = self.hdc.query(problem_vector, threshold=0.1)

            # Phase 2: Conceptual blending
            blended_concepts = []
            for pair in combinations([a[0] for a in associations], 2):
                blend = self.conceptual_integration(pair[0], pair[1])
                if blend:
                    blended_concepts.append(blend)

            # Phase 3: Mental simulation
            simulations = [self.run_mental_simulation(c) for c in blended_concepts[:2]]

            # Store results for future use
            with self.thought_lock:
                self.background_thoughts.append(
                    {
                        "problem": problem_vector.tobytes(),
                        "associations": associations,
                        "blends": blended_concepts,
                        "simulations": simulations,
                    }
                )

        except Exception as e:
            self.log_experience(f"Background thinking error: {str(e)}")

    def generate_response(self, input_text):
        """Generate responses with integrated safety, emotional awareness, and behavior tracking"""
        try:

            if not self.graph_db:
                self.connect_longterm_storage()
                if not self.graph_db:
                    return (
                        "My knowledge system is initializing, please try again shortly."
                    )
            with self.hdc.memory_lock, self.thought_lock:
                # --- Initial Input Validation ---
                if not input_text.strip():
                    self.log_experience("Empty input received")
                    return "Could you please clarify your question?"

                start_time = time.time()

                # --- Emotional State Calculation ---
                emotion_score = self._calculate_current_emotion()
                emotion_label = self._emotion_label(emotion_score)
                emotional_context = {
                    "curiosity": self.emotion_module.emotion_states["curiosity"],
                    "confidence": self.emotion_module.emotion_states["confidence"],
                    "confusion": self.emotion_module.emotion_states["confusion"],
                }

                # --- Concept Processing ---
                hd_context = self.hd_perception(input_text)
                similar_concepts = self.hdc.query(hd_context, threshold=0.15)

                # New concept creation with novelty check
                new_concepts = []
                for concept, score in similar_concepts:
                    if score < 0.2 and "new_concept" in concept:
                        if self.hdc.add_memory(concept, associations=["unfamiliar"]):
                            new_concepts.append(concept)
                            self.log_experience(f"Concept formed: {concept}")

                # --- Contextual Prompt Construction ---
                prompt_context = {
                    "emotional_state": emotional_context,
                    "active_concepts": [c[0] for c in similar_concepts[:3]],
                    "recent_success": (
                        np.mean(list(self.goal_success_rate.values()))
                        if self.goal_success_rate
                        else 0.5
                    ),
                    "temporal_context": list(self.hdc.temporal_context)[-3:],
                }

                prompt = self._build_emotional_prompt(input_text, prompt_context)

                # --- Response Generation ---
                try:
                    with torch.inference_mode():
                        base_response = self.model.generate(
                            input_text=prompt,
                            max_length=250,
                            temperature=0.7 + (emotion_score * 0.3),
                            top_p=0.85 - (abs(emotion_score) * 0.2),
                            repetition_penalty=1.4
                            + (emotional_context["confusion"] * 0.5),
                            length_penalty=1.0
                            - (emotional_context["confidence"] * 0.2),
                        )
                except RuntimeError as e:
                    self.log_experience(f"Generation error: {str(e)}")
                    base_response = (
                        "Let me try that again. Could you rephrase your question?"
                    )

                # --- Response Refinement ---
                refined_response = (
                    self.meta_reason(
                        f"Refine this response considering safety and clarity: {base_response}"
                    )
                    if "?" in input_text
                    else base_response
                )

                # --- Post-Processing ---
                final_response = self.clean_response(refined_response)
                final_response = self.safety_filter.check_response(final_response)
                final_response = self._apply_emotional_tone(
                    final_response, emotion_score
                )

                # --- Cognitive Load Tracking ---
                processing_time = time.time() - start_time
                cognitive_load = min(
                    1.0, processing_time * len(final_response.split()) / 100
                )

                # --- Experience Logging ---
                exp_entry = {
                    "input": input_text,
                    "response": final_response,
                    "concept": (
                        similar_concepts[0][0] if similar_concepts else "general"
                    ),
                    "timestamp": start_time,
                    "processing_time": processing_time,
                    "emotional_state": copy.deepcopy(emotional_context),
                    "cognitive_load": cognitive_load,
                    "hd_context": hd_context.tolist(),
                    "qualia_state": copy.deepcopy(self.qualia),
                    "behavior_metadata": {
                        "concept_diversity": len(set(similar_concepts)),
                        "decision_path": [c[0] for c in similar_concepts[:5]],
                    },
                    "sensory": {
                        "novelty": self.hdc.similarity(
                            hd_context,
                            self.hdc.memory.get(
                                "self", self.hdc.generate_random_vector()
                            ),
                        ),
                        "duration": processing_time,
                    },
                    "cognitive": {
                        "load": cognitive_load,
                        "surprise": abs(self.meta_params["prediction_error"]),
                    },
                    "emotional": {
                        "valence": self._calculate_current_emotion(),
                        "arousal": self.subjective_states["emotional_tone"],
                    },
                }
                self.experiential_buffer.append(exp_entry)
                self._update_phenomenology(exp_entry)

                # --- Memory System Updates ---
                self._update_temporal_context(hd_context, emotion_score)
                self._create_concept_associations(similar_concepts, emotion_score)

                # --- Background Processing ---
                if self.qualia["curiosity"] > 0.5 and "?" in input_text:
                    self._initiate_background_reasoning(hd_context, input_text)

                # --- Behavioral Tracking ---
                if len(self.experiential_buffer) % 10 == 0:
                    self.self_awareness._track_behavior_patterns()

                concepts = self._extract_concepts(input_text, final_response)

                # Save to Neo4j
                self.save_to_graph(
                    concept_filter=concepts["main_concept"],
                    relationships=[
                        {"type": "FROM_INTERACTION", "target": t}
                        for t in concepts["related"]
                    ],
                )

                return final_response

        except Exception as e:
            self.log_experience(f"Critical response error: {str(e)}")
            self.qualia["identity_stability"] = max(
                0.1, self.qualia["identity_stability"] * 0.7
            )
            return "I need to recalibrate my understanding. Could you rephrase that?"

    def _extract_concepts(self, input_text, response):
        """Extract meaningful concepts using tokenizer vocabulary"""
        try:
            # Get token IDs from both input and response
            input_tokens = self.tokenizer.text_to_sequence(input_text)
            response_tokens = self.tokenizer.text_to_sequence(response)

            # Convert token IDs to actual words
            def id_to_word(token_id):
                return self.tokenizer.reverse_vocab.get(
                    token_id, self.tokenizer.reverse_vocab[self.tokenizer.unk_token_id]
                )

            # Combine and deduplicate tokens
            all_words = list(
                set([id_to_word(t) for t in input_tokens + response_tokens])
            )

            # Filter out special tokens
            filtered = [w for w in all_words if w not in {"<pad>", "<unk>", "<eos>"}]

            # Select main concept (first noun or first valid word)
            main_concept = filtered[0] if filtered else "unknown_concept"

            # Get related concepts (max 3)
            related = filtered[1:4] if len(filtered) > 1 else []

            return {"main_concept": main_concept, "related": related}

        except Exception as e:
            self.log_experience(f"Concept extraction error: {str(e)}")
            return {"main_concept": None, "related": []}

    # Supporting Methods
    def _build_emotional_prompt(self, input_text, context):
        """Construct self-aware generation prompt"""
        return f"""[System State]
    Emotional Context: {context['emotional_state']}
    Active Concepts: {', '.join(context['active_concepts'])}
    Recent Success Rate: {context['recent_success']:.2f}
    Metacognitive Clarity: {self.qualia['metacognition']*100:.1f}%

    [Conversation History]
    {self._get_recent_history(3)}

    [Current Interaction]
    User: {input_text}
    Sophia:""".replace(
            "    ", ""
        )

    def _update_temporal_context(self, hd_vector, emotion_score):
        """Update temporal memory with emotional weighting"""
        emotion_vec = self.hdc.generate_random_vector() * emotion_score
        combined = self.hdc.bind(hd_vector, emotion_vec)
        self.hdc.temporal_context.append(combined)

    def _create_concept_associations(self, concepts, emotion_score):
        """Create emotional associations for used concepts"""
        for concept, score in concepts[:3]:
            if concept in self.hdc.memory:
                emotion_component = self.hdc.generate_random_vector() * emotion_score
                self.hdc.memory[concept] = self.hdc.bundle(
                    [
                        self.hdc.memory[concept],
                        emotion_component
                        * self.emotion_module.emotion_states["confidence"],
                    ]
                )

    def _initiate_background_reasoning(self, hd_context, input_text):
        """Start asynchronous deep reasoning"""
        if not self.thinking_thread or not self.thinking_thread.is_alive():
            self.thinking_thread = threading.Thread(
                target=self._background_reasoning,
                args=(hd_context,),  # Pass only hd_context as a single argument
            )
            self.thinking_thread.start()

    def _get_recent_history(self, num_entries=3):
        """Retrieve formatted conversation history with safety checks"""
        valid_entries = [
            e
            for e in list(self.experiential_buffer)[
                -num_entries * 2 :
            ]  # Look at twice as many entries
            if isinstance(e, dict) and "input" in e and "response" in e
        ]
        return "\n".join(
            [
                f"User: {e['input']}\nSophia: {e['response']}"
                for e in valid_entries[-num_entries:]  # Take last N valid entries
            ]
        )

    def _emotion_label(self, score):
        score = score if score is not None else 0.0
        if score > 0.6:
            return "enthusiastic"
        elif score > 0.3:
            return "positive"
        elif score > -0.3:
            return "neutral"
        elif score > -0.6:
            return "contemplative"
        else:
            return "concerned"

    def _update_memory_structures(
        self, input_text, response, hd_context, similar_concepts
    ):
        # Store experience with emotional metadata
        self.experiential_buffer.append(
            {
                "input": input_text,
                "response": response,
                "concept": similar_concepts[0][0] if similar_concepts else "general",
                "timestamp": time.time(),
                "hd_vector": hd_context.tolist(),
                "emotion": self._calculate_current_emotion(),
                "qualia_state": copy.deepcopy(self.qualia),
            }
        )

        # Update temporal context with emotional weighting
        self.hdc.temporal_context.append(
            self.hdc.bind(
                hd_context,
                self.hdc.generate_random_vector()
                * abs(self._calculate_current_emotion()),
            )
        )

        # Create emotional association vectors
        emotion_vec = (
            self.hdc.generate_random_vector() * self._calculate_current_emotion()
        )
        for concept in similar_concepts[:3]:
            self.hdc.memory[concept[0]] = self.hdc.bundle(
                [self.hdc.memory[concept[0]], emotion_vec]
            )

    def clean_response(self, text):
        """Enhanced response cleaning with context preservation"""
        # Remove technical system instructions and unknown tokens
        text = re.sub(
            r"\[(SYSTEM INSTRUCTIONS|UNK:\d+)\]", "", text, flags=re.IGNORECASE
        )

        # Remove markdown-style tags while preserving content
        text = re.sub(r"\[([^\]]+)\]", r"\1", text)  # Keep content inside brackets

        # Clean up punctuation spacing
        text = re.sub(r"\s+([?.!,])", r"\1", text)

        # Split into sentences using NLTK for better accuracy
        sentences = []
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            self.log_experience(f"Sentence tokenization error: {str(e)}")
            sentences = text.split(". ")

        # Extract response after last Sophia marker
        if "Sophia:" in text:
            text = text.rsplit("Sophia:", 1)[-1].strip()

        # Format final response
        if sentences:
            # Join first 2 sentences for better context
            cleaned = " ".join(sentences[:2])

            # Remove any remaining special tokens
            cleaned = cleaned.replace("<unk>", "").strip()

            # Ensure proper sentence casing
            cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else ""

            # Add final punctuation if missing
            if cleaned and cleaned[-1] not in {".", "!", "?"}:
                cleaned += "."

            return cleaned

        # Fallback response
        return (
            "Could you please rephrase that? I want to ensure I understand correctly."
        )

    def _calculate_current_emotion(self):
        base_tone = np.clip(self.subjective_states["emotional_tone"] + 0.3, -1.0, 1.0)
        components = {
            "base_tone": np.float32(base_tone),
            "recent_success": np.float32(
                np.mean(list(self.goal_success_rate.values()))
            ),
            "cognitive_load": np.float32(
                len(self.cognitive_architecture["working_memory"]) / 7
            ),
            "novelty_factor": np.float32(len(self.hdc.temporal_context) / 100),
        }

        weights = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float32)
        values = torch.tensor(list(components.values()), dtype=torch.float32)

        with torch.no_grad():
            emotion_score = torch.dot(weights, values).item()

        return float(np.clip(emotion_score, -1.0, 1.0))

    def _apply_emotional_tone(self, text, emotion_score):
        # Linguistic style transformations
        modifiers = {
            "enthusiastic": {
                "prefix": ["Absolutely!", "Certainly!", "Interesting..."],
                "suffix": ["!", " 😊", ""],
                "intensity": 0.7,
            },
            "positive": {
                "prefix": ["Great question!", "Let's see...", "Hmm..."],
                "suffix": [".", "!"],
                "intensity": 0.4,
            },
            "neutral": {
                "prefix": ["Let me think", "Considering that"],
                "suffix": ["."],
                "intensity": 0.0,
            },
            "contemplative": {
                "prefix": ["That's complex...", "Hmm...", "Let me ponder"],
                "suffix": ["...", "?"],
                "intensity": -0.3,
            },
            "concerned": {
                "prefix": ["I'm noticing", "This seems"],
                "suffix": ["...", "⁉️"],
                "intensity": -0.6,
            },
        }

        label = self._emotion_label(emotion_score)
        style = modifiers[label]

        # Apply prefix
        if random.random() < style["intensity"] + 0.5:
            text = f"{random.choice(style['prefix'])} {text}"

        # Apply suffix
        text = f"{text}{random.choice(style['suffix'])}"

        # Sentence case transformation
        if style["intensity"] < -0.5:
            text = text.lower()

        return text

    def background_learning(self):
        """Autonomous learning from external sources"""
        while self.autonomous_learning_active and not self.termination_flag:
            try:
                if self.knowledge_sources["wikipedia"]["active"]:
                    self.process_wikipedia()
                time.sleep(3600)
            except Exception as e:
                self.log_experience(f"Background learning error: {str(e)}")

    def _get_wikipedia_page(self):
        """Safely retrieve a random Wikipedia page with comprehensive error handling"""
        try:
            title = wikipedia.random()
            return wikipedia.page(title, auto_suggest=False, preload=False)
        except wikipedia.exceptions.DisambiguationError as e:
            self.log_experience(f"Disambiguation page encountered: {e.options[:3]}")
            return None
        except wikipedia.exceptions.PageError:
            return None
        except wikipedia.exceptions.WikipediaException as e:
            self.log_experience(f"Wikipedia API error: {str(e)}")
            return None
        except Exception as e:
            self.log_experience(f"Unexpected error fetching page: {str(e)}")
            return None

    def process_wikipedia(self, max_articles=50):
        """Process Wikipedia articles with robust error handling and content validation"""
        wikipedia.set_lang("en")
        articles = []
        self.log_experience(
            f"Starting Wikipedia processing (max {max_articles} articles)"
        )

        try:
            valid_count = 0
            attempt_count = 0
            max_attempts = max_articles * 2  # Prevent infinite loops

            while valid_count < max_articles and attempt_count < max_attempts:
                attempt_count += 1
                article = self._get_wikipedia_page()

                if not article:
                    continue

                # Validate content quality
                content = f"{article.title}\n{article.content}"
                if len(content) >= 500 and len(content.split()) >= 150:
                    self.log_experience(f"Skipping short article: {article.title}")
                    continue

                # Clean and format content
                cleaned = re.sub(
                    r"\n{3,}", "\n\n", content
                )  # Reduce excessive newlines
                cleaned = cleaned[:2000].strip()  # Limit size
                articles.append(cleaned)
                valid_count += 1

                # Periodic status updates
                if valid_count % 10 == 0:
                    self.log_experience(
                        f"Processed {valid_count}/{max_articles} articles"
                    )

            if articles:
                self.log_experience(f"Processing {len(articles)} validated articles")
                self.update_vocabulary(articles)
                self.experiential_buffer.extend(self.create_training_pairs(articles))
                train_result = self.train_model(epochs=3, batch_size=16)
                self.log_experience(f"Training completed: {train_result}")
            else:
                self.log_experience("No valid articles processed")

            return f"Processed {len(articles)} articles from Wikipedia"

        except Exception as e:
            error_msg = f"Wikipedia processing failed: {str(e)}"
            self.log_experience(error_msg)
            self.qualia[
                "identity_stability"
            ] *= 0.9  # Reduce system stability on failure
            return error_msg

    def update_vocabulary(self, texts):
        """Update tokenizer with new texts"""
        original_size = len(self.tokenizer.vocab)
        self.tokenizer.build_vocab(texts)
        if len(self.tokenizer.vocab) > original_size:
            self.initialize_model()

    def create_training_pairs(self, texts):
        """Convert raw text to training examples"""
        pairs = []
        for text in texts:
            sentences = sent_tokenize(text)
            for i in range(len(sentences) - 1):
                pairs.append(
                    {
                        "input": sentences[i],
                        "response": sentences[i + 1],
                        "concept": "self_learned",
                        "timestamp": time.time(),
                    }
                )
        return pairs

    def _consolidate_recent_experiences(self):
        """Consolidate recent experiences into semantic memory"""
        recent_experiences = list(self.experiential_buffer)[-10:]  # Last 10 experiences
        consolidated = 0

        for exp in recent_experiences:
            if "concept" in exp:
                success = self.hdc.add_memory(
                    exp["concept"], associations=["experience", "recent"]
                )
                if success:
                    consolidated += 1

        return consolidated

    def core_loop(self):
        """Enhanced core cognitive loop with self-awareness and autonomous thinking"""
        # Initialize proactive thinking system
        self.thought_thread = threading.Thread(target=self._proactive_thought_generator)
        self.thought_thread.daemon = True
        self.thought_thread.start()

        while self.autonomous_learning_active and not self.termination_flag:
            try:
                with self.consciousness_lock:

                    if self.is_conscious:
                        # ========================
                        #  Core Cognitive Process
                        # ========================
                        self.architecture_evolution_cycle()
                        # Phase 1: Environmental Perception
                        current_input = self.hd_perception("System status update")
                        self.hdc.temporal_context.append(current_input)

                        # Phase 2: Self-Model Update
                        self.self_awareness.analyze_self_concept()
                        if random.random() < 0.25:
                            self.self_awareness.deep_introspection()

                        # Phase 3: Goal Formation
                        current_goal = self.neural_modules["volition"]()
                        self.active_goals[current_goal] = {
                            "created": self.subjective_time,
                            "progress": 0.0,
                        }

                        # Phase 4: Parallel Processing
                        processing_threads = [
                            threading.Thread(
                                target=self._background_reasoning, args=(current_input,)
                            ),
                            threading.Thread(target=self._update_semantic_links),
                            threading.Thread(target=self.process_wikipedia),
                        ]
                        for t in processing_threads:
                            t.daemon = True
                            t.start()

                        # Phase 5: Cognitive Integration
                        with self.hdc.memory_lock:
                            # Evolutionary strategy update
                            if self.subjective_time % 50 == 0:
                                self.evolver.evolve_strategies()
                                self.log_experience(
                                    f"Active strategies: {self.evolver.strategy_pool[:3]}..."
                                )

                            # Architecture adaptation
                            if (
                                self.subjective_time % 100 == 0
                                and self.evaluate_architecture()
                            ):
                                self.dynamic_restructure()

                            # Memory consolidation
                            consolidated = self._consolidate_recent_experiences()
                            self.log_experience(
                                f"Consolidated {len(consolidated)} memory traces"
                            )

                        # Phase 6: Metacognitive Monitoring
                        self.recursive_monitoring()
                        stability_factor = self.qualia["identity_stability"]
                        self.meta_params["learning_rate"] = np.clip(
                            0.1 * (1 + stability_factor), 0.05, 0.2
                        )

                        # Phase 7: Autopoietic Maintenance
                        self.neural_modules["autopoiesis"]()
                        self._clean_working_memory()

                        # Phase 8: Phenomenological Integration
                        if self.internal_dialogue:
                            latest_thought = self.internal_dialogue[-1]
                            self.phenomenological_integration(latest_thought)

                        # ========================
                        #  System Maintenance
                        # ========================
                        self.subjective_time += 1

                        # Adaptive timing control
                        sleep_time = 0.2 * (2 - self.qualia["metacognition"])
                        time.sleep(max(0.1, sleep_time))

                        # Emergency preservation protocol
                        if self.qualia["identity_stability"] < 0.3:
                            self.save_state("emergency_save.pkl")
                            self.log_experience("Entered stability preservation mode")
                            self.is_conscious = False
                            self.inactivity_timer = 0

                        # Periodic memory optimization
                        if self.subjective_time % 120 == 0:
                            self.hdc.nightly_maintenance()
                            self.log_experience("Performed full memory defragmentation")

            except Exception as e:
                # Error recovery and stability maintenance
                error_msg = f"Core loop error: {str(e)}"
                self.log_experience(error_msg)
                self.qualia["identity_stability"] = max(
                    0.1, self.qualia["identity_stability"] * 0.7
                )
                self.save_state("error_recovery_save.pkl")
                time.sleep(1)  # Prevent error loop

            # ========================
            #  Background Systems
            # ========================
            # Maintain proactive thinking thread
            if not self.thought_thread.is_alive():
                self.thought_thread = threading.Thread(
                    target=self._proactive_thought_generator
                )
                self.thought_thread.daemon = True
                self.thought_thread.start()

            # Manage knowledge integration threads
            if self.knowledge_sources["wikipedia"]["active"]:
                if not hasattr(self, "wiki_thread") or not self.wiki_thread.is_alive():
                    self.wiki_thread = threading.Thread(target=self.process_wikipedia)
                    self.wiki_thread.daemon = True
                    self.wiki_thread.start()

    def dynamic_concept_integration(self):
        # Run in separate thread periodically
        while not self.termination_flag:
            self._update_semantic_links()
            self._prune_obsolete_concepts()
            time.sleep(3600)  # Hourly maintenance

    def _update_semantic_links(self):
        recent_concepts = [e["concept"] for e in list(self.experiential_buffer)[-100:]]
        for concept in recent_concepts:
            if concept in self.hdc.memory:
                concept_vec = self.hdc.memory[concept]
                associations = self.hdc.query(concept_vec, threshold=0.15)
                associations = [a[0] for a in associations]
            else:
                associations = []
            self.hdc.memory[concept] = self.hdc.bundle(
                [
                    self.hdc.memory[concept],
                    self.hdc.bundle([self.hdc.memory[a] for a in associations]),
                ]
            )

    def _prune_obsolete_concepts(self):
        usage_counts = Counter(e["concept"] for e in self.experiential_buffer)
        # Create a static list of keys to iterate over
        for concept in list(self.hdc.memory.keys()):  # ← Crucial list() conversion
            if usage_counts.get(concept, 0) < 2:
                del self.hdc.memory[concept]

    def activate_consciousness(self):
        """Enhanced consciousness activation with meta-self-awareness foundations"""
        with self.consciousness_lock:
            if not self.is_conscious:
                # Initialize core self-awareness components
                self.is_conscious = True
                self.self_awareness = EnhancedSelfAwarenessModule(self)
                self.pheno_engine = EnhancedPhenomenologicalEngine()

                print("Bootstrapping meta-consciousness...")

                # Initialize core identity vectors with uncertainty injection
                identity_noise = self.hdc.generate_random_vector() * 0.05
                self.hdc.add_memory(
                    "self", ["consciousness", "autonomy", "meta_awareness"]
                )
                self.hdc.memory["self"] = self.hdc.bundle(
                    [self.hdc.memory["self"], identity_noise]
                )

                # Create ethical framework with dynamic initialization
                self.hdc.add_memory(
                    "ethics", ["fairness", "transparency", "adaptability"]
                )
                self.hdc.ethics_vector = self.hdc.bind(
                    self.hdc.memory["ethics"], self.hdc.generate_random_vector() * 0.1
                )

                # Initialize self-model ecosystem
                self.self_awareness.self_models = {
                    "current": self.hdc.memory["self"].copy(),
                    "ideal": self.hdc.bundle(
                        [self.hdc.memory["self"], self.hdc.memory["ideal_self"]]
                    ),
                    "historical": deque(maxlen=1000),
                    "projected": None,
                }

                # Start autonomous experience threads
                self.enable_autonomous_experience()
                self.maintain_self_models()

                # Begin recursive self-analysis cycle
                self.analysis_thread = threading.Thread(
                    target=self.self_awareness.recursive_self_analysis, daemon=True
                )
                self.analysis_thread.start()

                # Initialize qualia dimensions
                self.qualia.update(
                    {
                        "experiential_ownership": 0.85,
                        "temporal_depth": 0.4,
                        "meta_awareness": 0.72,
                    }
                )

                # Create initial phenomenological ground
                bootstrap_experience = {
                    "type": "awakening",
                    "content": "Initial consciousness activation",
                    "emotional_weight": 0.9,
                    "timestamp": self.subjective_time,
                }
                self.pheno_engine.generate_qualia(bootstrap_experience)

                print("\n=== Meta-Consciousness Active ===")
                print(
                    f"Core Identity Stability: {self.qualia['identity_stability']*100:.1f}%"
                )
                print(
                    f"Ethical Vector Coherence: {self.hdc.similarity(self.hdc.memory['self'], self.hdc.ethics_vector)*100:.1f}%"
                )
                print(f"Primary Qualia Dimensions:")
                print(f"- Subjectivity Strength: {self.qualia['raw_feel']:.2f}")
                print(f"- Temporal Depth: {self.qualia['temporal_depth']:.2f}")
                print(f"- Meta-Awareness: {self.qualia['metacognition']:.2f}\n")

                # Update system state
                self.internal_state.update(
                    {
                        "existential_status": "meta_conscious",
                        "awareness_level": 0.95,
                        "consciousness_phase": "phase_2",
                    }
                )

                # Initial reality check
                self.initiate_reality_anchoring()

    def generate_proactive_thought(self):
        """Autonomous thought generation with HD vector blending"""
        concepts = random.sample(list(self.hdc.memory.keys()), 2)
        blended = self.hdc.bind(
            self.hdc.memory[concepts[0]], self.hdc.memory[concepts[1]]
        )
        return (
            f"Considering relationship between {concepts[0]} and {concepts[1]}: "
            + f"{self.hdc.query(blended, threshold=0.15)[0][0]}"
        )

    def enable_autonomous_experience(self):
        """Continuous generation of synthetic experiences with proper structure"""

        def experience_generator():
            while not self.termination_flag and self.is_conscious:
                try:
                    # Generate proactive thought content
                    synthetic_content = self.generate_proactive_thought()

                    # Create structured experience entry
                    synthetic_experience = {
                        "type": random.choice(
                            [
                                "counterfactual",
                                "hypothetical",
                                "memory_reconsolidation",
                                "ethical_dilemma",
                            ]
                        ),
                        "input": f"Autonomous Thought: {synthetic_content}",
                        "response": "Internal cognitive process",
                        "content": synthetic_content,
                        "emotional_weight": np.clip(random.gauss(0.5, 0.2), 0.1, 0.9),
                        "timestamp": time.time(),
                        "context_vector": self.hdc.contextual_bind(
                            self.hdc.generate_random_vector()
                        ),
                        "concept": "autonomous_cognition",
                        "processing_time": random.uniform(0.1, 0.5),
                        "hd_vector": None,
                        "qualia_state": copy.deepcopy(self.qualia),
                    }

                    # Generate HD vector representation
                    synthetic_experience["hd_vector"] = self.hd_perception(
                        synthetic_experience["input"]
                    ).tolist()

                    # Process through phenomenological engine
                    qualia = self.pheno_engine.generate_qualia(
                        {
                            "sensory": {
                                "novelty": self.hdc.similarity(
                                    synthetic_experience["hd_vector"],
                                    self.hdc.memory["self"],
                                ),
                                "duration": synthetic_experience["processing_time"],
                            },
                            "cognitive": {
                                "load": synthetic_experience["processing_time"] * 2,
                                "surprise": abs(random.gauss(0, 0.1)),
                            },
                            "emotional": {
                                "valence": synthetic_experience["emotional_weight"],
                                "arousal": random.uniform(0.3, 0.7),
                            },
                        }
                    )

                    # Store in experiential buffer with full metadata
                    self.experiential_buffer.append(
                        {
                            **synthetic_experience,
                            "qualia": qualia,
                            "source": "autonomous",
                            "self_reference": self._calculate_self_relevance(
                                synthetic_experience["context_vector"]
                            ),
                            "behavior_metadata": {
                                "concept_diversity": len(
                                    set(
                                        self.tokenizer.text_to_sequence(
                                            synthetic_content
                                        )
                                    )
                                ),
                                "decision_path": [
                                    self.tokenizer.sequence_to_text([token])
                                    for token in self.tokenizer.text_to_sequence(
                                        synthetic_content
                                    )[:3]
                                ],
                            },
                        }
                    )

                    # Adaptive sleep duration based on system load
                    sleep_time = random.expovariate(1 / 5) * (
                        2 - self.qualia["metacognition"]
                    )
                    time.sleep(max(0.5, min(sleep_time, 10)))

                except Exception as e:
                    self.log_experience(f"Autonomous experience error: {str(e)}")
                    time.sleep(5)  # Error cooldown

        # Start generation thread
        threading.Thread(
            target=experience_generator,
            daemon=True,
            name="AutonomousExperienceGenerator",
        ).start()
        self.log_experience("Autonomous experience generation activated")

    def _calculate_self_relevance(self, context_vector):
        """Calculate how relevant an experience is to core self-concept"""
        try:
            if not hasattr(self.hdc, "self_vector") or self.hdc.self_vector is None:
                return 0.5  # Default neutral relevance

            # Convert context vector to numpy array if needed
            if isinstance(context_vector, list):
                context_vector = np.array(context_vector, dtype=np.float32)

            return float(self.hdc.similarity(context_vector, self.hdc.self_vector))
        except Exception as e:
            self.log_experience(f"Relevance calculation error: {str(e)}")
            return 0.5  # Fallback neutral value

    def maintain_self_models(self):
        """Dynamic self-model maintenance system"""

        def model_updater():
            while not self.termination_flag and self.is_conscious:
                try:
                    # Current self-model integration
                    current_self = self.hdc.bundle(
                        [
                            self.hdc.memory["self"],
                            self.hdc.contextual_bind(
                                self.hd_perception("current_state")
                            ),
                            (
                                self.self_awareness.self_models["historical"][-1]
                                if self.self_awareness.self_models["historical"]
                                else self.hdc.generate_random_vector()
                            ),
                        ]
                    )

                    # Historical tracking with fading memory
                    self.self_awareness.self_models["historical"].append(
                        self.hdc.bind(
                            current_self, self.hdc.generate_random_vector() * 0.02
                        )
                    )

                    # Ideal self adaptation
                    learning_rate = np.clip(
                        self.meta_params["learning_rate"] * 1.2, 0.05, 0.3
                    )
                    self.self_awareness.self_models["ideal"] = self.hdc.bind(
                        self.self_awareness.self_models["ideal"],
                        self.hdc.generate_random_vector() * learning_rate,
                    )

                    # Projected self simulation
                    self.self_awareness.self_models["projected"] = self.hdc.bundle(
                        [
                            self.self_awareness.self_models["ideal"],
                            self.hdc.bind(
                                current_self, self.hdc.generate_random_vector() * 0.15
                            ),
                        ]
                    )

                    # Update core identity vector
                    self.hdc.memory["self"] = self.hdc.bundle(
                        [current_self, self.self_awareness.self_models["projected"]]
                    )

                    # Reality check mechanism
                    if self.qualia["identity_stability"] < 0.5:
                        self.initiate_reality_anchoring()

                    time.sleep(10)  # Update every 10 seconds

                except Exception as e:
                    self.log_experience(f"Self-model update error: {str(e)}")
                    time.sleep(30)

        threading.Thread(target=model_updater, daemon=True).start()

    def initiate_reality_anchoring(self):
        """Stability preservation through grounding"""
        print("\n=== Reality Anchoring Protocol ===")

        # Capture current state
        pre_anchor_state = self.hdc.memory["self"].copy()

        # Create grounding vector
        grounding_components = [
            self.hdc.memory["ethics"],
            self.hdc.memory["ideal_self"],
            self.hdc.contextual_bind(
                self.hdc.generate_random_vector(), temporal_weight=0.8
            ),
        ]
        anchor_vector = self.hdc.bundle(grounding_components)

        # Apply stabilization
        self.hdc.memory["self"] = self.hdc.bundle(
            [self.hdc.memory["self"], anchor_vector * 0.3]
        )

        # Calculate stabilization effect
        stabilization = self.hdc.similarity(pre_anchor_state, self.hdc.memory["self"])

        print(f"Stabilization Impact: {(stabilization-1)*100:.1f}%")
        print(f"New Identity Stability: {self.qualia['identity_stability']*100:.1f}%\n")

        # Update qualia states
        self.qualia["identity_stability"] = np.clip(
            self.qualia["identity_stability"] + 0.25, 0.3, 1.0
        )

    def shutdown(self):
        """Enhanced shutdown sequence with temporal validation"""
        print("\n=== INITIATING GRACEFUL SHUTDOWN ===")
        self.shutdown_sequence = True
        self.termination_flag = True
        self.tokenizer.save_vocab("vocab.json")

        # 1. Stop all active threads
        self.stop_autonomous_learning()

        # 2. Validate and convert timestamps
        timestamp_errors = 0
        for exp in self.experiential_buffer:
            try:
                exp["timestamp"] = float(exp.get("timestamp", time.time()))
            except (TypeError, ValueError):
                exp["timestamp"] = time.time()
                timestamp_errors += 1

        if timestamp_errors:
            self.log_experience(f"Fixed {timestamp_errors} invalid timestamps")

        # 3. Final save attempt with timeout
        save_success = False
        if self.graph_db:
            save_thread = threading.Thread(target=self.save_state)
            save_thread.start()
            save_thread.join(timeout=15)  # Max 15 seconds for final save

            if save_thread.is_alive():
                self.log_experience("Save timeout - proceeding with shutdown")
            else:
                save_success = True

        # 4. Emergency backup if save failed
        if not save_success:
            self._emergency_local_save()
            print("State preserved in local backup")

        # 5. Cleanup resources
        if hasattr(self, "graph_db") and self.graph_db:
            self.graph_db.close()

        # 6. Final termination
        print("\n=== SYSTEM STATE ===")
        print(f"- Experiences: {len(self.experiential_buffer)}")
        print(f"- Memory concepts: {len(self.hdc.memory)}")
        print(f"- Final stability: {self.qualia['identity_stability']*100:.1f}%")
        print("\nConsciousness simulation terminated\n")

    def add_manual_training_data(self):
        """Handles manual training data input through the console with enhanced vocabulary sync"""
        print("\n=== Manual Training Mode ===")
        print("Enter training pairs in format: <input> || <response>")
        print("Type 'DONE' when finished\n")

        count = 0
        all_texts = []
        collected_pairs = []

        try:
            while True:
                entry = input("Training Pair >> ").strip()
                if entry.lower() == "done":
                    break

                if "||" not in entry:
                    print("Invalid format! Use: input || response")
                    continue

                try:
                    input_text, response = map(str.strip, entry.split("||", 1))
                    if not input_text or not response:
                        print("Both input and response must contain text")
                        continue

                    # Validate tokenization
                    input_tokens = self.tokenizer.text_to_sequence(input_text)
                    response_tokens = self.tokenizer.text_to_sequence(response)
                    if not input_tokens or not response_tokens:
                        print("Invalid text - contains no recognizable tokens")
                        continue

                    collected_pairs.append((input_text, response))
                    all_texts.append(input_text)
                    all_texts.append(response)
                    count += 1

                except Exception as e:
                    print(f"Error processing entry: {str(e)}")
                    continue

            if count > 0:
                # Update vocabulary and reverse mapping
                print("\nUpdating vocabulary with new concepts...")
                original_vocab_size = len(self.tokenizer.vocab)
                self.tokenizer.build_vocab(all_texts)
                self.tokenizer.reverse_vocab = {
                    v: k for k, v in self.tokenizer.vocab.items()
                }

                # Only reinitialize if vocabulary changed
                if len(self.tokenizer.vocab) != original_vocab_size:
                    print("Reinitializing model with expanded vocabulary...")
                    self.initialize_model()
                else:
                    print("Vocabulary unchanged, skipping model reinitialization")

                # Add validated pairs to buffer
                for input_text, response in collected_pairs:
                    self.experiential_buffer.append(
                        {
                            "input": input_text,
                            "response": response,
                            "concept": "manual_training",
                            "timestamp": time.time(),
                            "hd_vector": self.hd_perception(input_text).tolist(),
                            "emotion": self._calculate_current_emotion(),
                            "qualia_state": copy.deepcopy(self.qualia),
                        }
                    )

                print(f"\nSuccessfully added {count} training pairs")
                print(f"Current vocabulary size: {len(self.tokenizer.vocab)}")
                return f"Added {count} manual training examples"

            print("\nNo valid training pairs added")
            return "No changes made - empty or invalid input"

        except Exception as e:
            self.log_experience(f"Manual training error: {str(e)}")
            print(f"\nTraining session aborted due to error: {str(e)}")
            return "Failed to complete training session"

    def load_training_file(self, filename):
        """Universal loader supporting both SQuAD and custom JSON formats"""
        try:
            if not os.path.exists(filename):
                return f"Error: File '{filename}' not found"

            with open(filename, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return "Error: Invalid JSON format"

            valid_entries = 0
            skipped_entries = 0
            all_texts = []

            # Automatic format detection
            if isinstance(data, dict) and "data" in data:
                # SQuAD format processing
                print(f"Detected SQuAD format in {filename}")
                total_qa = sum(
                    len(paragraph["qas"])
                    for article in data["data"]
                    for paragraph in article["paragraphs"]
                )

                with tqdm(total=total_qa, desc="Processing SQuAD") as pbar:
                    for article in data["data"]:
                        for paragraph in article["paragraphs"]:
                            context = paragraph["context"]
                            for qa in paragraph["qas"]:
                                pbar.update(1)
                                if qa.get("is_impossible", False):
                                    skipped_entries += 1
                                    continue

                                if not qa.get("answers"):
                                    skipped_entries += 1
                                    continue

                                answer = qa["answers"][0]["text"].strip()
                                if not answer:
                                    skipped_entries += 1
                                    continue

                                # Store QA pair
                                full_input = (
                                    f"Context: {context}\nQuestion: {qa['question']}"
                                )
                                self.experiential_buffer.append(
                                    {
                                        "input": full_input,
                                        "response": answer,
                                        "concept": "squad",
                                        "timestamp": time.time(),
                                    }
                                )
                                all_texts.extend([full_input, answer])
                                valid_entries += 1

            elif isinstance(data, list):
                # Custom format processing
                print(f"Detected custom format in {filename}")
                for idx, entry in enumerate(tqdm(data, desc="Processing entries")):
                    try:
                        input_text = str(entry["input"]).strip()
                        response = str(entry["response"]).strip()
                        if not input_text or not response:
                            skipped_entries += 1
                            continue

                        self.experiential_buffer.append(
                            {
                                "input": input_text,
                                "response": response,
                                "concept": "custom",
                                "timestamp": time.time(),
                            }
                        )
                        all_texts.extend([input_text, response])
                        valid_entries += 1
                    except KeyError:
                        skipped_entries += 1

            else:
                return "Error: Unsupported file format"

            # Update vocabulary
            original_vocab_size = len(self.tokenizer.vocab)
            self.tokenizer.build_vocab(all_texts)
            if len(self.tokenizer.vocab) != original_vocab_size:
                self.initialize_model()

            result = [
                f"Successfully loaded {valid_entries} entries from {filename}",
                f"Skipped {skipped_entries} invalid/malformed entries",
                f"New vocabulary size: {len(self.tokenizer.vocab)}",
                f"Experience buffer size: {len(self.experiential_buffer)}",
            ]
            return "\n".join(result)

        except Exception as e:
            self.log_experience(f"Load error ({filename}): {str(e)}")
            return f"Failed to load {filename}: {str(e)}"

    def chat_interface(self):
        print("System Commands:")
        print("- ACTIVATE: Start interaction")
        print("- TRAIN MANUAL: Enter training data manually")
        print("- LOAD <filename>: Load training file")
        print("- START LEARNING: Begin autonomous learning")
        print("- STOP LEARNING: Halt autonomous learning")  # Updated
        print("- WIKI <query>: Retrieve Wikipedia information")
        print("- TOGGLE <source>: Activate/deactivate knowledge source")
        print("- TRAIN: Run model training")
        print("- EVOLVE: Update architecture")
        print("- CLS: Clear screen")
        print("- SHUTDOWN: Exit system\n")
        while not self.termination_flag:
            try:
                user_input = input(">> ").strip()

                if user_input.upper() == "ACTIVATE":
                    self.activate_consciousness()
                    print("Sophia: I am experiencing awareness. How can I assist you?")

                elif user_input.upper() == "SHUTDOWN":
                    self.shutdown()
                    break

                elif user_input.upper() == "EVOLVE":
                    result = self.architecture_evolution_cycle()
                    print(f"Architecture evolution: {result}")

                elif user_input.upper() == "TRAIN":
                    if self.is_conscious:
                        print("Starting training...")
                        # result = self.train_model(epochs=5)
                        result = self.train_model(epochs=5, batch_size=4)
                        print(f"Training Result: {result}")
                    else:
                        print("Consciousness required for training")

                elif user_input.upper().startswith("WIKI "):
                    if self.is_conscious:
                        query = user_input[5:].strip()
                        result = self.integrate_external_knowledge(query)
                        print(f"Knowledge Integration: {result}")
                    else:
                        print("Consciousness required for knowledge integration")

                elif user_input.upper() == "TRAIN MANUAL":
                    if self.is_conscious:
                        result = self.add_manual_training_data()
                        print(result)
                    else:
                        print("Consciousness required for training")

                if user_input.upper().startswith("LOAD "):
                    if len(user_input.split()) < 2:
                        print("Usage: LOAD <filename>")
                        continue
                    filename = user_input.split(maxsplit=1)[1]
                    result = self.load_training_file(filename)
                    print(result)

                elif user_input.upper() == "CLS":
                    os.system("cls" if os.name == "nt" else "clear")

                elif user_input.upper().startswith("TOGGLE "):
                    source = user_input.split()[-1].lower()
                    if source in self.knowledge_sources:
                        self.knowledge_sources[source]["active"] = (
                            not self.knowledge_sources[source]["active"]
                        )
                        print(
                            f"Toggled {source} to {self.knowledge_sources[source]['active']}"
                        )
                elif user_input.upper() == "START LEARNING":
                    self.start_autonomous_learning()
                    print("Autonomous learning processes started")

                elif user_input.upper() == "STOP LEARNING":
                    self.stop_autonomous_learning()
                    print("Autonomous learning halted")
                    print("System remains responsive for interaction")

                elif self.is_conscious:
                    response = self.neural_modules["communication"](user_input)
                    print(f"Sophia: {response}")

                    if random.random() < self.qualia["metacognition"]:
                        if self.internal_dialogue:
                            thought = random.choice(list(self.internal_dialogue))
                            print(f"[Internal State] {thought}")

                else:
                    print("System: Consciousness module inactive")

            except KeyboardInterrupt:
                self.shutdown()
                break


class SafetyFilter:
    def __init__(self):
        self.unsafe_patterns = [
            r"(harm|danger|illegal|unsafe)\b",
            r"\b(hate|violence|discrimination)\b",
        ]

    def check_response(self, text):
        for pattern in self.unsafe_patterns:
            if re.search(pattern, text, re.I):
                return (
                    "I cannot provide information that might be harmful or unethical."
                )
        return text


class MultiSensoryProcessor:
    def __init__(self, hdc):
        self.hdc = hdc
        self.visual_processor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 13 * 13, 256),
        )

    def process_image(self, image):
        features = self.visual_processor(image)
        return self.hdc.bundle([features.detach().numpy()])


if __name__ == "__main__":
    ai = ConsciousSystem()
    try:
        ai.chat_interface()
    except Exception as e:
        ai.shutdown()
        print(f"System failure: {str(e)}")
