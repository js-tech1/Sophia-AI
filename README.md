
# ConsciousSystem (Sophia) – Cognitive AI Architecture

## Overview

**ConsciousSystem (codename: Sophia)** is an advanced cognitive AI framework designed to simulate aspects of consciousness, self-awareness, and adaptive learning. It integrates neural, symbolic, and hyperdimensional memory systems, supporting autonomous reasoning, experience-driven learning, and dynamic self-organization.

## Architecture

### Core Class
- **ConsciousSystem** orchestrates all cognitive processes.

### Memory
- Uses a **Hyperdimensional Memory (HDM)** system for concept storage, semantic associations, and temporal context.

### Neural Modules
- Modular functions for reasoning, problem-solving, simulation, and self-inquiry.

### Qualia & Subjective States
- Tracks metacognition, emotional tone, and identity stability.

### Experience Buffer
- Stores structured experiences for reinforcement and continual learning.

### External Knowledge
- Integrates with sources like Wikipedia and supports graph-based long-term storage (Neo4j).

### Threaded Processes
- Background learning, proactive thought generation, and self-model maintenance run in parallel threads.

## Usage

### Start the System
1. Run the main script:
   ```bash
   python 12-05-25 V1.py
   ```
2. Use the interactive console interface.

### Console Commands

- **ACTIVATE**: Bootstraps consciousness and self-awareness.
- **TRAIN MANUAL**: Add training data via console.
- **LOAD <filename>**: Load training data from JSON/SQuAD files.
- **START LEARNING / STOP LEARNING**: Toggle autonomous background learning.
- **WIKI <query>**: Integrate knowledge from Wikipedia.
- **TRAIN**: Run model training on current experience buffer.
- **EVOLVE**: Trigger architecture evolution cycle.
- **SHUTDOWN**: Save state and exit safely.

### Interaction

Once activated, converse with Sophia by entering natural language queries or tasks.

## Training

### Manual Training
- Use **TRAIN MANUAL** to input question/answer pairs.

### File-based Training
- Use `LOAD train-v2.0.json` or other supported files (SQuAD or custom JSON).

### Model Training
- Use **TRAIN** to fine-tune the model on accumulated experiences.

### Autonomous Learning
- Enable with **START LEARNING** for background Wikipedia ingestion and self-improvement.

## Goals

- **Simulate Consciousness**: Model aspects of self-awareness, introspection, and subjective experience.
- **Adaptive Reasoning**: Support meta-reasoning, conceptual blending, and mental simulation.
- **Continual Learning**: Learn from new data, user input, and external sources in real time.
- **Self-Organization**: Dynamically restructure memory and cognitive modules for optimal performance.
- **Safety & Ethics**: Integrate ethical reasoning and safety checks in all responses.

## Notes

- Requires Python 3.x and dependencies:
  - PyTorch
  - numpy
  - tqdm
  - neo4j
  - wikipedia
  - nltk
  - and others.

- For **Neo4j** integration, ensure a running Neo4j instance and update connection parameters if needed.

- The system is experimental and intended for research in cognitive architectures and AI safety.

## License

This project is intended for research purposes and is licensed under the [MIT License](LICENSE).

---

Feel free to ask for more details or a specific section expansion!
