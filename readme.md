# Pacman AI Project: Multi-Agent Search and Adversarial Intelligence

## Project Overview

This project implements an AI-powered version of the classic Pacman game, focusing on multi-agent search algorithms and adversarial game-playing strategies. The implementation demonstrates various artificial intelligence techniques applied to a dynamic, real-time gaming environment.

## Core Objectives

### 1. Intelligent Agent Development
- **Reflex Agents**: Create responsive agents that make decisions based on current game state evaluation
- **Multi-Agent Search**: Implement algorithms that consider the behavior of multiple agents (Pacman and ghosts) simultaneously
- **Adversarial Search**: Develop strategies for competitive environments where agents have opposing goals

### 2. Search Algorithm Implementation
The project implements three fundamental adversarial search algorithms:

#### Minimax Algorithm
- Models the game as a zero-sum competition between Pacman and ghosts
- Assumes opponents play optimally
- Explores the full game tree to determine the best possible moves

#### Alpha-Beta Pruning
- Optimizes minimax by eliminating branches that cannot affect the final decision
- Significantly reduces computational complexity while maintaining optimal play
- Enables deeper search within time constraints

#### Expectimax Algorithm
- Handles uncertainty in opponent behavior
- Models ghosts as probabilistic agents rather than optimal adversaries
- More realistic for scenarios where opponents don't play perfectly

### 3. Advanced Evaluation Functions
- **Feature Engineering**: Consider multiple factors including:
  - Distance to food pellets
  - Ghost proximity and threat levels
  - Power pellet locations and scared ghost timers
  - Overall board control and positioning

## Educational Value

### Algorithm Understanding
This project helps gain hands-on experience with:
- **Game Theory**: Understanding competitive multi-agent scenarios
- **Search Strategies**: Comparing different approaches to decision-making
- **Optimization Techniques**: Learning when and how to apply various algorithmic improvements

### Practical AI Applications
The project demonstrates real-world AI concepts:
- **Real-time Decision Making**: Operating under time constraints
- **Uncertainty Handling**: Dealing with unpredictable opponent behavior
- **Evaluation Function Design**: Translating domain knowledge into computational heuristics
