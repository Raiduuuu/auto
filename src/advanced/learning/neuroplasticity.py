"""
Neuroplasticity Engine for HRM Trading Bot

Simulates brain neuroplasticity principles for adaptive model learning:
- Hebbian Learning: "Neurons that fire together, wire together"
- Spike-Timing Dependent Plasticity (STDP)
- Homeostatic Scaling
- Synaptic Pruning
- Memory Consolidation
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SynapticConnection:
    """Represents a synaptic connection between neurons."""
    source_id: int
    target_id: int
    weight: float
    strength: float = 1.0
    last_update: datetime = field(default_factory=datetime.now)
    activation_count: int = 0
    plasticity: float = 1.0  # How easily the connection can change

    def decay(self, factor: float = 0.99):
        """Apply synaptic decay."""
        self.strength *= factor
        if self.strength < 0.1:
            self.plasticity *= 0.9


@dataclass
class NeuronState:
    """State of a single neuron."""
    neuron_id: int
    activation: float = 0.0
    threshold: float = 0.5
    refractory_period: float = 0.0
    last_spike_time: Optional[datetime] = None
    spike_history: deque = field(default_factory=lambda: deque(maxlen=100))

    def fire(self, current_time: datetime) -> bool:
        """Check if neuron should fire."""
        if self.activation >= self.threshold and self.refractory_period <= 0:
            self.last_spike_time = current_time
            self.spike_history.append(current_time)
            self.refractory_period = 0.01  # Reset refractory period
            return True
        return False


class MemoryConsolidation:
    """
    Memory consolidation system for long-term learning.

    Implements:
    - Short-term memory buffer
    - Long-term memory storage
    - Memory replay for consolidation
    - Importance-based retention
    """

    def __init__(self, short_term_capacity: int = 1000, long_term_capacity: int = 50000):
        self.short_term_memory: deque = deque(maxlen=short_term_capacity)
        self.long_term_memory: deque = deque(maxlen=long_term_capacity)
        self.importance_threshold = 0.7
        self.consolidation_interval = timedelta(hours=1)
        self.last_consolidation = datetime.now()

    def store_short_term(self, memory: dict):
        """Store memory in short-term buffer."""
        memory["timestamp"] = datetime.now()
        memory["importance"] = self._calculate_importance(memory)
        self.short_term_memory.append(memory)

    def _calculate_importance(self, memory: dict) -> float:
        """Calculate memory importance for retention."""
        importance = 0.5

        # High error = high importance
        if "error" in memory:
            importance += min(memory["error"], 0.3)

        # High confidence predictions that were wrong = very important
        if "confidence" in memory and "correct" in memory:
            if memory["confidence"] > 0.8 and not memory["correct"]:
                importance += 0.3

        # Significant profit/loss = important
        if "profit" in memory:
            importance += min(abs(memory["profit"]) * 10, 0.2)

        return min(importance, 1.0)

    async def consolidate(self):
        """Consolidate short-term memories to long-term storage."""
        if datetime.now() - self.last_consolidation < self.consolidation_interval:
            return

        consolidated_count = 0
        for memory in list(self.short_term_memory):
            if memory["importance"] >= self.importance_threshold:
                self.long_term_memory.append(memory)
                consolidated_count += 1

        self.last_consolidation = datetime.now()
        logger.info(f"Consolidated {consolidated_count} memories to long-term storage")

    def replay_memories(self, count: int = 32) -> list:
        """Replay memories for learning (prioritizing important ones)."""
        if not self.long_term_memory:
            return list(self.short_term_memory)[:count]

        # Combine short and long term memories
        all_memories = list(self.short_term_memory) + list(self.long_term_memory)

        # Sort by importance and recency
        all_memories.sort(
            key=lambda m: (m["importance"], -((datetime.now() - m["timestamp"]).total_seconds())),
            reverse=True
        )

        return all_memories[:count]


class NeuroplasticityEngine:
    """
    Neuroplasticity Engine that simulates brain-like learning dynamics.

    Implements:
    - Hebbian Learning Rule
    - Spike-Timing Dependent Plasticity (STDP)
    - Homeostatic Scaling
    - Synaptic Pruning
    - Memory Consolidation
    """

    def __init__(
        self,
        num_neurons: int = 100,
        learning_rate: float = 0.01,
        stdp_window: float = 20.0,  # milliseconds
        target_activity: float = 0.1,
    ):
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.stdp_window = stdp_window
        self.target_activity = target_activity

        # Initialize neurons
        self.neurons: list[NeuronState] = [
            NeuronState(neuron_id=i) for i in range(num_neurons)
        ]

        # Initialize synaptic connections (sparse connectivity)
        self.synapses: dict[tuple[int, int], SynapticConnection] = {}
        self._initialize_synapses()

        # Memory consolidation system
        self.memory_consolidation = MemoryConsolidation()

        # Activity tracking
        self.activity_history: deque = deque(maxlen=1000)
        self.weight_history: deque = deque(maxlen=100)

        # Learning state
        self.total_updates = 0
        self.last_pruning = datetime.now()
        self.pruning_interval = timedelta(hours=24)

        logger.info(f"NeuroplasticityEngine initialized with {num_neurons} neurons")

    def _initialize_synapses(self, connectivity: float = 0.1):
        """Initialize sparse synaptic connections."""
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and np.random.random() < connectivity:
                    weight = np.random.randn() * 0.1
                    self.synapses[(i, j)] = SynapticConnection(
                        source_id=i,
                        target_id=j,
                        weight=weight,
                    )

    def hebbian_learning(
        self,
        pre_activation: np.ndarray,
        post_activation: np.ndarray,
        learning_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Hebbian Learning Rule: "Neurons that fire together, wire together"

        Args:
            pre_activation: Activations of pre-synaptic neurons
            post_activation: Activations of post-synaptic neurons
            learning_rate: Optional custom learning rate

        Returns:
            Weight change matrix
        """
        lr = learning_rate or self.learning_rate

        # Compute outer product for weight changes
        weight_change = lr * np.outer(pre_activation, post_activation)

        # Apply weight change to synapses
        for (i, j), synapse in self.synapses.items():
            if i < len(pre_activation) and j < len(post_activation):
                synapse.weight += weight_change[i, j] * synapse.plasticity
                synapse.activation_count += 1
                synapse.last_update = datetime.now()

        self.total_updates += 1
        return weight_change

    def spike_timing_dependent_plasticity(
        self,
        pre_spike_time: datetime,
        post_spike_time: datetime,
        synapse_id: tuple[int, int]
    ) -> float:
        """
        Spike-Timing Dependent Plasticity (STDP).

        Strengthens connections where pre-synaptic neuron fires before post-synaptic,
        and weakens connections in the opposite case.

        Args:
            pre_spike_time: Time of pre-synaptic spike
            post_spike_time: Time of post-synaptic spike
            synapse_id: Tuple of (source_id, target_id)

        Returns:
            Weight change value
        """
        if synapse_id not in self.synapses:
            return 0.0

        # Calculate time difference in milliseconds
        time_diff = (post_spike_time - pre_spike_time).total_seconds() * 1000

        # STDP learning rule
        if time_diff > 0:  # Post fires after pre (potentiation)
            weight_change = 0.01 * np.exp(-time_diff / self.stdp_window)
        else:  # Pre fires after post (depression)
            weight_change = -0.012 * np.exp(time_diff / self.stdp_window)

        # Apply weight change
        synapse = self.synapses[synapse_id]
        synapse.weight += weight_change * synapse.plasticity
        synapse.weight = np.clip(synapse.weight, -1.0, 1.0)  # Bound weights
        synapse.last_update = datetime.now()

        return weight_change

    def homeostatic_scaling(self, neuron_id: int) -> float:
        """
        Homeostatic Scaling for activity regulation.

        Scales synaptic weights to maintain target activity level.

        Args:
            neuron_id: ID of neuron to scale

        Returns:
            Scaling factor applied
        """
        neuron = self.neurons[neuron_id]

        # Calculate current activity (spike rate)
        recent_spikes = len([t for t in neuron.spike_history
                            if (datetime.now() - t).total_seconds() < 1.0])
        current_activity = recent_spikes / 1.0  # Spikes per second

        # Calculate scaling factor
        scaling_factor = self.target_activity / (current_activity + 1e-8)
        scaling_factor = np.clip(scaling_factor, 0.5, 2.0)

        # Apply scaling to incoming synapses
        for (source, target), synapse in self.synapses.items():
            if target == neuron_id:
                synapse.weight *= scaling_factor

        return scaling_factor

    def synaptic_pruning(self, strength_threshold: float = 0.1):
        """
        Prune weak synaptic connections.

        Removes connections that have become too weak, simulating
        the brain's "use it or lose it" principle.

        Args:
            strength_threshold: Minimum strength to keep connection
        """
        if datetime.now() - self.last_pruning < self.pruning_interval:
            return

        pruned = []
        for synapse_id, synapse in list(self.synapses.items()):
            # Decay strength based on inactivity
            time_since_update = (datetime.now() - synapse.last_update).total_seconds()
            synapse.decay(factor=0.99 ** (time_since_update / 3600))

            # Prune if too weak
            if synapse.strength < strength_threshold:
                pruned.append(synapse_id)

        for synapse_id in pruned:
            del self.synapses[synapse_id]

        self.last_pruning = datetime.now()
        logger.info(f"Pruned {len(pruned)} weak synapses")

    def forward_pass(self, input_activations: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neuroplastic network.

        Args:
            input_activations: Input activation vector

        Returns:
            Output activation vector
        """
        current_time = datetime.now()

        # Set input neuron activations
        num_inputs = min(len(input_activations), self.num_neurons // 2)
        for i in range(num_inputs):
            self.neurons[i].activation = input_activations[i]

        # Propagate activations
        output_activations = np.zeros(self.num_neurons - num_inputs)

        for (source, target), synapse in self.synapses.items():
            if source < num_inputs and target >= num_inputs:
                output_idx = target - num_inputs
                if output_idx < len(output_activations):
                    output_activations[output_idx] += (
                        self.neurons[source].activation * synapse.weight * synapse.strength
                    )

        # Apply activation function and check for spikes
        output_activations = 1 / (1 + np.exp(-output_activations))  # Sigmoid

        for i, activation in enumerate(output_activations):
            neuron_idx = num_inputs + i
            if neuron_idx < len(self.neurons):
                self.neurons[neuron_idx].activation = activation
                self.neurons[neuron_idx].fire(current_time)

        # Track activity
        avg_activity = np.mean(output_activations)
        self.activity_history.append(avg_activity)

        return output_activations

    def backward_learning(
        self,
        input_activations: np.ndarray,
        target_output: np.ndarray,
        output_activations: np.ndarray
    ):
        """
        Backward learning pass combining multiple plasticity rules.

        Args:
            input_activations: Input activations
            target_output: Target output values
            output_activations: Actual output activations
        """
        current_time = datetime.now()

        # Calculate error
        error = target_output - output_activations[:len(target_output)]

        # Apply Hebbian learning with error-modulated signal
        error_signal = np.pad(error, (0, max(0, self.num_neurons // 2 - len(error))))

        # Update weights using error-modulated Hebbian learning
        pre_act = input_activations[:self.num_neurons // 2]
        pre_act = np.pad(pre_act, (0, max(0, self.num_neurons // 2 - len(pre_act))))

        self.hebbian_learning(pre_act, error_signal)

        # Apply STDP for recently active neurons
        for neuron in self.neurons:
            if neuron.last_spike_time and (current_time - neuron.last_spike_time).total_seconds() < 0.1:
                for other_neuron in self.neurons:
                    if other_neuron.last_spike_time and other_neuron.neuron_id != neuron.neuron_id:
                        synapse_id = (other_neuron.neuron_id, neuron.neuron_id)
                        if synapse_id in self.synapses:
                            self.spike_timing_dependent_plasticity(
                                other_neuron.last_spike_time,
                                neuron.last_spike_time,
                                synapse_id
                            )

        # Store experience in memory
        self.memory_consolidation.store_short_term({
            "input": input_activations.tolist(),
            "output": output_activations.tolist(),
            "target": target_output.tolist(),
            "error": float(np.mean(np.abs(error))),
        })

    async def consolidate_memories(self):
        """Consolidate memories for long-term retention."""
        await self.memory_consolidation.consolidate()

    def replay_learning(self, batch_size: int = 32):
        """Learn from replayed memories."""
        memories = self.memory_consolidation.replay_memories(batch_size)

        for memory in memories:
            input_act = np.array(memory["input"])
            target = np.array(memory["target"])
            output = self.forward_pass(input_act)
            self.backward_learning(input_act, target, output)

    def get_connection_matrix(self) -> np.ndarray:
        """Get the current weight matrix."""
        matrix = np.zeros((self.num_neurons, self.num_neurons))
        for (i, j), synapse in self.synapses.items():
            matrix[i, j] = synapse.weight * synapse.strength
        return matrix

    def get_state(self) -> dict:
        """Get current engine state."""
        return {
            "num_neurons": self.num_neurons,
            "num_synapses": len(self.synapses),
            "total_updates": self.total_updates,
            "avg_activity": float(np.mean(self.activity_history)) if self.activity_history else 0.0,
            "short_term_memories": len(self.memory_consolidation.short_term_memory),
            "long_term_memories": len(self.memory_consolidation.long_term_memory),
        }


class AdaptiveNeuroplasticityController:
    """
    Controller that adapts neuroplasticity parameters based on learning performance.
    """

    def __init__(self, engine: NeuroplasticityEngine):
        self.engine = engine
        self.performance_history: deque = deque(maxlen=100)
        self.adaptation_interval = 100  # Updates between adaptations

    def update_performance(self, accuracy: float, loss: float):
        """Update performance metrics."""
        self.performance_history.append({
            "accuracy": accuracy,
            "loss": loss,
            "timestamp": datetime.now(),
        })

        if len(self.performance_history) >= self.adaptation_interval:
            self._adapt_parameters()

    def _adapt_parameters(self):
        """Adapt neuroplasticity parameters based on performance."""
        recent = list(self.performance_history)[-self.adaptation_interval:]

        avg_accuracy = np.mean([p["accuracy"] for p in recent])
        accuracy_trend = np.polyfit(range(len(recent)), [p["accuracy"] for p in recent], 1)[0]

        # Adapt learning rate
        if accuracy_trend < 0:  # Performance declining
            self.engine.learning_rate *= 1.2
        elif avg_accuracy > 0.8:  # High performance, fine-tune
            self.engine.learning_rate *= 0.9

        # Adapt pruning threshold
        if avg_accuracy < 0.5:  # Poor performance
            self.engine.pruning_interval = timedelta(hours=48)  # Less aggressive pruning
        else:
            self.engine.pruning_interval = timedelta(hours=24)

        # Bound learning rate
        self.engine.learning_rate = np.clip(self.engine.learning_rate, 1e-4, 0.1)

        logger.info(f"Adapted parameters: lr={self.engine.learning_rate:.6f}")
