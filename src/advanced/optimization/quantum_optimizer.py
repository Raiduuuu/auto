"""
Quantum-Inspired Optimizer for HRM Trading Bot

Implements optimization techniques inspired by quantum mechanics:
- Quantum Tunneling for escaping local minima
- Superposition for exploring multiple parameter configurations
- Quantum Annealing for global optimization
- Entanglement for correlated parameter updates
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantumState:
    """Represents a quantum state for a parameter."""
    amplitude: complex  # Probability amplitude
    phase: float  # Phase angle
    collapsed_value: Optional[float] = None

    @property
    def probability(self) -> float:
        """Get probability from amplitude."""
        return abs(self.amplitude) ** 2

    def collapse(self) -> float:
        """Collapse quantum state to classical value."""
        if self.collapsed_value is None:
            self.collapsed_value = self.probability * np.cos(self.phase)
        return self.collapsed_value


@dataclass
class ParameterGroup:
    """Group of parameters with their quantum states."""
    params: list
    quantum_states: dict = field(default_factory=dict)
    entanglement_partners: list = field(default_factory=list)
    learning_rate: float = 0.01


class QuantumTunneling:
    """
    Quantum Tunneling mechanism for escaping local minima.

    Uses probabilistic jumps to explore beyond energy barriers
    in the loss landscape.
    """

    def __init__(self, barrier_threshold: float = 0.1, tunneling_rate: float = 0.01):
        self.barrier_threshold = barrier_threshold
        self.tunneling_rate = tunneling_rate
        self.tunneling_events = 0
        self.successful_tunnels = 0

    def should_tunnel(self, gradient_magnitude: float, quantum_strength: float) -> bool:
        """
        Determine if tunneling should occur.

        Args:
            gradient_magnitude: Magnitude of current gradient
            quantum_strength: Strength of quantum effects

        Returns:
            True if tunneling should be attempted
        """
        # Lower gradient = more likely to be stuck in local minimum
        tunneling_probability = np.exp(-gradient_magnitude / quantum_strength)
        return np.random.random() < tunneling_probability * self.tunneling_rate

    def tunnel(self, param_value: float, gradient: float, quantum_strength: float) -> float:
        """
        Perform quantum tunneling jump.

        Args:
            param_value: Current parameter value
            gradient: Current gradient
            quantum_strength: Strength of quantum effects

        Returns:
            New parameter value after tunneling
        """
        self.tunneling_events += 1

        # Generate tunneling direction (perpendicular to gradient)
        tunnel_direction = np.sign(np.random.randn())

        # Tunneling distance based on quantum strength
        tunnel_distance = quantum_strength * np.random.exponential(1.0)

        # Apply tunneling jump
        new_value = param_value + tunnel_direction * tunnel_distance

        return new_value


class QuantumAnnealing:
    """
    Quantum Annealing for global optimization.

    Simulates quantum fluctuations that decrease over time,
    allowing exploration early and exploitation later.
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.001,
        annealing_rate: float = 0.99
    ):
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.final_temperature = final_temperature
        self.annealing_rate = annealing_rate
        self.step = 0

    def get_quantum_fluctuation(self, param_shape: tuple) -> np.ndarray:
        """
        Get quantum fluctuation for parameters.

        Args:
            param_shape: Shape of parameter tensor

        Returns:
            Quantum fluctuation array
        """
        # Quantum fluctuations decrease with temperature
        fluctuation_magnitude = self.temperature
        fluctuation = np.random.randn(*param_shape) * fluctuation_magnitude
        return fluctuation

    def anneal(self):
        """Perform one annealing step."""
        self.step += 1
        self.temperature = max(
            self.final_temperature,
            self.temperature * self.annealing_rate
        )

    def reset(self):
        """Reset annealing schedule."""
        self.temperature = self.initial_temperature
        self.step = 0

    @property
    def exploration_ratio(self) -> float:
        """Current exploration vs exploitation ratio."""
        return (self.temperature - self.final_temperature) / (
            self.initial_temperature - self.final_temperature
        )


class QuantumEntanglement:
    """
    Quantum Entanglement for correlated parameter updates.

    Creates correlations between related parameters so they
    update together in a coordinated manner.
    """

    def __init__(self):
        self.entangled_pairs: list[tuple[int, int, float]] = []  # (param1_id, param2_id, correlation)
        self.correlation_matrix: Optional[np.ndarray] = None

    def entangle(self, param1_id: int, param2_id: int, correlation: float = 0.5):
        """
        Create entanglement between two parameters.

        Args:
            param1_id: ID of first parameter
            param2_id: ID of second parameter
            correlation: Correlation strength (-1 to 1)
        """
        self.entangled_pairs.append((param1_id, param2_id, correlation))

    def detect_entanglement(self, gradient_history: np.ndarray, threshold: float = 0.7):
        """
        Automatically detect parameters that should be entangled.

        Args:
            gradient_history: History of gradients (steps x params)
            threshold: Correlation threshold for entanglement
        """
        if len(gradient_history) < 10:
            return

        # Compute correlation matrix of gradients
        self.correlation_matrix = np.corrcoef(gradient_history.T)

        # Find highly correlated parameter pairs
        n_params = self.correlation_matrix.shape[0]
        for i in range(n_params):
            for j in range(i + 1, n_params):
                corr = abs(self.correlation_matrix[i, j])
                if corr > threshold:
                    self.entangle(i, j, self.correlation_matrix[i, j])

    def get_entangled_update(
        self,
        param_id: int,
        base_update: float,
        all_updates: np.ndarray
    ) -> float:
        """
        Get update modified by entanglement.

        Args:
            param_id: ID of parameter to update
            base_update: Base update value
            all_updates: All parameter updates

        Returns:
            Modified update considering entanglement
        """
        entangled_contribution = 0.0
        entanglement_count = 0

        for p1_id, p2_id, correlation in self.entangled_pairs:
            if param_id == p1_id:
                partner_update = all_updates[p2_id]
                entangled_contribution += correlation * partner_update
                entanglement_count += 1
            elif param_id == p2_id:
                partner_update = all_updates[p1_id]
                entangled_contribution += correlation * partner_update
                entanglement_count += 1

        if entanglement_count > 0:
            entangled_contribution /= entanglement_count
            return 0.7 * base_update + 0.3 * entangled_contribution

        return base_update


class QuantumInspiredOptimizer:
    """
    Quantum-Inspired Optimizer for neural network training.

    Combines multiple quantum-inspired techniques:
    - Quantum Tunneling for escaping local minima
    - Quantum Annealing for temperature-based exploration
    - Quantum Superposition for exploring multiple configurations
    - Quantum Entanglement for correlated updates
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        quantum_strength: float = 0.1,
        enable_tunneling: bool = True,
        enable_annealing: bool = True,
        enable_entanglement: bool = True,
    ):
        self.learning_rate = learning_rate
        self.quantum_strength = quantum_strength
        self.enable_tunneling = enable_tunneling
        self.enable_annealing = enable_annealing
        self.enable_entanglement = enable_entanglement

        # Quantum components
        self.tunneling = QuantumTunneling(
            barrier_threshold=0.1,
            tunneling_rate=quantum_strength
        )
        self.annealing = QuantumAnnealing(
            initial_temperature=1.0,
            final_temperature=0.001,
            annealing_rate=0.995
        )
        self.entanglement = QuantumEntanglement()

        # State tracking
        self.quantum_states: dict[int, QuantumState] = {}
        self.param_groups: list[ParameterGroup] = []
        self.gradient_history: list[np.ndarray] = []

        # Momentum
        self.momentum = 0.9
        self.velocities: dict[int, np.ndarray] = {}

        # Statistics
        self.step_count = 0
        self.tunnel_count = 0
        self.best_loss = float('inf')
        self.loss_history: deque = deque(maxlen=100)

        logger.info(f"QuantumInspiredOptimizer initialized with strength={quantum_strength}")

    def register_parameters(self, params: list, group_lr: Optional[float] = None):
        """
        Register parameters for optimization.

        Args:
            params: List of parameter arrays
            group_lr: Optional learning rate for this group
        """
        param_group = ParameterGroup(
            params=params,
            learning_rate=group_lr or self.learning_rate
        )

        for i, param in enumerate(params):
            param_id = id(param)

            # Initialize quantum state
            self.quantum_states[param_id] = QuantumState(
                amplitude=complex(1.0, 0.0),
                phase=0.0
            )

            # Initialize velocity
            if hasattr(param, 'shape'):
                self.velocities[param_id] = np.zeros_like(param)

        self.param_groups.append(param_group)

    def quantum_tunneling_step(self, params: list, gradients: list):
        """
        Apply quantum tunneling to escape local minima.

        Args:
            params: List of parameter arrays
            gradients: List of gradient arrays
        """
        if not self.enable_tunneling:
            return

        for param, grad in zip(params, gradients):
            if grad is None:
                continue

            param_id = id(param)
            grad_array = np.array(grad)
            gradient_magnitude = np.linalg.norm(grad_array)

            if self.tunneling.should_tunnel(gradient_magnitude, self.quantum_strength):
                # Apply tunneling to parameter
                if hasattr(param, 'shape'):
                    tunnel_jump = np.random.randn(*param.shape) * self.quantum_strength * 0.1
                    param[:] = param + tunnel_jump
                else:
                    new_value = self.tunneling.tunnel(
                        float(param),
                        float(grad_array.mean()),
                        self.quantum_strength
                    )
                    param = new_value

                self.tunnel_count += 1
                logger.debug(f"Quantum tunnel applied to parameter {param_id}")

    def superposition_update(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Apply superposition-based parameter update.

        Maintains quantum state that mixes current and historical values.

        Args:
            param: Parameter array
            grad: Gradient array

        Returns:
            Updated parameter array
        """
        param_id = id(param)

        if param_id not in self.quantum_states:
            self.quantum_states[param_id] = QuantumState(
                amplitude=complex(1.0, 0.0),
                phase=0.0
            )

        quantum_state = self.quantum_states[param_id]

        # Update phase based on gradient
        grad_magnitude = np.linalg.norm(grad)
        quantum_state.phase += grad_magnitude * 0.01

        # Superposition: blend current state with quantum perturbation
        alpha = self.quantum_strength * self.annealing.exploration_ratio

        # Quantum perturbation
        quantum_perturbation = np.random.randn(*param.shape) * alpha

        # Blend: (1-alpha) * current + alpha * perturbed
        updated_param = (1 - alpha) * param + alpha * (param + quantum_perturbation)

        return updated_param

    def step(self, params: list, gradients: list, loss: Optional[float] = None):
        """
        Perform one optimization step.

        Args:
            params: List of parameter arrays
            gradients: List of gradient arrays
            loss: Optional current loss value
        """
        self.step_count += 1

        if loss is not None:
            self.loss_history.append(loss)
            if loss < self.best_loss:
                self.best_loss = loss

        # Store gradient history for entanglement detection
        flat_grads = []
        for grad in gradients:
            if grad is not None:
                flat_grads.extend(np.array(grad).flatten()[:10])  # Sample first 10
        if flat_grads:
            self.gradient_history.append(np.array(flat_grads))

        # Detect entanglement periodically
        if self.enable_entanglement and self.step_count % 100 == 0:
            if len(self.gradient_history) >= 50:
                grad_array = np.array(self.gradient_history[-50:])
                self.entanglement.detect_entanglement(grad_array)

        # Apply quantum tunneling
        self.quantum_tunneling_step(params, gradients)

        # Compute all updates
        all_updates = []
        for param, grad in zip(params, gradients):
            if grad is None:
                all_updates.append(0.0)
                continue

            param_id = id(param)
            grad_array = np.array(grad)

            # Add momentum
            if param_id in self.velocities:
                self.velocities[param_id] = (
                    self.momentum * self.velocities[param_id] -
                    self.learning_rate * grad_array
                )
                update = self.velocities[param_id]
            else:
                update = -self.learning_rate * grad_array

            # Add quantum annealing fluctuation
            if self.enable_annealing:
                fluctuation = self.annealing.get_quantum_fluctuation(grad_array.shape)
                update = update + fluctuation * self.learning_rate

            all_updates.append(np.mean(update))

        all_updates = np.array(all_updates)

        # Apply updates with entanglement
        for i, (param, grad) in enumerate(zip(params, gradients)):
            if grad is None:
                continue

            param_id = id(param)
            grad_array = np.array(grad)

            # Get base update
            if param_id in self.velocities:
                base_update = self.velocities[param_id]
            else:
                base_update = -self.learning_rate * grad_array

            # Apply entanglement modification
            if self.enable_entanglement and len(all_updates) > 0:
                entangled_factor = self.entanglement.get_entangled_update(
                    i, np.mean(base_update), all_updates
                )
                base_update = base_update * (1 + 0.1 * np.sign(entangled_factor))

            # Apply superposition update
            if hasattr(param, 'shape'):
                param[:] = self.superposition_update(param + base_update, grad_array)
            elif hasattr(param, '__setitem__'):
                param[:] = param + base_update

        # Anneal temperature
        if self.enable_annealing:
            self.annealing.anneal()

    def zero_grad(self):
        """Reset gradients (for compatibility with standard optimizers)."""
        pass

    def get_state(self) -> dict:
        """Get optimizer state."""
        return {
            "step_count": self.step_count,
            "tunnel_count": self.tunnel_count,
            "best_loss": self.best_loss,
            "temperature": self.annealing.temperature,
            "exploration_ratio": self.annealing.exploration_ratio,
            "num_entangled_pairs": len(self.entanglement.entangled_pairs),
            "quantum_strength": self.quantum_strength,
        }

    def load_state(self, state: dict):
        """Load optimizer state."""
        self.step_count = state.get("step_count", 0)
        self.tunnel_count = state.get("tunnel_count", 0)
        self.best_loss = state.get("best_loss", float('inf'))
        self.annealing.temperature = state.get("temperature", 1.0)


class AdaptiveQuantumOptimizer(QuantumInspiredOptimizer):
    """
    Adaptive version that adjusts quantum parameters based on performance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_interval = 50
        self.performance_window: deque = deque(maxlen=100)

    def step(self, params: list, gradients: list, loss: Optional[float] = None):
        """Perform optimization step with adaptation."""
        super().step(params, gradients, loss)

        if loss is not None:
            self.performance_window.append(loss)

        # Adapt quantum parameters periodically
        if self.step_count % self.adaptation_interval == 0:
            self._adapt_parameters()

    def _adapt_parameters(self):
        """Adapt quantum parameters based on performance."""
        if len(self.performance_window) < 20:
            return

        recent_losses = list(self.performance_window)
        avg_loss = np.mean(recent_losses)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

        # If loss is decreasing, reduce exploration
        if loss_trend < 0:
            self.quantum_strength *= 0.95
            self.annealing.annealing_rate = min(0.999, self.annealing.annealing_rate * 1.01)
        # If loss is stuck or increasing, increase exploration
        elif loss_trend > 0 or np.std(recent_losses[-10:]) < 0.001:
            self.quantum_strength = min(0.5, self.quantum_strength * 1.1)
            self.annealing.annealing_rate = max(0.9, self.annealing.annealing_rate * 0.99)

        # Bound quantum strength
        self.quantum_strength = np.clip(self.quantum_strength, 0.01, 0.5)

        logger.debug(
            f"Adapted quantum params: strength={self.quantum_strength:.4f}, "
            f"annealing_rate={self.annealing.annealing_rate:.4f}"
        )
