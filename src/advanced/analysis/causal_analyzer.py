"""
Causal Market Analyzer for HRM Trading Bot

Implements causal inference methods for understanding market dynamics:
- Causal Discovery (PC Algorithm)
- Granger Causality
- Treatment Effect Estimation
- Counterfactual Analysis
- Intervention Analysis
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    """Represents a causal edge in the causal graph."""
    source: str
    target: str
    strength: float
    confidence: float
    direction: str = "forward"  # forward, backward, bidirectional
    lag: int = 0  # Time lag in periods


@dataclass
class InterventionEffect:
    """Result of an intervention analysis."""
    treatment: str
    outcome: str
    effect: float
    confidence_interval: tuple[float, float]
    p_value: float
    sample_size: int


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis."""
    scenario: str
    observed_outcome: float
    counterfactual_outcome: float
    effect: float
    probability: float


class CausalGraph:
    """
    Directed Acyclic Graph (DAG) representing causal relationships.
    """

    def __init__(self):
        self.nodes: set[str] = set()
        self.edges: list[CausalEdge] = []
        self.adjacency: dict[str, list[str]] = {}

    def add_node(self, node: str):
        """Add a node to the graph."""
        self.nodes.add(node)
        if node not in self.adjacency:
            self.adjacency[node] = []

    def add_edge(self, edge: CausalEdge):
        """Add a causal edge to the graph."""
        self.add_node(edge.source)
        self.add_node(edge.target)
        self.edges.append(edge)
        self.adjacency[edge.source].append(edge.target)

    def get_parents(self, node: str) -> list[str]:
        """Get parent nodes (direct causes) of a node."""
        parents = []
        for edge in self.edges:
            if edge.target == node:
                parents.append(edge.source)
        return parents

    def get_children(self, node: str) -> list[str]:
        """Get child nodes (direct effects) of a node."""
        return self.adjacency.get(node, [])

    def get_ancestors(self, node: str, visited: Optional[set] = None) -> set[str]:
        """Get all ancestor nodes (all causes) of a node."""
        if visited is None:
            visited = set()

        ancestors = set()
        for parent in self.get_parents(node):
            if parent not in visited:
                visited.add(parent)
                ancestors.add(parent)
                ancestors.update(self.get_ancestors(parent, visited))

        return ancestors

    def to_dict(self) -> dict:
        """Convert graph to dictionary representation."""
        return {
            "nodes": list(self.nodes),
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "strength": e.strength,
                    "confidence": e.confidence,
                    "lag": e.lag,
                }
                for e in self.edges
            ],
        }


class GrangerCausality:
    """
    Granger Causality test for time series data.

    Tests whether one time series helps predict another.
    """

    def __init__(self, max_lag: int = 10, significance_level: float = 0.05):
        self.max_lag = max_lag
        self.significance_level = significance_level

    def test(
        self,
        cause_series: np.ndarray,
        effect_series: np.ndarray,
        lag: Optional[int] = None
    ) -> dict:
        """
        Test Granger causality between two time series.

        Args:
            cause_series: Potential cause time series
            effect_series: Potential effect time series
            lag: Specific lag to test (if None, tests all lags up to max_lag)

        Returns:
            Dictionary with test results
        """
        if len(cause_series) != len(effect_series):
            raise ValueError("Time series must have equal length")

        n = len(cause_series)
        if lag is None:
            lags_to_test = range(1, min(self.max_lag + 1, n // 4))
        else:
            lags_to_test = [lag]

        best_result = None
        best_p_value = 1.0

        for test_lag in lags_to_test:
            # Create lagged features
            X_restricted = self._create_lagged_features(effect_series, test_lag)
            X_unrestricted = np.column_stack([
                X_restricted,
                self._create_lagged_features(cause_series, test_lag)
            ])
            y = effect_series[test_lag:]

            # Truncate to match
            min_len = min(len(y), X_restricted.shape[0], X_unrestricted.shape[0])
            y = y[:min_len]
            X_restricted = X_restricted[:min_len]
            X_unrestricted = X_unrestricted[:min_len]

            # Fit restricted model (only past values of effect)
            rss_restricted = self._compute_rss(X_restricted, y)

            # Fit unrestricted model (past values of both)
            rss_unrestricted = self._compute_rss(X_unrestricted, y)

            # F-test
            df1 = test_lag  # Number of restrictions
            df2 = len(y) - 2 * test_lag - 1

            if df2 <= 0 or rss_unrestricted == 0:
                continue

            f_stat = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)

            if p_value < best_p_value:
                best_p_value = p_value
                best_result = {
                    "lag": test_lag,
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "significant": p_value < self.significance_level,
                    "rss_restricted": float(rss_restricted),
                    "rss_unrestricted": float(rss_unrestricted),
                }

        return best_result or {
            "lag": 0,
            "f_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    def _create_lagged_features(self, series: np.ndarray, lag: int) -> np.ndarray:
        """Create lagged feature matrix."""
        n = len(series)
        features = []
        for l in range(1, lag + 1):
            features.append(series[lag - l:n - l])

        if not features:
            return np.zeros((n - lag, 1))

        return np.column_stack(features)

    def _compute_rss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute residual sum of squares for OLS regression."""
        try:
            # Add constant term
            X_with_const = np.column_stack([np.ones(len(X)), X])
            # OLS solution
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            predictions = X_with_const @ beta
            residuals = y - predictions
            return float(np.sum(residuals ** 2))
        except np.linalg.LinAlgError:
            return float('inf')


class PCAlgorithm:
    """
    PC (Peter-Clark) Algorithm for causal structure discovery.

    Learns causal structure from observational data using
    conditional independence tests.
    """

    def __init__(self, alpha: float = 0.05, max_cond_set: int = 3):
        self.alpha = alpha
        self.max_cond_set = max_cond_set

    def learn(self, data: np.ndarray, feature_names: list[str]) -> CausalGraph:
        """
        Learn causal structure from data.

        Args:
            data: Data matrix (samples x features)
            feature_names: Names of features

        Returns:
            CausalGraph with discovered causal structure
        """
        n_features = len(feature_names)
        graph = CausalGraph()

        for name in feature_names:
            graph.add_node(name)

        # Start with complete undirected graph
        adjacency = {i: set(range(n_features)) - {i} for i in range(n_features)}

        # Phase 1: Edge removal based on conditional independence
        for cond_set_size in range(self.max_cond_set + 1):
            for i in range(n_features):
                neighbors = list(adjacency[i])
                for j in neighbors:
                    if j not in adjacency[i]:
                        continue

                    # Find conditioning sets
                    other_neighbors = [n for n in adjacency[i] if n != j]
                    if len(other_neighbors) < cond_set_size:
                        continue

                    # Test conditional independence
                    for cond_set in self._get_subsets(other_neighbors, cond_set_size):
                        if self._conditional_independence_test(data, i, j, list(cond_set)):
                            adjacency[i].discard(j)
                            adjacency[j].discard(i)
                            break

        # Phase 2: Orient edges (simplified version)
        oriented_edges = set()
        for i in range(n_features):
            for j in adjacency[i]:
                if (j, i) not in oriented_edges and (i, j) not in oriented_edges:
                    # Use correlation magnitude to orient
                    corr = np.corrcoef(data[:, i], data[:, j])[0, 1]

                    # Check temporal ordering if possible
                    if self._has_temporal_precedence(data, i, j):
                        source, target = i, j
                    elif self._has_temporal_precedence(data, j, i):
                        source, target = j, i
                    else:
                        # Default: higher variance variable is cause
                        if np.var(data[:, i]) > np.var(data[:, j]):
                            source, target = i, j
                        else:
                            source, target = j, i

                    edge = CausalEdge(
                        source=feature_names[source],
                        target=feature_names[target],
                        strength=abs(corr),
                        confidence=1 - self.alpha,
                    )
                    graph.add_edge(edge)
                    oriented_edges.add((source, target))

        logger.info(f"PC Algorithm discovered {len(graph.edges)} causal edges")
        return graph

    def _conditional_independence_test(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        cond_set: list[int]
    ) -> bool:
        """Test if X_i and X_j are conditionally independent given conditioning set."""
        if not cond_set:
            # Simple correlation test
            corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
            n = len(data)
            t_stat = corr * np.sqrt((n - 2) / (1 - corr ** 2 + 1e-10))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            return p_value > self.alpha

        # Partial correlation test
        try:
            partial_corr = self._partial_correlation(data, i, j, cond_set)
            n = len(data)
            k = len(cond_set)
            t_stat = partial_corr * np.sqrt((n - k - 2) / (1 - partial_corr ** 2 + 1e-10))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 2))
            return p_value > self.alpha
        except Exception:
            return False

    def _partial_correlation(
        self,
        data: np.ndarray,
        i: int,
        j: int,
        cond_set: list[int]
    ) -> float:
        """Compute partial correlation."""
        # Residualize i and j on conditioning set
        X_cond = data[:, cond_set]
        X_cond = np.column_stack([np.ones(len(X_cond)), X_cond])

        try:
            beta_i = np.linalg.lstsq(X_cond, data[:, i], rcond=None)[0]
            beta_j = np.linalg.lstsq(X_cond, data[:, j], rcond=None)[0]

            residual_i = data[:, i] - X_cond @ beta_i
            residual_j = data[:, j] - X_cond @ beta_j

            return np.corrcoef(residual_i, residual_j)[0, 1]
        except np.linalg.LinAlgError:
            return 0.0

    def _get_subsets(self, elements: list, size: int):
        """Generate all subsets of given size."""
        from itertools import combinations
        return combinations(elements, size)

    def _has_temporal_precedence(self, data: np.ndarray, i: int, j: int) -> bool:
        """Check if variable i temporally precedes j (using cross-correlation)."""
        # Simple check: correlation with lagged values
        if len(data) < 10:
            return False

        corr_forward = np.corrcoef(data[:-1, i], data[1:, j])[0, 1]
        corr_backward = np.corrcoef(data[:-1, j], data[1:, i])[0, 1]

        return abs(corr_forward) > abs(corr_backward)


class TreatmentEffectEstimator:
    """
    Estimates causal treatment effects using various methods.
    """

    def __init__(self):
        self.propensity_model = None
        self.outcome_model = None

    def estimate_ate(
        self,
        data: np.ndarray,
        treatment_col: int,
        outcome_col: int,
        confounder_cols: list[int]
    ) -> InterventionEffect:
        """
        Estimate Average Treatment Effect using Inverse Propensity Weighting.

        Args:
            data: Data matrix
            treatment_col: Column index of treatment variable
            outcome_col: Column index of outcome variable
            confounder_cols: Column indices of confounders

        Returns:
            InterventionEffect with estimated causal effect
        """
        n = len(data)
        treatment = data[:, treatment_col]
        outcome = data[:, outcome_col]
        confounders = data[:, confounder_cols]

        # Binarize treatment if continuous
        treatment_binary = (treatment > np.median(treatment)).astype(float)

        # Estimate propensity scores (probability of treatment)
        propensity_scores = self._estimate_propensity(confounders, treatment_binary)

        # Clip propensity scores to avoid extreme weights
        propensity_scores = np.clip(propensity_scores, 0.1, 0.9)

        # Compute IPW weights
        weights_treated = treatment_binary / propensity_scores
        weights_control = (1 - treatment_binary) / (1 - propensity_scores)

        # Estimate ATE
        ate_treated = np.sum(weights_treated * outcome) / np.sum(weights_treated)
        ate_control = np.sum(weights_control * outcome) / np.sum(weights_control)
        ate = ate_treated - ate_control

        # Bootstrap confidence interval
        bootstrap_ates = []
        for _ in range(100):
            idx = np.random.choice(n, n, replace=True)
            boot_ate = self._compute_ate_sample(
                treatment_binary[idx],
                outcome[idx],
                propensity_scores[idx]
            )
            bootstrap_ates.append(boot_ate)

        ci_low = np.percentile(bootstrap_ates, 2.5)
        ci_high = np.percentile(bootstrap_ates, 97.5)

        # Compute p-value (test if ATE is different from 0)
        ate_std = np.std(bootstrap_ates)
        z_stat = ate / (ate_std + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return InterventionEffect(
            treatment=f"col_{treatment_col}",
            outcome=f"col_{outcome_col}",
            effect=float(ate),
            confidence_interval=(float(ci_low), float(ci_high)),
            p_value=float(p_value),
            sample_size=n,
        )

    def _estimate_propensity(
        self,
        confounders: np.ndarray,
        treatment: np.ndarray
    ) -> np.ndarray:
        """Estimate propensity scores using logistic regression."""
        # Simple logistic regression
        X = np.column_stack([np.ones(len(confounders)), confounders])
        y = treatment

        # Gradient descent for logistic regression
        beta = np.zeros(X.shape[1])
        lr = 0.01

        for _ in range(100):
            logits = X @ beta
            probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
            gradient = X.T @ (probs - y) / len(y)
            beta -= lr * gradient

        # Compute propensity scores
        logits = X @ beta
        propensity = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))

        return propensity

    def _compute_ate_sample(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        propensity: np.ndarray
    ) -> float:
        """Compute ATE for a single sample."""
        weights_treated = treatment / propensity
        weights_control = (1 - treatment) / (1 - propensity)

        ate_treated = np.sum(weights_treated * outcome) / (np.sum(weights_treated) + 1e-10)
        ate_control = np.sum(weights_control * outcome) / (np.sum(weights_control) + 1e-10)

        return ate_treated - ate_control


class CausalMarketAnalyzer:
    """
    Comprehensive causal analysis for market data.

    Combines multiple causal inference methods:
    - Granger Causality for temporal relationships
    - PC Algorithm for structure discovery
    - Treatment Effect Estimation for intervention analysis
    - Counterfactual reasoning
    """

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.causal_graph = CausalGraph()
        self.granger = GrangerCausality(max_lag=10)
        self.pc_algorithm = PCAlgorithm(alpha=0.05)
        self.treatment_estimator = TreatmentEffectEstimator()

        # Cache for discovered relationships
        self.intervention_effects: dict[tuple[str, str], InterventionEffect] = {}
        self.granger_results: dict[tuple[str, str], dict] = {}

        logger.info(f"CausalMarketAnalyzer initialized with {len(feature_names)} features")

    def discover_causal_structure(self, data: np.ndarray) -> CausalGraph:
        """
        Discover causal structure from market data.

        Args:
            data: Market data matrix (samples x features)

        Returns:
            CausalGraph with discovered relationships
        """
        logger.info("Discovering causal structure with PC algorithm")

        # Use PC algorithm for structure discovery
        self.causal_graph = self.pc_algorithm.learn(data, self.feature_names)

        # Enhance with Granger causality for time series
        self._enhance_with_granger(data)

        return self.causal_graph

    def _enhance_with_granger(self, data: np.ndarray):
        """Enhance causal graph with Granger causality results."""
        n_features = len(self.feature_names)

        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    continue

                result = self.granger.test(data[:, i], data[:, j])
                self.granger_results[(self.feature_names[i], self.feature_names[j])] = result

                if result["significant"]:
                    # Check if edge already exists
                    edge_exists = any(
                        e.source == self.feature_names[i] and e.target == self.feature_names[j]
                        for e in self.causal_graph.edges
                    )

                    if not edge_exists:
                        edge = CausalEdge(
                            source=self.feature_names[i],
                            target=self.feature_names[j],
                            strength=1 - result["p_value"],
                            confidence=1 - result["p_value"],
                            lag=result["lag"],
                        )
                        self.causal_graph.add_edge(edge)

    def estimate_treatment_effect(
        self,
        data: np.ndarray,
        treatment: str,
        outcome: str,
        confounders: list[str]
    ) -> InterventionEffect:
        """
        Estimate the causal effect of treatment on outcome.

        Args:
            data: Market data matrix
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            confounders: Names of confounding variables

        Returns:
            InterventionEffect with causal effect estimate
        """
        treatment_idx = self.feature_names.index(treatment)
        outcome_idx = self.feature_names.index(outcome)
        confounder_idx = [self.feature_names.index(c) for c in confounders]

        effect = self.treatment_estimator.estimate_ate(
            data, treatment_idx, outcome_idx, confounder_idx
        )

        effect.treatment = treatment
        effect.outcome = outcome

        self.intervention_effects[(treatment, outcome)] = effect
        return effect

    def counterfactual_analysis(
        self,
        observed_data: np.ndarray,
        intervention: dict[str, float]
    ) -> list[CounterfactualResult]:
        """
        Perform counterfactual analysis: "What would have happened if..."

        Args:
            observed_data: Observed market data
            intervention: Dictionary of variable -> counterfactual value

        Returns:
            List of CounterfactualResult for each outcome variable
        """
        results = []

        for var_name, cf_value in intervention.items():
            var_idx = self.feature_names.index(var_name)

            # Find children (effects) of this variable
            children = self.causal_graph.get_children(var_name)

            for child in children:
                child_idx = self.feature_names.index(child)

                # Estimate causal effect
                edge = next(
                    (e for e in self.causal_graph.edges
                     if e.source == var_name and e.target == child),
                    None
                )

                if edge:
                    # Simple linear counterfactual
                    observed_treatment = np.mean(observed_data[:, var_idx])
                    observed_outcome = np.mean(observed_data[:, child_idx])

                    # Estimate effect size from correlation
                    correlation = np.corrcoef(
                        observed_data[:, var_idx],
                        observed_data[:, child_idx]
                    )[0, 1]

                    treatment_change = cf_value - observed_treatment
                    outcome_std = np.std(observed_data[:, child_idx])
                    treatment_std = np.std(observed_data[:, var_idx])

                    cf_outcome = observed_outcome + correlation * (treatment_change / treatment_std) * outcome_std

                    result = CounterfactualResult(
                        scenario=f"{var_name}={cf_value:.4f}",
                        observed_outcome=float(observed_outcome),
                        counterfactual_outcome=float(cf_outcome),
                        effect=float(cf_outcome - observed_outcome),
                        probability=float(edge.confidence),
                    )
                    results.append(result)

        return results

    def get_causal_path(self, source: str, target: str) -> list[str]:
        """
        Find causal path from source to target.

        Args:
            source: Source variable name
            target: Target variable name

        Returns:
            List of variable names in causal path
        """
        # BFS to find path
        from collections import deque

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if current == target:
                return path

            for child in self.causal_graph.get_children(current):
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))

        return []  # No path found

    def identify_confounders(self, treatment: str, outcome: str) -> list[str]:
        """
        Identify potential confounders between treatment and outcome.

        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name

        Returns:
            List of potential confounder names
        """
        treatment_ancestors = self.causal_graph.get_ancestors(treatment)
        outcome_ancestors = self.causal_graph.get_ancestors(outcome)

        # Confounders are common ancestors
        confounders = treatment_ancestors.intersection(outcome_ancestors)

        return list(confounders)

    def get_summary(self) -> dict:
        """Get summary of causal analysis."""
        return {
            "num_features": len(self.feature_names),
            "num_causal_edges": len(self.causal_graph.edges),
            "num_granger_relationships": sum(
                1 for r in self.granger_results.values() if r.get("significant")
            ),
            "num_intervention_effects": len(self.intervention_effects),
            "graph": self.causal_graph.to_dict(),
        }
