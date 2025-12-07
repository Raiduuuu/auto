"""
Real-Time Performance Monitor for HRM Trading Bot

Provides comprehensive performance monitoring and analysis:
- Real-time metrics collection
- Performance trend analysis
- Anomaly detection
- Alert system
- Live dashboard data
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time."""
    timestamp: datetime
    accuracy: float
    profit: float
    risk: float
    sharpe_ratio: float
    win_rate: float
    avg_trade_duration: float
    total_trades: int
    open_positions: int
    drawdown: float
    latency_ms: float
    custom_metrics: dict = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: str  # "info", "warning", "critical"
    category: str
    message: str
    timestamp: datetime
    value: Optional[float] = None
    threshold: Optional[float] = None
    acknowledged: bool = False


@dataclass
class PerformanceAnalysis:
    """Analysis of performance trends."""
    accuracy_trend: float  # Slope of accuracy over time
    profit_trend: float  # Slope of profit over time
    risk_trend: float  # Slope of risk over time
    anomalies: list[dict]
    current_performance: MetricSnapshot
    health_score: float  # 0-1 overall health
    recommendations: list[str]


class AlertSystem:
    """
    Alert system for performance monitoring.
    """

    def __init__(self):
        self.alerts: deque = deque(maxlen=1000)
        self.alert_callbacks: list[Callable] = []
        self.alert_thresholds: dict[str, dict] = {
            "accuracy": {"warning": 0.45, "critical": 0.35},
            "drawdown": {"warning": 0.05, "critical": 0.10},
            "sharpe_ratio": {"warning": 0.5, "critical": 0.0},
            "latency_ms": {"warning": 500, "critical": 1000},
            "win_rate": {"warning": 0.40, "critical": 0.30},
        }
        self.alert_counter = 0

    def check_metrics(self, metrics: MetricSnapshot) -> list[Alert]:
        """Check metrics against thresholds and generate alerts."""
        new_alerts = []

        # Accuracy check
        if metrics.accuracy < self.alert_thresholds["accuracy"]["critical"]:
            new_alerts.append(self._create_alert(
                "critical", "accuracy",
                f"Critical: Accuracy dropped to {metrics.accuracy:.1%}",
                metrics.accuracy, self.alert_thresholds["accuracy"]["critical"]
            ))
        elif metrics.accuracy < self.alert_thresholds["accuracy"]["warning"]:
            new_alerts.append(self._create_alert(
                "warning", "accuracy",
                f"Warning: Accuracy at {metrics.accuracy:.1%}",
                metrics.accuracy, self.alert_thresholds["accuracy"]["warning"]
            ))

        # Drawdown check
        if metrics.drawdown > self.alert_thresholds["drawdown"]["critical"]:
            new_alerts.append(self._create_alert(
                "critical", "drawdown",
                f"Critical: Drawdown at {metrics.drawdown:.1%}",
                metrics.drawdown, self.alert_thresholds["drawdown"]["critical"]
            ))
        elif metrics.drawdown > self.alert_thresholds["drawdown"]["warning"]:
            new_alerts.append(self._create_alert(
                "warning", "drawdown",
                f"Warning: Drawdown at {metrics.drawdown:.1%}",
                metrics.drawdown, self.alert_thresholds["drawdown"]["warning"]
            ))

        # Sharpe ratio check
        if metrics.sharpe_ratio < self.alert_thresholds["sharpe_ratio"]["critical"]:
            new_alerts.append(self._create_alert(
                "critical", "sharpe_ratio",
                f"Critical: Sharpe ratio dropped to {metrics.sharpe_ratio:.2f}",
                metrics.sharpe_ratio, self.alert_thresholds["sharpe_ratio"]["critical"]
            ))
        elif metrics.sharpe_ratio < self.alert_thresholds["sharpe_ratio"]["warning"]:
            new_alerts.append(self._create_alert(
                "warning", "sharpe_ratio",
                f"Warning: Sharpe ratio at {metrics.sharpe_ratio:.2f}",
                metrics.sharpe_ratio, self.alert_thresholds["sharpe_ratio"]["warning"]
            ))

        # Latency check
        if metrics.latency_ms > self.alert_thresholds["latency_ms"]["critical"]:
            new_alerts.append(self._create_alert(
                "critical", "latency",
                f"Critical: Latency at {metrics.latency_ms:.0f}ms",
                metrics.latency_ms, self.alert_thresholds["latency_ms"]["critical"]
            ))
        elif metrics.latency_ms > self.alert_thresholds["latency_ms"]["warning"]:
            new_alerts.append(self._create_alert(
                "warning", "latency",
                f"Warning: Latency at {metrics.latency_ms:.0f}ms",
                metrics.latency_ms, self.alert_thresholds["latency_ms"]["warning"]
            ))

        # Store alerts
        for alert in new_alerts:
            self.alerts.append(alert)

        return new_alerts

    def _create_alert(
        self,
        severity: str,
        category: str,
        message: str,
        value: float,
        threshold: float
    ) -> Alert:
        """Create a new alert."""
        self.alert_counter += 1
        return Alert(
            id=f"alert_{self.alert_counter}",
            severity=severity,
            category=category,
            message=message,
            timestamp=datetime.now(),
            value=value,
            threshold=threshold,
        )

    def register_callback(self, callback: Callable):
        """Register callback for alert notifications."""
        self.alert_callbacks.append(callback)

    async def notify(self, alerts: list[Alert]):
        """Notify all registered callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alerts)
                else:
                    callback(alerts)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def get_recent_alerts(
        self,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> list[Alert]:
        """Get recent alerts, optionally filtered by severity."""
        alerts = list(self.alerts)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                break


class AnomalyDetector:
    """
    Detects anomalies in performance metrics.
    """

    def __init__(self, window_size: int = 100, threshold_std: float = 3.0):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.metric_history: dict[str, deque] = {}

    def add_observation(self, metric_name: str, value: float):
        """Add an observation for a metric."""
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = deque(maxlen=self.window_size)

        self.metric_history[metric_name].append(value)

    def detect(self, metric_name: str, value: float) -> Optional[dict]:
        """
        Detect if a value is anomalous.

        Returns:
            Anomaly info dict if anomalous, None otherwise
        """
        if metric_name not in self.metric_history:
            return None

        history = list(self.metric_history[metric_name])
        if len(history) < 20:
            return None

        mean = np.mean(history)
        std = np.std(history)

        if std == 0:
            return None

        z_score = abs(value - mean) / std

        if z_score > self.threshold_std:
            return {
                "metric": metric_name,
                "value": value,
                "mean": mean,
                "std": std,
                "z_score": z_score,
                "direction": "high" if value > mean else "low",
                "timestamp": datetime.now(),
            }

        return None

    def detect_all(self, metrics: MetricSnapshot) -> list[dict]:
        """Detect anomalies across all metrics."""
        anomalies = []

        metric_values = {
            "accuracy": metrics.accuracy,
            "profit": metrics.profit,
            "risk": metrics.risk,
            "sharpe_ratio": metrics.sharpe_ratio,
            "win_rate": metrics.win_rate,
            "latency_ms": metrics.latency_ms,
            "drawdown": metrics.drawdown,
        }

        for name, value in metric_values.items():
            self.add_observation(name, value)
            anomaly = self.detect(name, value)
            if anomaly:
                anomalies.append(anomaly)

        return anomalies


class TrendAnalyzer:
    """
    Analyzes trends in performance metrics.
    """

    def __init__(self, short_window: int = 20, long_window: int = 100):
        self.short_window = short_window
        self.long_window = long_window

    def calculate_trend(self, values: list[float]) -> float:
        """
        Calculate trend (slope) of values.

        Returns:
            Slope coefficient (positive = improving, negative = declining)
        """
        if len(values) < 10:
            return 0.0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)

        return float(slope)

    def analyze_momentum(self, values: list[float]) -> dict:
        """
        Analyze momentum of a metric.

        Returns:
            Dict with momentum metrics
        """
        if len(values) < self.short_window:
            return {"momentum": 0, "acceleration": 0}

        short_values = values[-self.short_window:]
        short_mean = np.mean(short_values)

        if len(values) >= self.long_window:
            long_values = values[-self.long_window:]
            long_mean = np.mean(long_values)
        else:
            long_mean = np.mean(values)

        # Momentum: short-term mean vs long-term mean
        momentum = (short_mean - long_mean) / (long_mean + 1e-10)

        # Acceleration: change in momentum
        if len(values) >= self.short_window * 2:
            prev_short = values[-self.short_window*2:-self.short_window]
            prev_momentum = (np.mean(prev_short) - long_mean) / (long_mean + 1e-10)
            acceleration = momentum - prev_momentum
        else:
            acceleration = 0.0

        return {
            "momentum": float(momentum),
            "acceleration": float(acceleration),
            "short_mean": float(short_mean),
            "long_mean": float(long_mean),
        }


class Dashboard:
    """
    Dashboard data provider for visualization.
    """

    def __init__(self):
        self.metrics_history: deque = deque(maxlen=10000)
        self.last_update: Optional[datetime] = None

    def update(self, metrics: MetricSnapshot):
        """Update dashboard with new metrics."""
        self.metrics_history.append(metrics)
        self.last_update = datetime.now()

    def get_current_summary(self) -> dict:
        """Get current summary for dashboard."""
        if not self.metrics_history:
            return {}

        current = self.metrics_history[-1]

        return {
            "timestamp": current.timestamp.isoformat(),
            "accuracy": current.accuracy,
            "profit": current.profit,
            "risk": current.risk,
            "sharpe_ratio": current.sharpe_ratio,
            "win_rate": current.win_rate,
            "total_trades": current.total_trades,
            "open_positions": current.open_positions,
            "drawdown": current.drawdown,
            "latency_ms": current.latency_ms,
        }

    def get_time_series(
        self,
        metric: str,
        period: timedelta = timedelta(hours=24)
    ) -> list[dict]:
        """Get time series data for a metric."""
        cutoff = datetime.now() - period
        series = []

        for snapshot in self.metrics_history:
            if snapshot.timestamp >= cutoff:
                value = getattr(snapshot, metric, None)
                if value is not None:
                    series.append({
                        "timestamp": snapshot.timestamp.isoformat(),
                        "value": value,
                    })

        return series

    def get_statistics(self, period: timedelta = timedelta(hours=24)) -> dict:
        """Get statistics for the period."""
        cutoff = datetime.now() - period
        relevant = [m for m in self.metrics_history if m.timestamp >= cutoff]

        if not relevant:
            return {}

        return {
            "accuracy": {
                "mean": np.mean([m.accuracy for m in relevant]),
                "min": min(m.accuracy for m in relevant),
                "max": max(m.accuracy for m in relevant),
                "std": np.std([m.accuracy for m in relevant]),
            },
            "profit": {
                "total": sum(m.profit for m in relevant),
                "mean": np.mean([m.profit for m in relevant]),
            },
            "sharpe_ratio": {
                "mean": np.mean([m.sharpe_ratio for m in relevant]),
            },
            "trades": {
                "total": relevant[-1].total_trades - relevant[0].total_trades if len(relevant) > 1 else 0,
            },
        }


class RealTimePerformanceMonitor:
    """
    Real-time performance monitoring and analysis system.

    Features:
    - Continuous metrics collection
    - Trend analysis
    - Anomaly detection
    - Alert system
    - Dashboard data
    """

    def __init__(
        self,
        collection_interval: float = 1.0,
        analysis_interval: float = 60.0,
    ):
        self.collection_interval = collection_interval
        self.analysis_interval = analysis_interval

        # Components
        self.alert_system = AlertSystem()
        self.anomaly_detector = AnomalyDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.dashboard = Dashboard()

        # Metrics buffer
        self.metrics_buffer: deque = deque(maxlen=10000)

        # State
        self.is_running = False
        self.last_analysis: Optional[datetime] = None
        self.collection_task: Optional[asyncio.Task] = None

        # Callbacks
        self.metric_collectors: list[Callable] = []

        logger.info("RealTimePerformanceMonitor initialized")

    def register_collector(self, collector: Callable):
        """
        Register a metric collector function.

        Collector should return a dict of metric name -> value.
        """
        self.metric_collectors.append(collector)

    async def collect_metrics(self, trading_system: Any) -> MetricSnapshot:
        """
        Collect current metrics from trading system.

        Args:
            trading_system: Trading system to collect from

        Returns:
            MetricSnapshot with current metrics
        """
        start_time = datetime.now()

        # Default metrics
        metrics_dict = {
            "accuracy": 0.5,
            "profit": 0.0,
            "risk": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.5,
            "avg_trade_duration": 0.0,
            "total_trades": 0,
            "open_positions": 0,
            "drawdown": 0.0,
        }

        # Collect from trading system
        if hasattr(trading_system, "get_metrics"):
            system_metrics = trading_system.get_metrics()
            metrics_dict.update(system_metrics)

        # Collect from registered collectors
        for collector in self.metric_collectors:
            try:
                if asyncio.iscoroutinefunction(collector):
                    collected = await collector()
                else:
                    collected = collector()
                if collected:
                    metrics_dict.update(collected)
            except Exception as e:
                logger.error(f"Metric collector failed: {e}")

        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        return MetricSnapshot(
            timestamp=datetime.now(),
            accuracy=metrics_dict.get("accuracy", 0.5),
            profit=metrics_dict.get("profit", 0.0),
            risk=metrics_dict.get("risk", 0.0),
            sharpe_ratio=metrics_dict.get("sharpe_ratio", 0.0),
            win_rate=metrics_dict.get("win_rate", 0.5),
            avg_trade_duration=metrics_dict.get("avg_trade_duration", 0.0),
            total_trades=metrics_dict.get("total_trades", 0),
            open_positions=metrics_dict.get("open_positions", 0),
            drawdown=metrics_dict.get("drawdown", 0.0),
            latency_ms=latency_ms,
            custom_metrics={k: v for k, v in metrics_dict.items()
                          if k not in ["accuracy", "profit", "risk", "sharpe_ratio",
                                      "win_rate", "avg_trade_duration", "total_trades",
                                      "open_positions", "drawdown"]},
        )

    def analyze_performance(self) -> Optional[PerformanceAnalysis]:
        """Analyze performance trends and health."""
        if len(self.metrics_buffer) < 100:
            return None

        recent_metrics = list(self.metrics_buffer)[-100:]

        # Calculate trends
        accuracy_values = [m.accuracy for m in recent_metrics]
        profit_values = [m.profit for m in recent_metrics]
        risk_values = [m.risk for m in recent_metrics]

        accuracy_trend = self.trend_analyzer.calculate_trend(accuracy_values)
        profit_trend = self.trend_analyzer.calculate_trend(profit_values)
        risk_trend = self.trend_analyzer.calculate_trend(risk_values)

        # Detect anomalies
        anomalies = self.anomaly_detector.detect_all(recent_metrics[-1])

        # Calculate health score
        health_factors = []

        # Accuracy health (higher is better)
        health_factors.append(min(recent_metrics[-1].accuracy / 0.6, 1.0))

        # Trend health (positive trends are better)
        trend_health = 0.5 + 0.25 * np.sign(accuracy_trend) + 0.25 * np.sign(profit_trend)
        health_factors.append(trend_health)

        # Risk health (lower risk is better)
        health_factors.append(max(0, 1 - recent_metrics[-1].drawdown / 0.1))

        # Anomaly health (fewer anomalies is better)
        health_factors.append(max(0, 1 - len(anomalies) * 0.2))

        health_score = np.mean(health_factors)

        # Generate recommendations
        recommendations = []
        if accuracy_trend < -0.001:
            recommendations.append("Consider reducing position sizes due to declining accuracy")
        if recent_metrics[-1].drawdown > 0.05:
            recommendations.append("High drawdown detected - review risk parameters")
        if risk_trend > 0.001:
            recommendations.append("Risk is increasing - consider tightening stop losses")
        if len(anomalies) > 0:
            recommendations.append(f"Detected {len(anomalies)} metric anomalies - investigate")

        return PerformanceAnalysis(
            accuracy_trend=accuracy_trend,
            profit_trend=profit_trend,
            risk_trend=risk_trend,
            anomalies=anomalies,
            current_performance=recent_metrics[-1],
            health_score=float(health_score),
            recommendations=recommendations,
        )

    async def monitor_performance(self, trading_system: Any):
        """
        Continuous performance monitoring loop.

        Args:
            trading_system: Trading system to monitor
        """
        self.is_running = True
        logger.info("Starting performance monitoring")

        while self.is_running:
            try:
                # Collect metrics
                current_metrics = await self.collect_metrics(trading_system)
                self.metrics_buffer.append(current_metrics)

                # Update dashboard
                self.dashboard.update(current_metrics)

                # Check alerts
                alerts = self.alert_system.check_metrics(current_metrics)
                if alerts:
                    await self.alert_system.notify(alerts)

                # Periodic analysis
                if self.last_analysis is None or \
                   (datetime.now() - self.last_analysis).total_seconds() >= self.analysis_interval:
                    analysis = self.analyze_performance()
                    if analysis:
                        logger.info(
                            f"Performance analysis: health={analysis.health_score:.2f}, "
                            f"accuracy_trend={analysis.accuracy_trend:.4f}"
                        )
                    self.last_analysis = datetime.now()

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")

            await asyncio.sleep(self.collection_interval)

    async def start(self, trading_system: Any):
        """Start monitoring in background."""
        if self.collection_task is not None:
            return

        self.collection_task = asyncio.create_task(
            self.monitor_performance(trading_system)
        )

    async def stop(self):
        """Stop monitoring."""
        self.is_running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
            self.collection_task = None

    def get_current_analysis(self) -> Optional[PerformanceAnalysis]:
        """Get current performance analysis."""
        return self.analyze_performance()

    def get_dashboard_data(self) -> dict:
        """Get dashboard data."""
        return {
            "summary": self.dashboard.get_current_summary(),
            "statistics": self.dashboard.get_statistics(),
            "recent_alerts": [
                {
                    "id": a.id,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in self.alert_system.get_recent_alerts(limit=10)
            ],
        }

    def get_state(self) -> dict:
        """Get monitor state."""
        return {
            "is_running": self.is_running,
            "metrics_count": len(self.metrics_buffer),
            "alerts_count": len(self.alert_system.alerts),
            "last_analysis": self.last_analysis.isoformat() if self.last_analysis else None,
        }
