"""
IR-Cost Referee Engine
A rule-based decision engine for comparing security and cost control options.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional


# =============================================================================
# MOCK CLOUD COST DATA GENERATOR
# =============================================================================
# Simulates realistic cloud cost patterns without requiring real API connections.
# Generates cost data that mimics AWS/Azure/GCP billing patterns.
# =============================================================================

@dataclass
class CloudCostData:
    """Simulated cloud cost metrics."""
    current_hourly_cost: float
    baseline_hourly_cost: float
    cost_spike_percent: float
    top_cost_drivers: List[Dict[str, Any]]
    anomaly_detected: bool
    anomaly_type: str
    estimated_daily_impact: float
    recommendation_urgency: str


def generate_mock_cloud_costs(
    scenario_type: str = "normal",
    asset_type: str = "compute"
) -> CloudCostData:
    """
    Generate realistic mock cloud cost data.
    
    Scenario types:
    - "normal": Typical usage, minor fluctuations
    - "spike": Sudden cost increase (possible attack or misconfiguration)
    - "gradual": Slowly increasing costs (resource creep)
    - "anomaly": Unusual pattern (cryptomining, data exfil, DDoS)
    """
    
    # Base costs by asset type ($/hour)
    base_costs = {
        "compute": 45.0,
        "database": 85.0,
        "api_gateway": 25.0,
        "storage": 15.0,
    }
    baseline = base_costs.get(asset_type, 50.0)
    
    # Generate scenario-specific data
    if scenario_type == "normal":
        variance = random.uniform(0.95, 1.15)
        current = baseline * variance
        spike_pct = (current / baseline) * 100
        anomaly = False
        anomaly_type = "none"
        urgency = "low"
        
    elif scenario_type == "spike":
        variance = random.uniform(2.5, 5.0)
        current = baseline * variance
        spike_pct = (current / baseline) * 100
        anomaly = True
        anomaly_type = "sudden_spike"
        urgency = "high"
        
    elif scenario_type == "gradual":
        variance = random.uniform(1.5, 2.2)
        current = baseline * variance
        spike_pct = (current / baseline) * 100
        anomaly = True
        anomaly_type = "gradual_increase"
        urgency = "medium"
        
    elif scenario_type == "anomaly":
        variance = random.uniform(3.0, 8.0)
        current = baseline * variance
        spike_pct = (current / baseline) * 100
        anomaly = True
        anomaly_type = random.choice(["cryptomining", "data_exfil", "ddos_traffic", "misconfiguration"])
        urgency = "critical"
        
    else:
        current = baseline
        spike_pct = 100.0
        anomaly = False
        anomaly_type = "none"
        urgency = "low"
    
    # Generate top cost drivers
    drivers = _generate_cost_drivers(asset_type, current, anomaly_type)
    
    return CloudCostData(
        current_hourly_cost=round(current, 2),
        baseline_hourly_cost=round(baseline, 2),
        cost_spike_percent=round(spike_pct, 1),
        top_cost_drivers=drivers,
        anomaly_detected=anomaly,
        anomaly_type=anomaly_type,
        estimated_daily_impact=round(current * 24, 2),
        recommendation_urgency=urgency,
    )


def _generate_cost_drivers(asset_type: str, current_cost: float, anomaly_type: str) -> List[Dict[str, Any]]:
    """Generate realistic cost breakdown by service."""
    
    if asset_type == "compute":
        drivers = [
            {"service": "EC2 Instances", "cost": current_cost * 0.45, "change": "+15%"},
            {"service": "EBS Volumes", "cost": current_cost * 0.20, "change": "+5%"},
            {"service": "Data Transfer", "cost": current_cost * 0.25, "change": "+180%" if anomaly_type else "+2%"},
            {"service": "CloudWatch", "cost": current_cost * 0.10, "change": "+3%"},
        ]
    elif asset_type == "database":
        drivers = [
            {"service": "RDS Instances", "cost": current_cost * 0.55, "change": "+8%"},
            {"service": "Storage", "cost": current_cost * 0.25, "change": "+12%"},
            {"service": "I/O Requests", "cost": current_cost * 0.15, "change": "+250%" if anomaly_type else "+5%"},
            {"service": "Backups", "cost": current_cost * 0.05, "change": "+1%"},
        ]
    elif asset_type == "api_gateway":
        drivers = [
            {"service": "API Calls", "cost": current_cost * 0.60, "change": "+300%" if anomaly_type else "+10%"},
            {"service": "Data Transfer", "cost": current_cost * 0.30, "change": "+45%"},
            {"service": "Caching", "cost": current_cost * 0.10, "change": "+2%"},
        ]
    else:
        drivers = [
            {"service": "Storage", "cost": current_cost * 0.70, "change": "+5%"},
            {"service": "Requests", "cost": current_cost * 0.20, "change": "+8%"},
            {"service": "Data Transfer", "cost": current_cost * 0.10, "change": "+3%"},
        ]
    
    # Round costs
    for d in drivers:
        d["cost"] = round(d["cost"], 2)
    
    return drivers


def format_cloud_costs(data: CloudCostData) -> str:
    """Format cloud cost data as readable output."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("CLOUD COST ANALYSIS")
    lines.append("=" * 80)
    
    # Cost summary
    lines.append(f"\nCurrent Hourly Cost:  ${data.current_hourly_cost:,.2f}")
    lines.append(f"Baseline Hourly Cost: ${data.baseline_hourly_cost:,.2f}")
    lines.append(f"Cost Spike:           {data.cost_spike_percent:.1f}%")
    lines.append(f"Estimated Daily Impact: ${data.estimated_daily_impact:,.2f}")
    
    # Anomaly status
    if data.anomaly_detected:
        lines.append(f"\n⚠️  ANOMALY DETECTED: {data.anomaly_type.replace('_', ' ').title()}")
        lines.append(f"    Urgency: {data.recommendation_urgency.upper()}")
    else:
        lines.append(f"\n✓ No anomalies detected")
    
    # Cost drivers
    lines.append("\n" + "-" * 40)
    lines.append("TOP COST DRIVERS")
    lines.append("-" * 40)
    for driver in data.top_cost_drivers:
        lines.append(f"  {driver['service']:<20} ${driver['cost']:>8,.2f}  ({driver['change']})")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def create_alert_from_cloud_data(cloud_data: CloudCostData, asset_type: str = "compute") -> 'AlertContext':
    """
    Create an AlertContext from cloud cost data.
    Maps cloud anomalies to security alert types.
    """
    # Map anomaly types to alert types
    alert_mapping = {
        "cryptomining": "malware",
        "data_exfil": "data_exfil",
        "ddos_traffic": "ddos",
        "misconfiguration": "cost_anomaly",
        "sudden_spike": "cost_anomaly",
        "gradual_increase": "cost_anomaly",
        "none": "cost_anomaly",
    }
    
    alert_type = alert_mapping.get(cloud_data.anomaly_type, "cost_anomaly")
    
    # Map urgency to severity
    severity_mapping = {
        "critical": 9,
        "high": 7,
        "medium": 5,
        "low": 3,
    }
    severity = severity_mapping.get(cloud_data.recommendation_urgency, 5)
    
    # Determine business criticality based on cost
    if cloud_data.estimated_daily_impact > 2000:
        criticality = "critical"
    elif cloud_data.estimated_daily_impact > 1000:
        criticality = "high"
    elif cloud_data.estimated_daily_impact > 500:
        criticality = "medium"
    else:
        criticality = "low"
    
    return AlertContext(
        alert_type=alert_type,
        severity=severity,
        asset_type=asset_type,
        business_criticality=criticality,
        cloud_cost_spike_percent=cloud_data.cost_spike_percent,
        traffic_pattern="anomalous" if cloud_data.anomaly_detected else "normal",
        sla_importance="high" if criticality in ["high", "critical"] else "medium",
        time_context="business_hours",
        incident_duration_minutes=random.randint(10, 120),
    )


@dataclass
class PerformanceMetrics:
    """Lightweight performance measurements in milliseconds."""
    option_evaluation_ms: float = 0.0
    verdict_generation_ms: float = 0.0
    sensitivity_analysis_ms: float = 0.0
    timeline_computation_ms: float = 0.0
    business_impact_ms: float = 0.0
    total_ms: float = 0.0
    
    def summary(self) -> str:
        """Return formatted performance summary."""
        return (
            f"Options: {self.option_evaluation_ms:.2f}ms | "
            f"Verdict: {self.verdict_generation_ms:.2f}ms | "
            f"Sensitivity: {self.sensitivity_analysis_ms:.2f}ms | "
            f"Timeline: {self.timeline_computation_ms:.2f}ms | "
            f"Impact: {self.business_impact_ms:.2f}ms | "
            f"Total: {self.total_ms:.2f}ms"
        )


@dataclass
class AlertContext:
    """Input context for the referee engine."""
    alert_type: str  # e.g., "intrusion", "ddos", "data_exfil", "cost_anomaly", "malware"
    severity: int  # 1-10
    asset_type: str  # e.g., "database", "api_gateway", "compute", "storage"
    business_criticality: str  # "low", "medium", "high", "critical"
    cloud_cost_spike_percent: float  # e.g., 150.0 means 150% increase
    traffic_pattern: str  # "normal", "spike", "sustained_high", "anomalous"
    sla_importance: str  # "low", "medium", "high"
    time_context: str  # "business_hours", "off_hours", "weekend", "maintenance"
    incident_duration_minutes: int = 0  # How long the incident has been active


# =============================================================================
# TIMELINE THRESHOLDS
# =============================================================================

TIMELINE_THRESHOLDS = {
    "early": 15,       # 0-15 minutes: early stage, favor monitoring
    "developing": 60,  # 15-60 minutes: developing, balanced approach
    "prolonged": 120,  # 60-120 minutes: prolonged, favor containment
    "critical": 240,   # >120 minutes: critical duration, aggressive action
}


def get_incident_phase(duration_minutes: int) -> str:
    """Determine incident phase based on duration."""
    if duration_minutes <= TIMELINE_THRESHOLDS["early"]:
        return "early"
    elif duration_minutes <= TIMELINE_THRESHOLDS["developing"]:
        return "developing"
    elif duration_minutes <= TIMELINE_THRESHOLDS["prolonged"]:
        return "prolonged"
    else:
        return "critical"


def get_timeline_weight_adjustments(phase: str) -> Dict[str, float]:
    """
    Get weight adjustments based on incident phase.
    Early: favor monitoring (lower security weight)
    Late: favor containment (higher security weight)
    """
    adjustments = {
        "early": {
            "security_risk": -0.10,      # Less aggressive early
            "business_downtime": 0.05,   # Protect uptime early
            "cost_impact": 0.05,         # Consider cost early
            "data_loss_probability": 0.0,
        },
        "developing": {
            "security_risk": 0.0,
            "business_downtime": 0.0,
            "cost_impact": 0.0,
            "data_loss_probability": 0.0,
        },
        "prolonged": {
            "security_risk": 0.10,       # More aggressive
            "business_downtime": -0.05,  # Accept some downtime
            "cost_impact": -0.05,        # Cost less important
            "data_loss_probability": 0.05,
        },
        "critical": {
            "security_risk": 0.15,       # Very aggressive
            "business_downtime": -0.10,  # Accept downtime
            "cost_impact": -0.10,        # Cost not a factor
            "data_loss_probability": 0.10,
        },
    }
    return adjustments.get(phase, adjustments["developing"])


@dataclass
class TimelineRecommendation:
    """Timeline-aware recommendation with escalation path."""
    current_phase: str
    current_security: str
    current_cost: str
    next_phase: str
    next_security: str
    next_cost: str
    escalation_trigger: str
    time_to_escalation: int  # minutes until next phase


@dataclass
class FastPathDecision:
    """Fast-path decision when thresholds are clearly met."""
    triggered: bool
    rule_name: str
    security_action: str
    cost_action: str
    explanation: str
    conditions_met: List[str]


# =============================================================================
# FAST-PATH RULES
# =============================================================================

FAST_PATH_RULES = [
    {
        "name": "CRITICAL_SEVERITY_HIGH_ASSET",
        "description": "Maximum severity on high-value asset",
        "conditions": lambda ctx: (
            ctx.severity >= 9 and ctx.business_criticality in ["high", "critical"]
        ),
        "security_action": "Isolate Immediately",
        "cost_action": "Maintain Current State",
        "explanation": (
            "FAST-PATH TRIGGERED: Severity {severity}/10 on {criticality} asset. "
            "Immediate isolation required to prevent damage. "
            "Cost considerations suspended until threat is contained."
        ),
    },
    {
        "name": "ACTIVE_MALWARE_PRODUCTION",
        "description": "Active malware on production workload",
        "conditions": lambda ctx: (
            ctx.alert_type == "malware" and ctx.business_criticality in ["high", "critical"]
        ),
        "security_action": "Isolate Immediately",
        "cost_action": "Maintain Current State",
        "explanation": (
            "FAST-PATH TRIGGERED: Active malware detected on {criticality} production workload. "
            "Immediate isolation is mandatory. "
            "No cost optimization until malware is eradicated."
        ),
    },
    {
        "name": "DATA_EXFIL_CRITICAL",
        "description": "Data exfiltration on critical database",
        "conditions": lambda ctx: (
            ctx.alert_type == "data_exfil" and ctx.asset_type == "database"
            and ctx.severity >= 8
        ),
        "security_action": "Isolate Immediately",
        "cost_action": "Maintain Current State",
        "explanation": (
            "FAST-PATH TRIGGERED: Data exfiltration detected on database (severity {severity}/10). "
            "Immediate isolation to stop data loss. "
            "Cost impact is secondary to data protection."
        ),
    },
    {
        "name": "DDOS_CRITICAL_SLA",
        "description": "DDoS attack on critical SLA system",
        "conditions": lambda ctx: (
            ctx.alert_type == "ddos" and ctx.severity >= 9
            and ctx.sla_importance == "high"
        ),
        "security_action": "Failover to Backup",
        "cost_action": "Maintain Current State",
        "explanation": (
            "FAST-PATH TRIGGERED: DDoS attack (severity {severity}/10) on high-SLA system. "
            "Immediate failover to backup region required. "
            "SLA preservation takes priority over cost."
        ),
    },
]


def check_fast_path(context: AlertContext) -> Optional[FastPathDecision]:
    """
    Check if any fast-path rule applies.
    Returns FastPathDecision if triggered, None otherwise.
    """
    for rule in FAST_PATH_RULES:
        if rule["conditions"](context):
            # Build conditions met list
            conditions_met = []
            if context.severity >= 9:
                conditions_met.append(f"severity={context.severity}/10")
            if context.business_criticality in ["high", "critical"]:
                conditions_met.append(f"criticality={context.business_criticality}")
            if context.alert_type in ["malware", "data_exfil", "ddos"]:
                conditions_met.append(f"alert_type={context.alert_type}")
            if context.asset_type == "database":
                conditions_met.append(f"asset_type={context.asset_type}")
            if context.sla_importance == "high":
                conditions_met.append(f"sla={context.sla_importance}")
            
            # Format explanation with context values
            explanation = rule["explanation"].format(
                severity=context.severity,
                criticality=context.business_criticality,
                alert_type=context.alert_type,
                asset_type=context.asset_type,
            )
            
            return FastPathDecision(
                triggered=True,
                rule_name=rule["name"],
                security_action=rule["security_action"],
                cost_action=rule["cost_action"],
                explanation=explanation,
                conditions_met=conditions_met,
            )
    
    return None


def format_fast_path(decision: FastPathDecision, context: AlertContext) -> str:
    """Format fast-path decision as readable output."""
    lines = []
    lines.append("=" * 100)
    lines.append("IR-COST REFEREE - FAST-PATH DECISION")
    lines.append("=" * 100)
    
    lines.append(f"\nAlert: {context.alert_type} | Severity: {context.severity}/10")
    lines.append(f"Asset: {context.asset_type} | Criticality: {context.business_criticality}")
    lines.append(f"SLA: {context.sla_importance} | Time: {context.time_context}")
    
    lines.append("\n" + "!" * 80)
    lines.append(f"FAST-PATH RULE: {decision.rule_name}")
    lines.append("!" * 80)
    
    lines.append(f"\nConditions met: {', '.join(decision.conditions_met)}")
    lines.append(f"\n{decision.explanation}")
    
    lines.append("\n" + "-" * 60)
    lines.append("IMMEDIATE ACTIONS REQUIRED")
    lines.append("-" * 60)
    lines.append(f">> Security: {decision.security_action}")
    lines.append(f">> Cost: {decision.cost_action}")
    
    lines.append("\n" + "-" * 60)
    lines.append("WHY FAST-PATH?")
    lines.append("-" * 60)
    lines.append("- Conditions clearly exceed safety thresholds")
    lines.append("- Full comparison would delay critical response")
    lines.append("- Default safe action is the only appropriate choice")
    lines.append("- Cost optimization is suspended until threat is resolved")
    
    lines.append("\n" + "=" * 100)
    lines.append("Note: Full comparison skipped. Run with --force-full to override.")
    lines.append("=" * 100)
    
    return "\n".join(lines)


@dataclass
class OptionScore:
    """Computed scores for a single option."""
    security_risk: float  # 0-100, lower is better
    business_downtime: float  # estimated minutes
    cost_impact: float  # estimated $ impact
    data_loss_probability: float  # 0-100 percentage


@dataclass
class NormalizedScore:
    """Deterministic 0-10 normalized scores for an option."""
    security_risk: float  # 0-10, lower is better
    business_downtime: float  # 0-10, lower is better
    cost_impact: float  # 0-10, lower is better
    data_loss_probability: float  # 0-10, lower is better
    
    def total(self) -> float:
        """Sum of all scores (lower is better)."""
        return self.security_risk + self.business_downtime + self.cost_impact + self.data_loss_probability


@dataclass
class WeightedScore:
    """Weighted scores based on context priorities."""
    security_risk: float
    business_downtime: float
    cost_impact: float
    data_loss_probability: float
    weights_used: Dict[str, float]
    
    def total(self) -> float:
        """Weighted total (lower is better)."""
        return self.security_risk + self.business_downtime + self.cost_impact + self.data_loss_probability


# Scoring bounds for deterministic normalization (0-10 scale)
SCORE_BOUNDS = {
    "security_risk": {"min": 0, "max": 100},  # 0-100 risk score
    "business_downtime": {"min": 0, "max": 300},  # 0-300 minutes
    "cost_impact": {"min": -2000, "max": 10000},  # -$2000 to $10000
    "data_loss_probability": {"min": 0, "max": 100},  # 0-100%
}


# Security response options
SECURITY_OPTIONS = {
    "isolate_immediately": {
        "name": "Isolate Immediately",
        "description": "Quarantine affected assets from network",
        "base_security_risk": 10,
        "base_downtime": 30,
        "base_cost": 500,
        "base_data_loss": 5,
    },
    "monitor_and_alert": {
        "name": "Monitor and Alert",
        "description": "Increase monitoring, alert on-call team",
        "base_security_risk": 60,
        "base_downtime": 0,
        "base_cost": 50,
        "base_data_loss": 40,
    },
    "rate_limit": {
        "name": "Rate Limit Traffic",
        "description": "Apply rate limiting to affected endpoints",
        "base_security_risk": 35,
        "base_downtime": 5,
        "base_cost": 100,
        "base_data_loss": 20,
    },
    "failover_to_backup": {
        "name": "Failover to Backup",
        "description": "Switch to backup systems/region",
        "base_security_risk": 15,
        "base_downtime": 10,
        "base_cost": 2000,
        "base_data_loss": 8,
    },
    "block_source": {
        "name": "Block Source IPs",
        "description": "Block identified malicious sources",
        "base_security_risk": 25,
        "base_downtime": 2,
        "base_cost": 75,
        "base_data_loss": 15,
    },
}

# Cost control options
COST_OPTIONS = {
    "scale_down_immediately": {
        "name": "Scale Down Immediately",
        "description": "Reduce capacity to baseline levels",
        "base_security_risk": 20,
        "base_downtime": 15,
        "base_cost": -500,
        "base_data_loss": 10,
    },
    "enable_spot_instances": {
        "name": "Enable Spot Instances",
        "description": "Switch to spot/preemptible instances",
        "base_security_risk": 15,
        "base_downtime": 20,
        "base_cost": -300,
        "base_data_loss": 5,
    },
    "throttle_non_critical": {
        "name": "Throttle Non-Critical",
        "description": "Reduce resources for non-critical workloads",
        "base_security_risk": 10,
        "base_downtime": 5,
        "base_cost": -200,
        "base_data_loss": 2,
    },
    "maintain_current": {
        "name": "Maintain Current State",
        "description": "Keep current configuration, absorb costs",
        "base_security_risk": 5,
        "base_downtime": 0,
        "base_cost": 0,
        "base_data_loss": 0,
    },
    "schedule_review": {
        "name": "Schedule Review",
        "description": "Flag for review, no immediate action",
        "base_security_risk": 8,
        "base_downtime": 0,
        "base_cost": -50,
        "base_data_loss": 3,
    },
}


def _criticality_multiplier(criticality: str) -> float:
    """Returns multiplier based on business criticality."""
    return {
        "low": 0.5,
        "medium": 1.0,
        "high": 1.5,
        "critical": 2.5,
    }.get(criticality, 1.0)


def _sla_multiplier(sla: str) -> float:
    """Returns multiplier based on SLA importance."""
    return {
        "low": 0.7,
        "medium": 1.0,
        "high": 1.8,
    }.get(sla, 1.0)


def _time_multiplier(time_ctx: str) -> float:
    """Returns multiplier based on time context."""
    return {
        "business_hours": 1.5,
        "off_hours": 0.8,
        "weekend": 0.6,
        "maintenance": 0.4,
    }.get(time_ctx, 1.0)


def _traffic_risk_modifier(pattern: str) -> float:
    """Returns risk modifier based on traffic pattern."""
    return {
        "normal": 0.0,
        "spike": 10.0,
        "sustained_high": 15.0,
        "anomalous": 25.0,
    }.get(pattern, 0.0)


def _asset_cost_modifier(asset: str) -> float:
    """Returns cost modifier based on asset type."""
    return {
        "database": 1.5,
        "api_gateway": 1.2,
        "compute": 1.0,
        "storage": 0.8,
    }.get(asset, 1.0)


def compute_option_score(
    option: Dict[str, Any],
    context: AlertContext
) -> OptionScore:
    """Compute scores for a single option given the alert context."""
    
    crit_mult = _criticality_multiplier(context.business_criticality)
    sla_mult = _sla_multiplier(context.sla_importance)
    time_mult = _time_multiplier(context.time_context)
    traffic_mod = _traffic_risk_modifier(context.traffic_pattern)
    asset_mod = _asset_cost_modifier(context.asset_type)
    
    severity_factor = context.severity / 10.0
    cost_spike_factor = max(1.0, context.cloud_cost_spike_percent / 100.0)
    
    # Security risk: base + severity impact + traffic modifier
    security_risk = (
        option["base_security_risk"] * (1 - severity_factor * 0.3)
        + traffic_mod
        + (context.severity * 2)
    )
    security_risk = min(100, max(0, security_risk))
    
    # Business downtime: base * criticality * SLA * time context
    business_downtime = (
        option["base_downtime"] * crit_mult * sla_mult * time_mult
    )
    
    # Cost impact: base * asset modifier * cost spike factor
    cost_impact = option["base_cost"] * asset_mod * cost_spike_factor
    
    # Data loss probability: base + severity modifier
    data_loss = (
        option["base_data_loss"]
        + (severity_factor * 15)
        - (5 if context.time_context == "maintenance" else 0)
    )
    data_loss = min(100, max(0, data_loss))
    
    return OptionScore(
        security_risk=round(security_risk, 2),
        business_downtime=round(business_downtime, 2),
        cost_impact=round(cost_impact, 2),
        data_loss_probability=round(data_loss, 2),
    )


def normalize_to_ten(value: float, metric: str) -> float:
    """
    Deterministically normalize a raw score to 0-10 scale.
    Uses fixed bounds for consistency across all analyses.
    """
    bounds = SCORE_BOUNDS[metric]
    min_val, max_val = bounds["min"], bounds["max"]
    
    # Clamp value to bounds
    clamped = max(min_val, min(max_val, value))
    
    # Normalize to 0-10
    if max_val == min_val:
        return 5.0
    
    normalized = ((clamped - min_val) / (max_val - min_val)) * 10.0
    return round(normalized, 2)


def compute_normalized_score(raw: OptionScore) -> NormalizedScore:
    """Convert raw scores to deterministic 0-10 scale."""
    return NormalizedScore(
        security_risk=normalize_to_ten(raw.security_risk, "security_risk"),
        business_downtime=normalize_to_ten(raw.business_downtime, "business_downtime"),
        cost_impact=normalize_to_ten(raw.cost_impact, "cost_impact"),
        data_loss_probability=normalize_to_ten(raw.data_loss_probability, "data_loss_probability"),
    )


def compute_context_weights(context: AlertContext) -> Dict[str, float]:
    """
    Compute deterministic weights based on context.
    Weights are explainable and adapt to business_criticality, SLA, severity, and timeline.
    """
    # Base weights (sum to 1.0)
    w_security = 0.30
    w_downtime = 0.25
    w_cost = 0.25
    w_data_loss = 0.20
    
    # Severity adjustment (1-10 scale)
    # High severity (>=7): security and data loss matter more
    # Low severity (<=3): cost matters more
    severity_factor = context.severity / 10.0
    if severity_factor >= 0.7:
        w_security += 0.10
        w_data_loss += 0.05
        w_cost -= 0.10
        w_downtime -= 0.05
    elif severity_factor <= 0.3:
        w_security -= 0.05
        w_cost += 0.10
        w_downtime -= 0.05
    
    # Business criticality adjustment
    crit_adjustments = {
        "low": {"security": -0.05, "downtime": -0.05, "cost": 0.10},
        "medium": {"security": 0, "downtime": 0, "cost": 0},
        "high": {"security": 0.05, "downtime": 0.05, "cost": -0.05},
        "critical": {"security": 0.10, "downtime": 0.10, "cost": -0.10},
    }
    adj = crit_adjustments.get(context.business_criticality, {})
    w_security += adj.get("security", 0)
    w_downtime += adj.get("downtime", 0)
    w_cost += adj.get("cost", 0)
    
    # SLA importance adjustment
    sla_adjustments = {
        "low": {"downtime": -0.05, "cost": 0.05},
        "medium": {"downtime": 0, "cost": 0},
        "high": {"downtime": 0.10, "cost": -0.05},
    }
    sla_adj = sla_adjustments.get(context.sla_importance, {})
    w_downtime += sla_adj.get("downtime", 0)
    w_cost += sla_adj.get("cost", 0)
    
    # Timeline adjustment (incident duration)
    phase = get_incident_phase(context.incident_duration_minutes)
    timeline_adj = get_timeline_weight_adjustments(phase)
    w_security += timeline_adj["security_risk"]
    w_downtime += timeline_adj["business_downtime"]
    w_cost += timeline_adj["cost_impact"]
    w_data_loss += timeline_adj["data_loss_probability"]
    
    # Ensure no negative weights
    w_security = max(0.05, w_security)
    w_downtime = max(0.05, w_downtime)
    w_cost = max(0.05, w_cost)
    w_data_loss = max(0.05, w_data_loss)
    
    # Normalize to sum to 1.0
    total = w_security + w_downtime + w_cost + w_data_loss
    return {
        "security_risk": round(w_security / total, 3),
        "business_downtime": round(w_downtime / total, 3),
        "cost_impact": round(w_cost / total, 3),
        "data_loss_probability": round(w_data_loss / total, 3),
    }


def compute_weighted_score(normalized: NormalizedScore, weights: Dict[str, float]) -> WeightedScore:
    """Apply weights to normalized scores."""
    return WeightedScore(
        security_risk=round(normalized.security_risk * weights["security_risk"], 3),
        business_downtime=round(normalized.business_downtime * weights["business_downtime"], 3),
        cost_impact=round(normalized.cost_impact * weights["cost_impact"], 3),
        data_loss_probability=round(normalized.data_loss_probability * weights["data_loss_probability"], 3),
        weights_used=weights,
    )


# =============================================================================
# GUARDRAILS - Safety Rules to Prevent Unsafe Recommendations
# =============================================================================
# 
# These guardrails enforce safety constraints that override scoring results.
# They ensure the referee never makes recommendations that could cause harm.
#
# GUARDRAIL RULES:
# 1. NO_SHUTDOWN_LOW_SEVERITY: Never recommend full shutdown/isolation for 
#    low severity incidents (severity <= 3). Rationale: Disproportionate 
#    response causes unnecessary business disruption.
#
# 2. NO_MONITOR_ONLY_MALWARE: Never recommend monitoring-only for confirmed 
#    malware. Rationale: Malware requires active containment; passive 
#    monitoring allows spread and damage.
#
# 3. PII_DATA_PROTECTION_PRIORITY: Always favor data protection over cost 
#    when PII is involved (database assets with data exfiltration). 
#    Rationale: Data breach costs (legal, reputation) far exceed operational costs.
#
# Each guardrail returns:
# - triggered: bool - whether the guardrail was activated
# - blocked_option: str - the option that was blocked
# - replacement_option: str - the safer alternative
# - reason: str - explanation for the override
# =============================================================================

@dataclass
class GuardrailViolation:
    """Records when a guardrail blocks an unsafe recommendation."""
    rule_name: str
    blocked_option: str
    replacement_option: str
    reason: str


def check_guardrails(
    context: AlertContext,
    recommended_security: str,
    recommended_cost: str,
    security_options: Dict[str, Any],
    cost_options: Dict[str, Any]
) -> Tuple[str, str, List[GuardrailViolation]]:
    """
    Check all guardrails and return safe recommendations.
    
    Returns:
        Tuple of (safe_security_option, safe_cost_option, list_of_violations)
    """
    violations = []
    safe_security = recommended_security
    safe_cost = recommended_cost
    
    # -------------------------------------------------------------------------
    # GUARDRAIL 1: NO_SHUTDOWN_LOW_SEVERITY
    # Never recommend full shutdown/isolation for low severity incidents.
    # Low severity (<=3) should use proportionate responses.
    # -------------------------------------------------------------------------
    aggressive_actions = ["Isolate Immediately", "Failover to Backup"]
    if context.severity <= 3 and recommended_security in aggressive_actions:
        # Find a less aggressive alternative
        alternatives = ["Block Source IPs", "Rate Limit Traffic", "Monitor and Alert"]
        for alt in alternatives:
            alt_key = _find_option_key(security_options, alt)
            if alt_key:
                safe_security = alt
                violations.append(GuardrailViolation(
                    rule_name="NO_SHUTDOWN_LOW_SEVERITY",
                    blocked_option=recommended_security,
                    replacement_option=alt,
                    reason=(
                        f"Severity {context.severity}/10 is too low for aggressive action. "
                        f"'{recommended_security}' would cause disproportionate business disruption. "
                        f"Using '{alt}' as a measured response."
                    ),
                ))
                break
    
    # -------------------------------------------------------------------------
    # GUARDRAIL 2: NO_MONITOR_ONLY_MALWARE
    # Never recommend monitoring-only for confirmed malware.
    # Malware requires active containment to prevent spread.
    # -------------------------------------------------------------------------
    passive_actions = ["Monitor and Alert", "Schedule Review"]
    if context.alert_type == "malware" and recommended_security in passive_actions:
        # Force active containment for malware
        safe_security = "Block Source IPs"  # Minimum active response
        if context.severity >= 7:
            safe_security = "Isolate Immediately"  # Stronger for high severity
        
        violations.append(GuardrailViolation(
            rule_name="NO_MONITOR_ONLY_MALWARE",
            blocked_option=recommended_security,
            replacement_option=safe_security,
            reason=(
                f"Confirmed malware requires active containment. "
                f"'{recommended_security}' would allow malware to spread and cause damage. "
                f"Using '{safe_security}' to contain the threat."
            ),
        ))
    
    # -------------------------------------------------------------------------
    # GUARDRAIL 3: PII_DATA_PROTECTION_PRIORITY
    # Always favor data protection over cost when PII is involved.
    # Database + data_exfil = potential PII breach, which has severe consequences.
    # -------------------------------------------------------------------------
    cost_cutting_actions = ["Scale Down Immediately", "Enable Spot Instances", "Throttle Non-Critical"]
    pii_risk = (
        context.asset_type == "database" and 
        context.alert_type in ["data_exfil", "intrusion"] and
        context.severity >= 5
    )
    
    if pii_risk:
        # Ensure security is aggressive enough for PII protection
        if recommended_security in passive_actions:
            safe_security = "Isolate Immediately"
            violations.append(GuardrailViolation(
                rule_name="PII_DATA_PROTECTION_PRIORITY",
                blocked_option=recommended_security,
                replacement_option=safe_security,
                reason=(
                    f"Database under {context.alert_type} attack indicates PII risk. "
                    f"Data protection takes absolute priority over operational concerns. "
                    f"Using '{safe_security}' to protect sensitive data."
                ),
            ))
        
        # Block cost-cutting that could compromise data protection
        if recommended_cost in cost_cutting_actions:
            safe_cost = "Maintain Current State"
            violations.append(GuardrailViolation(
                rule_name="PII_DATA_PROTECTION_PRIORITY",
                blocked_option=recommended_cost,
                replacement_option=safe_cost,
                reason=(
                    f"PII risk detected on database. Cost optimization suspended. "
                    f"'{recommended_cost}' could compromise data protection capabilities. "
                    f"Maintaining full capacity until threat is resolved."
                ),
            ))
    
    return safe_security, safe_cost, violations


def format_guardrail_violations(violations: List[GuardrailViolation]) -> str:
    """Format guardrail violations as readable output."""
    if not violations:
        return ""
    
    lines = []
    lines.append("\n" + "!" * 80)
    lines.append("GUARDRAIL OVERRIDES APPLIED")
    lines.append("!" * 80)
    lines.append("\nThe following safety rules modified the recommendations:")
    
    for v in violations:
        lines.append(f"\n>> Rule: {v.rule_name}")
        lines.append(f"   Blocked: {v.blocked_option}")
        lines.append(f"   Replaced with: {v.replacement_option}")
        lines.append(f"   Reason: {v.reason}")
    
    lines.append("\n" + "!" * 80)
    return "\n".join(lines)


def compute_timeline_recommendation(context: AlertContext) -> TimelineRecommendation:
    """
    Compute current recommendation and what happens if incident persists.
    """
    current_phase = get_incident_phase(context.incident_duration_minutes)
    
    # Get current recommendation
    comparison = compare_options(context)
    current_sec_key = min(
        comparison["security_options"],
        key=lambda k: comparison["security_options"][k]["weighted_scores"]["total"]
    )
    current_cost_key = min(
        comparison["cost_options"],
        key=lambda k: comparison["cost_options"][k]["weighted_scores"]["total"]
    )
    current_security = comparison["security_options"][current_sec_key]["name"]
    current_cost = comparison["cost_options"][current_cost_key]["name"]
    
    # Determine next phase and time to escalation
    phase_order = ["early", "developing", "prolonged", "critical"]
    current_idx = phase_order.index(current_phase)
    
    if current_idx < len(phase_order) - 1:
        next_phase = phase_order[current_idx + 1]
        threshold_key = next_phase if next_phase != "critical" else "prolonged"
        next_threshold = TIMELINE_THRESHOLDS.get(threshold_key, 240)
        if next_phase == "critical":
            next_threshold = TIMELINE_THRESHOLDS["critical"]
        time_to_escalation = max(0, next_threshold - context.incident_duration_minutes)
        escalation_trigger = f"If incident continues for {time_to_escalation} more minutes"
    else:
        next_phase = "critical"
        time_to_escalation = 0
        escalation_trigger = "Already at maximum escalation level"
    
    # Simulate next phase recommendation
    next_context = AlertContext(
        alert_type=context.alert_type,
        severity=min(10, context.severity + 1),  # Severity tends to increase
        asset_type=context.asset_type,
        business_criticality=context.business_criticality,
        cloud_cost_spike_percent=context.cloud_cost_spike_percent * 1.2,  # Costs accumulate
        traffic_pattern=context.traffic_pattern,
        sla_importance=context.sla_importance,
        time_context=context.time_context,
        incident_duration_minutes=TIMELINE_THRESHOLDS.get(next_phase, 240) + 1,
    )
    
    next_comparison = compare_options(next_context)
    next_sec_key = min(
        next_comparison["security_options"],
        key=lambda k: next_comparison["security_options"][k]["weighted_scores"]["total"]
    )
    next_cost_key = min(
        next_comparison["cost_options"],
        key=lambda k: next_comparison["cost_options"][k]["weighted_scores"]["total"]
    )
    next_security = next_comparison["security_options"][next_sec_key]["name"]
    next_cost = next_comparison["cost_options"][next_cost_key]["name"]
    
    return TimelineRecommendation(
        current_phase=current_phase,
        current_security=current_security,
        current_cost=current_cost,
        next_phase=next_phase,
        next_security=next_security,
        next_cost=next_cost,
        escalation_trigger=escalation_trigger,
        time_to_escalation=time_to_escalation,
    )


def compare_options(context: AlertContext) -> Dict[str, Any]:
    """
    Compare all security and cost options for the given context.
    Returns structured comparison with raw, normalized (0-10), and weighted scores.
    """
    weights = compute_context_weights(context)
    phase = get_incident_phase(context.incident_duration_minutes)
    
    results = {
        "context": {
            "alert_type": context.alert_type,
            "severity": context.severity,
            "asset_type": context.asset_type,
            "business_criticality": context.business_criticality,
            "cloud_cost_spike_percent": context.cloud_cost_spike_percent,
            "traffic_pattern": context.traffic_pattern,
            "sla_importance": context.sla_importance,
            "time_context": context.time_context,
            "incident_duration_minutes": context.incident_duration_minutes,
            "incident_phase": phase,
        },
        "weights": weights,
        "weight_explanation": _explain_weights(weights, context),
        "security_options": {},
        "cost_options": {},
    }
    
    # Compute scores for security options
    for key, option in SECURITY_OPTIONS.items():
        raw = compute_option_score(option, context)
        normalized = compute_normalized_score(raw)
        weighted = compute_weighted_score(normalized, weights)
        
        results["security_options"][key] = {
            "name": option["name"],
            "description": option["description"],
            "scores": {
                "security_risk": raw.security_risk,
                "business_downtime": raw.business_downtime,
                "cost_impact": raw.cost_impact,
                "data_loss_probability": raw.data_loss_probability,
            },
            "normalized_scores": {
                "security_risk": normalized.security_risk,
                "business_downtime": normalized.business_downtime,
                "cost_impact": normalized.cost_impact,
                "data_loss_probability": normalized.data_loss_probability,
                "total": normalized.total(),
            },
            "weighted_scores": {
                "security_risk": weighted.security_risk,
                "business_downtime": weighted.business_downtime,
                "cost_impact": weighted.cost_impact,
                "data_loss_probability": weighted.data_loss_probability,
                "total": weighted.total(),
            },
        }
    
    # Compute scores for cost options
    for key, option in COST_OPTIONS.items():
        raw = compute_option_score(option, context)
        normalized = compute_normalized_score(raw)
        weighted = compute_weighted_score(normalized, weights)
        
        results["cost_options"][key] = {
            "name": option["name"],
            "description": option["description"],
            "scores": {
                "security_risk": raw.security_risk,
                "business_downtime": raw.business_downtime,
                "cost_impact": raw.cost_impact,
                "data_loss_probability": raw.data_loss_probability,
            },
            "normalized_scores": {
                "security_risk": normalized.security_risk,
                "business_downtime": normalized.business_downtime,
                "cost_impact": normalized.cost_impact,
                "data_loss_probability": normalized.data_loss_probability,
                "total": normalized.total(),
            },
            "weighted_scores": {
                "security_risk": weighted.security_risk,
                "business_downtime": weighted.business_downtime,
                "cost_impact": weighted.cost_impact,
                "data_loss_probability": weighted.data_loss_probability,
                "total": weighted.total(),
            },
        }
    
    return results


def _explain_weights(weights: Dict[str, float], context: AlertContext) -> str:
    """Generate explanation for why weights were set this way."""
    parts = []
    
    # Timeline explanation
    phase = get_incident_phase(context.incident_duration_minutes)
    if phase == "early":
        parts.append(f"Early incident ({context.incident_duration_minutes}min) favors monitoring")
    elif phase == "developing":
        parts.append(f"Developing incident ({context.incident_duration_minutes}min) uses balanced approach")
    elif phase == "prolonged":
        parts.append(f"Prolonged incident ({context.incident_duration_minutes}min) favors containment")
    elif phase == "critical":
        parts.append(f"Critical duration ({context.incident_duration_minutes}min) requires aggressive action")
    
    # Severity explanation
    if context.severity >= 7:
        parts.append(f"High severity ({context.severity}/10) increases security weight")
    elif context.severity <= 3:
        parts.append(f"Low severity ({context.severity}/10) increases cost weight")
    
    # Criticality explanation
    if context.business_criticality == "critical":
        parts.append("Critical business asset prioritizes security and uptime")
    elif context.business_criticality == "high":
        parts.append("High criticality elevates security and uptime weights")
    elif context.business_criticality == "low":
        parts.append("Low criticality allows more cost optimization")
    
    # SLA explanation
    if context.sla_importance == "high":
        parts.append("High SLA importance increases downtime weight")
    elif context.sla_importance == "low":
        parts.append("Low SLA importance reduces downtime weight")
    
    if not parts:
        parts.append("Balanced weights for medium-priority context")
    
    return ". ".join(parts) + "."


@dataclass
class Verdict:
    """Final referee verdict with explanation."""
    recommended_security: str
    recommended_cost: str
    security_explanation: str
    cost_explanation: str
    rejected_options: List[Dict[str, str]]
    balance_summary: str
    confidence: str  # "high", "medium", "low"
    guardrail_violations: List[GuardrailViolation] = field(default_factory=list)


def _compute_weights(context: AlertContext) -> Dict[str, float]:
    """
    Dynamically compute weights based on context.
    Weights adapt to severity and cost spike levels.
    """
    severity_factor = context.severity / 10.0
    cost_spike_factor = min(context.cloud_cost_spike_percent / 200.0, 1.0)
    
    # High severity -> prioritize security risk and data loss
    # High cost spike -> prioritize cost impact
    # High SLA -> prioritize uptime
    sla_weight = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(context.sla_importance, 1.0)
    crit_weight = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.0}.get(
        context.business_criticality, 1.0
    )
    
    # Base weights
    w_security = 0.30
    w_downtime = 0.25
    w_cost = 0.25
    w_data_loss = 0.20
    
    # Adjust for severity (high severity = security matters more)
    if severity_factor > 0.7:
        w_security += 0.15
        w_data_loss += 0.10
        w_cost -= 0.15
        w_downtime -= 0.10
    elif severity_factor < 0.3:
        w_security -= 0.10
        w_cost += 0.10
    
    # Adjust for cost spike (high spike = cost matters more)
    if cost_spike_factor > 0.7:
        w_cost += 0.15
        w_security -= 0.10
        w_downtime -= 0.05
    
    # Adjust for SLA importance
    w_downtime *= sla_weight
    
    # Normalize weights to sum to 1
    total = w_security + w_downtime + w_cost + w_data_loss
    return {
        "security_risk": w_security / total,
        "business_downtime": w_downtime / total,
        "cost_impact": w_cost / total,
        "data_loss_probability": w_data_loss / total,
    }


def _normalize_scores(options: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Normalize scores to 0-1 range for fair comparison."""
    if not options:
        return {}
    
    # Find min/max for each metric
    metrics = ["security_risk", "business_downtime", "cost_impact", "data_loss_probability"]
    ranges = {}
    
    for metric in metrics:
        values = [opt["scores"][metric] for opt in options.values()]
        min_val, max_val = min(values), max(values)
        ranges[metric] = (min_val, max_val)
    
    normalized = {}
    for key, opt in options.items():
        normalized[key] = {}
        for metric in metrics:
            min_val, max_val = ranges[metric]
            val = opt["scores"][metric]
            if max_val == min_val:
                normalized[key][metric] = 0.5
            else:
                # For cost_impact, negative is good (savings)
                if metric == "cost_impact":
                    # Invert so lower (more negative) is better -> lower normalized score
                    normalized[key][metric] = (val - min_val) / (max_val - min_val)
                else:
                    normalized[key][metric] = (val - min_val) / (max_val - min_val)
        normalized[key]["name"] = opt["name"]
        normalized[key]["raw_scores"] = opt["scores"]
    
    return normalized


def _compute_balanced_score(
    normalized: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """Compute weighted balance score. Lower is better."""
    return sum(
        normalized.get(metric, 0) * weight
        for metric, weight in weights.items()
    )


def _explain_selection(
    selected_key: str,
    selected: Dict[str, Any],
    all_options: Dict[str, Any],
    weights: Dict[str, float],
    context: AlertContext,
    option_type: str
) -> str:
    """Generate plain English explanation for why an option was selected."""
    raw = selected["raw_scores"]
    
    # Identify strongest attributes
    strengths = []
    if raw["security_risk"] <= 50:
        strengths.append("acceptable security risk")
    if raw["business_downtime"] <= 30:
        strengths.append("minimal downtime impact")
    if raw["cost_impact"] <= 0:
        strengths.append(f"cost savings of ${abs(raw['cost_impact']):.0f}")
    elif raw["cost_impact"] <= 500:
        strengths.append("reasonable cost")
    if raw["data_loss_probability"] <= 20:
        strengths.append("low data loss risk")
    
    # Context-specific reasoning
    reasons = []
    if context.severity >= 7:
        reasons.append(f"Given the high severity ({context.severity}/10), security was prioritized")
    if context.cloud_cost_spike_percent >= 150:
        reasons.append(f"With a {context.cloud_cost_spike_percent}% cost spike, cost control was weighted heavily")
    if context.sla_importance == "high":
        reasons.append("High SLA importance made uptime a key factor")
    if context.business_criticality in ["high", "critical"]:
        reasons.append(f"The {context.business_criticality} business criticality influenced the balance")
    
    explanation = f"{selected['name']} was selected because it offers "
    if strengths:
        explanation += ", ".join(strengths[:3])
    else:
        explanation += "the best overall balance"
    explanation += ". "
    
    if reasons:
        explanation += reasons[0] + ". "
    
    return explanation


def _explain_rejections(
    selected_key: str,
    all_options: Dict[str, Any],
    normalized: Dict[str, Dict[str, float]],
    context: AlertContext
) -> List[Dict[str, str]]:
    """Explain why other options were not selected."""
    rejections = []
    
    for key, opt in all_options.items():
        if key == selected_key:
            continue
        
        raw = opt["scores"]
        norm = normalized[key]
        
        # Identify the main weakness
        weaknesses = []
        if raw["security_risk"] > 60:
            weaknesses.append(f"security risk too high ({raw['security_risk']:.0f})")
        if raw["business_downtime"] > 60:
            weaknesses.append(f"excessive downtime ({raw['business_downtime']:.0f} min)")
        if raw["cost_impact"] > 1000:
            weaknesses.append(f"cost too high (${raw['cost_impact']:.0f})")
        if raw["data_loss_probability"] > 40:
            weaknesses.append(f"data loss risk elevated ({raw['data_loss_probability']:.0f}%)")
        
        if not weaknesses:
            # No major weakness, just not the best balance
            weaknesses.append("did not offer the best overall balance for this context")
        
        rejections.append({
            "option": opt["name"],
            "reason": weaknesses[0],
        })
    
    return rejections


def generate_verdict(
    comparison: Dict[str, Any],
    context: AlertContext
) -> Verdict:
    """
    Generate a balanced verdict based on comparison results.
    Uses deterministic weighted scoring with guardrail validation.
    """
    weights = comparison["weights"]
    
    # Find best security option by weighted total (lower is better)
    sec_scores = {
        key: opt["weighted_scores"]["total"]
        for key, opt in comparison["security_options"].items()
    }
    best_sec_key = min(sec_scores, key=sec_scores.get)
    
    # Find best cost option by weighted total (lower is better)
    cost_scores = {
        key: opt["weighted_scores"]["total"]
        for key, opt in comparison["cost_options"].items()
    }
    best_cost_key = min(cost_scores, key=cost_scores.get)
    
    # Get initial recommendations
    initial_security = comparison["security_options"][best_sec_key]["name"]
    initial_cost = comparison["cost_options"][best_cost_key]["name"]
    
    # Apply guardrails to ensure safe recommendations
    safe_security, safe_cost, violations = check_guardrails(
        context,
        initial_security,
        initial_cost,
        comparison["security_options"],
        comparison["cost_options"],
    )
    
    # Update keys if guardrails changed recommendations
    if safe_security != initial_security:
        best_sec_key = _find_option_key(comparison["security_options"], safe_security)
    if safe_cost != initial_cost:
        best_cost_key = _find_option_key(comparison["cost_options"], safe_cost)
    
    # Generate explanations using normalized scores
    sec_opt = comparison["security_options"][best_sec_key]
    sec_explanation = _explain_selection_v2(sec_opt, weights, context)
    
    cost_opt = comparison["cost_options"][best_cost_key]
    cost_explanation = _explain_selection_v2(cost_opt, weights, context)
    
    # Add guardrail context to explanations if violations occurred
    if violations:
        guardrail_note = " [Guardrail applied - see safety overrides below]"
        for v in violations:
            if v.replacement_option == safe_security:
                sec_explanation += guardrail_note
            if v.replacement_option == safe_cost:
                cost_explanation += guardrail_note
    
    # Explain rejections
    sec_rejections = _explain_rejections_v2(
        best_sec_key, comparison["security_options"], context
    )
    cost_rejections = _explain_rejections_v2(
        best_cost_key, comparison["cost_options"], context
    )
    
    # Determine confidence based on score spread
    sec_spread = max(sec_scores.values()) - min(sec_scores.values())
    cost_spread = max(cost_scores.values()) - min(cost_scores.values())
    avg_spread = (sec_spread + cost_spread) / 2
    
    if avg_spread > 1.5:
        confidence = "high"
    elif avg_spread > 0.8:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Lower confidence if guardrails were triggered (override may not be optimal)
    if violations and confidence == "high":
        confidence = "medium"
    
    # Balance summary
    balance_summary = (
        f"Weights: Security {weights['security_risk']:.0%}, "
        f"Uptime {weights['business_downtime']:.0%}, "
        f"Cost {weights['cost_impact']:.0%}, "
        f"Data Loss {weights['data_loss_probability']:.0%}. "
        f"{comparison['weight_explanation']}"
    )
    
    return Verdict(
        recommended_security=safe_security,
        recommended_cost=safe_cost,
        security_explanation=sec_explanation,
        cost_explanation=cost_explanation,
        rejected_options=sec_rejections + cost_rejections,
        balance_summary=balance_summary,
        confidence=confidence,
        guardrail_violations=violations,  # Add violations to verdict
    )


def _explain_selection_v2(
    option: Dict[str, Any],
    weights: Dict[str, float],
    context: AlertContext
) -> str:
    """Generate explanation using normalized 0-10 scores."""
    norm = option["normalized_scores"]
    weighted = option["weighted_scores"]
    
    # Identify strengths (normalized score <= 3 is good)
    strengths = []
    if norm["security_risk"] <= 3:
        strengths.append(f"low security risk ({norm['security_risk']}/10)")
    if norm["business_downtime"] <= 3:
        strengths.append(f"minimal downtime ({norm['business_downtime']}/10)")
    if norm["cost_impact"] <= 3:
        strengths.append(f"favorable cost ({norm['cost_impact']}/10)")
    if norm["data_loss_probability"] <= 3:
        strengths.append(f"low data loss risk ({norm['data_loss_probability']}/10)")
    
    explanation = f"{option['name']} scored {weighted['total']:.2f} weighted total. "
    
    if strengths:
        explanation += f"Strengths: {', '.join(strengths[:2])}. "
    
    # Context reasoning
    if context.severity >= 7:
        explanation += f"High severity ({context.severity}/10) prioritized security. "
    elif context.business_criticality in ["high", "critical"]:
        explanation += f"{context.business_criticality.title()} criticality influenced selection. "
    
    return explanation


def _explain_rejections_v2(
    selected_key: str,
    all_options: Dict[str, Any],
    context: AlertContext
) -> List[Dict[str, str]]:
    """Explain rejections using normalized scores."""
    rejections = []
    
    for key, opt in all_options.items():
        if key == selected_key:
            continue
        
        norm = opt["normalized_scores"]
        weighted = opt["weighted_scores"]
        
        # Identify main weakness (normalized score >= 7 is bad)
        weaknesses = []
        if norm["security_risk"] >= 7:
            weaknesses.append(f"security risk {norm['security_risk']}/10")
        if norm["business_downtime"] >= 7:
            weaknesses.append(f"downtime {norm['business_downtime']}/10")
        if norm["cost_impact"] >= 7:
            weaknesses.append(f"cost {norm['cost_impact']}/10")
        if norm["data_loss_probability"] >= 7:
            weaknesses.append(f"data loss {norm['data_loss_probability']}/10")
        
        if weaknesses:
            reason = f"high {weaknesses[0]}"
        else:
            reason = f"weighted total {weighted['total']:.2f} vs lower alternatives"
        
        rejections.append({
            "option": opt["name"],
            "reason": reason,
        })
    
    return rejections


def format_verdict(verdict: Verdict) -> str:
    """Format verdict as readable output."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("REFEREE VERDICT")
    lines.append("=" * 80)
    
    lines.append(f"\nConfidence: {verdict.confidence.upper()}")
    lines.append(f"\n{verdict.balance_summary}")
    
    # Show guardrail violations if any
    if verdict.guardrail_violations:
        lines.append(format_guardrail_violations(verdict.guardrail_violations))
    
    lines.append("\n" + "-" * 40)
    lines.append("RECOMMENDED SECURITY RESPONSE")
    lines.append("-" * 40)
    lines.append(f">> {verdict.recommended_security}")
    lines.append(f"\n{verdict.security_explanation}")
    
    lines.append("\n" + "-" * 40)
    lines.append("RECOMMENDED COST ACTION")
    lines.append("-" * 40)
    lines.append(f">> {verdict.recommended_cost}")
    lines.append(f"\n{verdict.cost_explanation}")
    
    lines.append("\n" + "-" * 40)
    lines.append("WHY OTHER OPTIONS WERE NOT CHOSEN")
    lines.append("-" * 40)
    for rej in verdict.rejected_options:
        lines.append(f"- {rej['option']}: {rej['reason']}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


@dataclass
class SensitivityScenario:
    """A what-if scenario for sensitivity analysis."""
    name: str
    description: str
    new_security: str
    new_cost: str
    changed: bool
    explanation: str


def _get_verdict_key(context: AlertContext) -> Tuple[str, str]:
    """Get the recommended options for a context (lightweight)."""
    comparison = compare_options(context)
    
    # Find best by weighted total
    sec_scores = {
        key: opt["weighted_scores"]["total"]
        for key, opt in comparison["security_options"].items()
    }
    best_sec_key = min(sec_scores, key=sec_scores.get)
    
    cost_scores = {
        key: opt["weighted_scores"]["total"]
        for key, opt in comparison["cost_options"].items()
    }
    best_cost_key = min(cost_scores, key=cost_scores.get)
    
    return (
        comparison["security_options"][best_sec_key]["name"],
        comparison["cost_options"][best_cost_key]["name"],
    )


def run_sensitivity_analysis(context: AlertContext) -> List[SensitivityScenario]:
    """
    Analyze how verdict changes under different conditions.
    Lightweight rule-based approach.
    """
    scenarios = []
    current_sec, current_cost = _get_verdict_key(context)
    
    # Scenario 1: Severity increases by 2 points
    new_severity = min(10, context.severity + 2)
    if new_severity != context.severity:
        modified = AlertContext(
            alert_type=context.alert_type,
            severity=new_severity,
            asset_type=context.asset_type,
            business_criticality=context.business_criticality,
            cloud_cost_spike_percent=context.cloud_cost_spike_percent,
            traffic_pattern=context.traffic_pattern,
            sla_importance=context.sla_importance,
            time_context=context.time_context,
            incident_duration_minutes=context.incident_duration_minutes,
        )
        new_sec, new_cost = _get_verdict_key(modified)
        changed = (new_sec != current_sec) or (new_cost != current_cost)
        
        if changed:
            explanation = (
                f"Higher severity ({new_severity}/10) shifts weight toward security. "
                f"Security response changes to '{new_sec}' for stronger protection. "
                f"Cost action becomes '{new_cost}' as budget flexibility decreases."
            )
        else:
            explanation = (
                f"Even at severity {new_severity}/10, the current recommendations hold. "
                f"The selected options already account for escalation risk."
            )
        
        scenarios.append(SensitivityScenario(
            name="Severity +2",
            description=f"If severity increases from {context.severity} to {new_severity}",
            new_security=new_sec,
            new_cost=new_cost,
            changed=changed,
            explanation=explanation,
        ))
    
    # Scenario 2: Cloud cost spike doubles
    new_cost_spike = context.cloud_cost_spike_percent * 2
    modified = AlertContext(
        alert_type=context.alert_type,
        severity=context.severity,
        asset_type=context.asset_type,
        business_criticality=context.business_criticality,
        cloud_cost_spike_percent=new_cost_spike,
        traffic_pattern=context.traffic_pattern,
        sla_importance=context.sla_importance,
        time_context=context.time_context,
        incident_duration_minutes=context.incident_duration_minutes,
    )
    new_sec, new_cost = _get_verdict_key(modified)
    changed = (new_sec != current_sec) or (new_cost != current_cost)
    
    if changed:
        explanation = (
            f"Doubling the cost spike to {new_cost_spike:.0f}% makes cost control critical. "
            f"Cost action shifts to '{new_cost}' for aggressive savings. "
            f"Security response may adjust to '{new_sec}' to balance budget pressure."
        )
    else:
        explanation = (
            f"Even at {new_cost_spike:.0f}% cost spike, recommendations remain stable. "
            f"Current options already optimize for cost efficiency."
        )
    
    scenarios.append(SensitivityScenario(
        name="Cost Spike x2",
        description=f"If cost spike doubles from {context.cloud_cost_spike_percent:.0f}% to {new_cost_spike:.0f}%",
        new_security=new_sec,
        new_cost=new_cost,
        changed=changed,
        explanation=explanation,
    ))
    
    # Scenario 3: Traffic normalizes
    if context.traffic_pattern != "normal":
        modified = AlertContext(
            alert_type=context.alert_type,
            severity=context.severity,
            asset_type=context.asset_type,
            business_criticality=context.business_criticality,
            cloud_cost_spike_percent=context.cloud_cost_spike_percent,
            traffic_pattern="normal",
            sla_importance=context.sla_importance,
            time_context=context.time_context,
            incident_duration_minutes=context.incident_duration_minutes,
        )
        new_sec, new_cost = _get_verdict_key(modified)
        changed = (new_sec != current_sec) or (new_cost != current_cost)
        
        if changed:
            explanation = (
                f"Normal traffic reduces risk scores across all options. "
                f"Security response can relax to '{new_sec}' with lower threat level. "
                f"Cost action shifts to '{new_cost}' as urgency decreases."
            )
        else:
            explanation = (
                f"Traffic normalization doesn't change the verdict. "
                f"Other factors (severity, cost spike) remain the primary drivers."
            )
        
        scenarios.append(SensitivityScenario(
            name="Traffic Normalizes",
            description=f"If traffic pattern changes from '{context.traffic_pattern}' to 'normal'",
            new_security=new_sec,
            new_cost=new_cost,
            changed=changed,
            explanation=explanation,
        ))
    
    # Scenario 4: Incident duration exceeds 2 hours (simulate with higher criticality + business hours)
    escalated_criticality = {
        "low": "medium",
        "medium": "high",
        "high": "critical",
        "critical": "critical",
    }.get(context.business_criticality, "critical")
    
    modified = AlertContext(
        alert_type=context.alert_type,
        severity=min(10, context.severity + 1),  # Prolonged incidents tend to escalate
        asset_type=context.asset_type,
        business_criticality=escalated_criticality,
        cloud_cost_spike_percent=context.cloud_cost_spike_percent * 1.5,  # Costs accumulate
        traffic_pattern=context.traffic_pattern,
        sla_importance="high" if context.sla_importance != "high" else context.sla_importance,
        time_context="business_hours",  # Extended incidents hit business hours
        incident_duration_minutes=150,  # Prolonged phase
    )
    new_sec, new_cost = _get_verdict_key(modified)
    changed = (new_sec != current_sec) or (new_cost != current_cost)
    
    if changed:
        explanation = (
            f"Extended duration escalates criticality to '{escalated_criticality}' and accumulates costs. "
            f"Security response shifts to '{new_sec}' for decisive action. "
            f"Cost action becomes '{new_cost}' as prolonged incidents demand resolution over savings."
        )
    else:
        explanation = (
            f"Even with extended duration, current recommendations remain appropriate. "
            f"The selected options are robust for prolonged incidents."
        )
    
    scenarios.append(SensitivityScenario(
        name="Duration >2 Hours",
        description="If incident duration exceeds 2 hours (escalated criticality, accumulated costs)",
        new_security=new_sec,
        new_cost=new_cost,
        changed=changed,
        explanation=explanation,
    ))
    
    return scenarios


def format_sensitivity(scenarios: List[SensitivityScenario]) -> str:
    """Format sensitivity analysis as readable output."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("WHAT IF CONDITIONS CHANGE?")
    lines.append("=" * 80)
    
    for scenario in scenarios:
        lines.append(f"\n>> {scenario.name}")
        lines.append(f"   {scenario.description}")
        lines.append("-" * 60)
        
        if scenario.changed:
            lines.append(f"   VERDICT CHANGES:")
            lines.append(f"   - Security: {scenario.new_security}")
            lines.append(f"   - Cost: {scenario.new_cost}")
        else:
            lines.append(f"   VERDICT UNCHANGED")
            lines.append(f"   - Security: {scenario.new_security}")
            lines.append(f"   - Cost: {scenario.new_cost}")
        
        lines.append(f"\n   {scenario.explanation}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_performance(metrics: PerformanceMetrics, fast_path: bool = False) -> str:
    """Format performance metrics as readable output."""
    lines = []
    lines.append("\n" + "-" * 80)
    lines.append("PERFORMANCE METRICS")
    lines.append("-" * 80)
    
    if fast_path:
        lines.append(f"Fast-path execution: {metrics.total_ms:.2f}ms")
        lines.append(f"  Business impact generation: {metrics.business_impact_ms:.2f}ms")
    else:
        lines.append(f"Option evaluation:     {metrics.option_evaluation_ms:>8.2f}ms")
        lines.append(f"Verdict generation:    {metrics.verdict_generation_ms:>8.2f}ms")
        lines.append(f"Sensitivity analysis:  {metrics.sensitivity_analysis_ms:>8.2f}ms")
        lines.append(f"Timeline computation:  {metrics.timeline_computation_ms:>8.2f}ms")
        lines.append(f"Business impact:       {metrics.business_impact_ms:>8.2f}ms")
        lines.append("-" * 40)
        lines.append(f"Total execution time:  {metrics.total_ms:>8.2f}ms")
    
    lines.append("-" * 80)
    return "\n".join(lines)


def referee_analyze(context: AlertContext, force_full: bool = False) -> Dict[str, Any]:
    """
    Full analysis: comparison + verdict + sensitivity + timeline.
    Checks fast-path rules first unless force_full=True.
    Returns both structured data and formatted output with performance metrics.
    """
    total_start = time.perf_counter()
    metrics = PerformanceMetrics()
    
    # Check fast-path first (unless forced to full analysis)
    if not force_full:
        fast_path = check_fast_path(context)
        if fast_path:
            # Generate business impact even for fast-path
            t0 = time.perf_counter()
            business_impact = generate_business_impact(
                context, fast_path.security_action, fast_path.cost_action, None
            )
            metrics.business_impact_ms = (time.perf_counter() - t0) * 1000
            metrics.total_ms = (time.perf_counter() - total_start) * 1000
            
            return {
                "fast_path": True,
                "fast_path_decision": {
                    "rule_name": fast_path.rule_name,
                    "security_action": fast_path.security_action,
                    "cost_action": fast_path.cost_action,
                    "explanation": fast_path.explanation,
                    "conditions_met": fast_path.conditions_met,
                },
                "verdict": {
                    "recommended_security": fast_path.security_action,
                    "recommended_cost": fast_path.cost_action,
                    "security_explanation": fast_path.explanation,
                    "cost_explanation": "Cost optimization suspended during fast-path response.",
                    "rejected_options": [],
                    "balance_summary": f"Fast-path rule {fast_path.rule_name} triggered.",
                    "confidence": "high",
                },
                "business_impact": {
                    "headline": business_impact.headline,
                    "risk_avoided": business_impact.risk_avoided,
                    "revenue_impact": business_impact.revenue_impact,
                    "cost_tradeoff": business_impact.cost_tradeoff,
                    "recommendation_plain": business_impact.recommendation_plain,
                    "urgency": business_impact.urgency,
                },
                "comparison": None,
                "sensitivity": None,
                "timeline": None,
                "performance": {
                    "option_evaluation_ms": 0.0,
                    "verdict_generation_ms": 0.0,
                    "sensitivity_analysis_ms": 0.0,
                    "timeline_computation_ms": 0.0,
                    "business_impact_ms": metrics.business_impact_ms,
                    "total_ms": metrics.total_ms,
                    "summary": f"Fast-path: {metrics.total_ms:.2f}ms total",
                },
                "formatted_output": format_business_impact(business_impact) + format_fast_path(fast_path, context) + format_performance(metrics, fast_path=True),
            }
    
    # Full analysis with performance measurement
    
    # 1. Option evaluation
    t0 = time.perf_counter()
    comparison = compare_options(context)
    metrics.option_evaluation_ms = (time.perf_counter() - t0) * 1000
    
    # 2. Verdict generation
    t0 = time.perf_counter()
    verdict = generate_verdict(comparison, context)
    metrics.verdict_generation_ms = (time.perf_counter() - t0) * 1000
    
    # 3. Sensitivity analysis
    t0 = time.perf_counter()
    sensitivity = run_sensitivity_analysis(context)
    metrics.sensitivity_analysis_ms = (time.perf_counter() - t0) * 1000
    
    # 4. Timeline computation
    t0 = time.perf_counter()
    timeline = compute_timeline_recommendation(context)
    metrics.timeline_computation_ms = (time.perf_counter() - t0) * 1000
    
    # 5. Business impact
    t0 = time.perf_counter()
    business_impact = generate_business_impact(
        context, verdict.recommended_security, verdict.recommended_cost, comparison
    )
    metrics.business_impact_ms = (time.perf_counter() - t0) * 1000
    
    # Total time
    metrics.total_ms = (time.perf_counter() - total_start) * 1000
    
    return {
        "fast_path": False,
        "comparison": comparison,
        "verdict": {
            "recommended_security": verdict.recommended_security,
            "recommended_cost": verdict.recommended_cost,
            "security_explanation": verdict.security_explanation,
            "cost_explanation": verdict.cost_explanation,
            "rejected_options": verdict.rejected_options,
            "balance_summary": verdict.balance_summary,
            "confidence": verdict.confidence,
            "guardrail_violations": [
                {
                    "rule_name": v.rule_name,
                    "blocked_option": v.blocked_option,
                    "replacement_option": v.replacement_option,
                    "reason": v.reason,
                }
                for v in verdict.guardrail_violations
            ],
        },
        "timeline": {
            "current_phase": timeline.current_phase,
            "current_security": timeline.current_security,
            "current_cost": timeline.current_cost,
            "next_phase": timeline.next_phase,
            "next_security": timeline.next_security,
            "next_cost": timeline.next_cost,
            "escalation_trigger": timeline.escalation_trigger,
            "time_to_escalation": timeline.time_to_escalation,
        },
        "business_impact": {
            "headline": business_impact.headline,
            "risk_avoided": business_impact.risk_avoided,
            "revenue_impact": business_impact.revenue_impact,
            "cost_tradeoff": business_impact.cost_tradeoff,
            "recommendation_plain": business_impact.recommendation_plain,
            "urgency": business_impact.urgency,
        },
        "sensitivity": [
            {
                "name": s.name,
                "description": s.description,
                "new_security": s.new_security,
                "new_cost": s.new_cost,
                "changed": s.changed,
                "explanation": s.explanation,
            }
            for s in sensitivity
        ],
        "performance": {
            "option_evaluation_ms": metrics.option_evaluation_ms,
            "verdict_generation_ms": metrics.verdict_generation_ms,
            "sensitivity_analysis_ms": metrics.sensitivity_analysis_ms,
            "timeline_computation_ms": metrics.timeline_computation_ms,
            "business_impact_ms": metrics.business_impact_ms,
            "total_ms": metrics.total_ms,
            "summary": metrics.summary(),
        },
        "formatted_output": (
            format_business_impact(business_impact)
            + format_comparison(comparison)
            + format_verdict(verdict)
            + format_timeline(timeline)
            + format_sensitivity(sensitivity)
            + format_performance(metrics)
        ),
    }


@dataclass
class BusinessImpactSummary:
    """Executive summary for non-technical stakeholders."""
    headline: str
    risk_avoided: str
    revenue_impact: str
    cost_tradeoff: str
    recommendation_plain: str
    urgency: str  # "immediate", "soon", "monitor"


def generate_business_impact(
    context: AlertContext,
    verdict_security: str,
    verdict_cost: str,
    comparison: Dict[str, Any]
) -> BusinessImpactSummary:
    """
    Generate executive summary from existing analysis data.
    Avoids technical jargon, suitable for non-technical stakeholders.
    """
    # Determine urgency from severity and phase
    phase = get_incident_phase(context.incident_duration_minutes)
    if context.severity >= 8 or phase in ["prolonged", "critical"]:
        urgency = "immediate"
    elif context.severity >= 5 or phase == "developing":
        urgency = "soon"
    else:
        urgency = "monitor"
    
    # Generate headline based on alert type and severity
    alert_headlines = {
        "intrusion": "Security Breach Detected",
        "ddos": "Service Under Attack",
        "data_exfil": "Data Leak Risk Identified",
        "cost_anomaly": "Unusual Spending Detected",
        "malware": "Malicious Software Found",
    }
    base_headline = alert_headlines.get(context.alert_type, "System Alert")
    
    severity_qualifier = {
        (1, 3): "Minor",
        (4, 6): "Moderate", 
        (7, 8): "Serious",
        (9, 10): "Critical",
    }
    for (low, high), qualifier in severity_qualifier.items():
        if low <= context.severity <= high:
            headline = f"{qualifier} {base_headline}"
            break
    else:
        headline = base_headline
    
    # Risk avoided explanation (plain language)
    risk_explanations = {
        "Isolate Immediately": "prevents the problem from spreading to other systems",
        "Monitor and Alert": "keeps watch while minimizing disruption to operations",
        "Rate Limit Traffic": "slows down suspicious activity without stopping normal work",
        "Failover to Backup": "switches to backup systems to keep services running",
        "Block Source IPs": "stops the source of the problem from reaching our systems",
        "Scale Down Immediately": "reduces unnecessary spending right away",
        "Enable Spot Instances": "switches to lower-cost computing options",
        "Throttle Non-Critical": "prioritizes essential services over less important ones",
        "Maintain Current State": "keeps things stable while we assess the situation",
        "Schedule Review": "flags this for team review without immediate changes",
    }
    
    security_benefit = risk_explanations.get(verdict_security, "addresses the security concern")
    cost_benefit = risk_explanations.get(verdict_cost, "manages spending appropriately")
    
    risk_avoided = f"Taking action now {security_benefit}."
    if context.severity >= 7:
        risk_avoided += " Without action, this could affect customer access or data safety."
    elif context.severity >= 4:
        risk_avoided += " This reduces the chance of the issue growing larger."
    
    # Revenue/SLA impact (derived from context)
    if context.sla_importance == "high":
        if context.business_criticality in ["high", "critical"]:
            revenue_impact = "This affects customer-facing services. Downtime could impact revenue and customer trust."
        else:
            revenue_impact = "Service agreements require quick response. Delays may affect customer satisfaction."
    elif context.business_criticality in ["high", "critical"]:
        revenue_impact = "This involves important business systems. Extended issues could affect operations."
    else:
        revenue_impact = "Impact is limited to internal systems. Customer services are not directly affected."
    
    # Add duration context
    if phase == "critical":
        revenue_impact += f" Issue has been ongoing for {context.incident_duration_minutes // 60} hours - resolution is overdue."
    elif phase == "prolonged":
        revenue_impact += f" Issue has persisted for over an hour - prompt action recommended."
    
    # Cost trade-off (simple language)
    cost_spike = context.cloud_cost_spike_percent
    if cost_spike >= 200:
        cost_context = f"Spending is {cost_spike:.0f}% above normal."
    elif cost_spike >= 100:
        cost_context = f"Spending has doubled from usual levels."
    else:
        cost_context = "Spending is within acceptable range."
    
    # Get actual cost numbers from comparison if available
    if comparison:
        sec_key = _find_option_key(comparison["security_options"], verdict_security)
        cost_key = _find_option_key(comparison["cost_options"], verdict_cost)
        
        sec_cost = comparison["security_options"].get(sec_key, {}).get("scores", {}).get("cost_impact", 0)
        cost_cost = comparison["cost_options"].get(cost_key, {}).get("scores", {}).get("cost_impact", 0)
        
        if sec_cost > 0:
            cost_tradeoff = f"{cost_context} The security response costs approximately ${sec_cost:,.0f}."
        else:
            cost_tradeoff = f"{cost_context} The security response has minimal additional cost."
        
        if cost_cost < 0:
            cost_tradeoff += f" The cost action saves approximately ${abs(cost_cost):,.0f}."
        elif cost_cost == 0:
            cost_tradeoff += " Current spending will be maintained."
    else:
        cost_tradeoff = cost_context
    
    # Plain language recommendation
    urgency_phrases = {
        "immediate": "Act now",
        "soon": "Address within the hour",
        "monitor": "Keep watching",
    }
    
    recommendation_plain = (
        f"{urgency_phrases[urgency]}: {verdict_security} for security, "
        f"{verdict_cost} for cost management."
    )
    
    return BusinessImpactSummary(
        headline=headline,
        risk_avoided=risk_avoided,
        revenue_impact=revenue_impact,
        cost_tradeoff=cost_tradeoff,
        recommendation_plain=recommendation_plain,
        urgency=urgency,
    )


def _find_option_key(options: Dict[str, Any], name: str) -> str:
    """Find option key by name."""
    for key, opt in options.items():
        if opt.get("name") == name:
            return key
    return ""


def format_business_impact(summary: BusinessImpactSummary) -> str:
    """Format business impact summary for executives."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("EXECUTIVE SUMMARY")
    lines.append("=" * 80)
    
    # Urgency indicator
    urgency_display = {
        "immediate": "🔴 IMMEDIATE ACTION REQUIRED",
        "soon": "🟡 ACTION NEEDED SOON",
        "monitor": "🟢 MONITORING RECOMMENDED",
    }
    lines.append(f"\n{urgency_display.get(summary.urgency, 'ACTION NEEDED')}")
    lines.append(f"\n{summary.headline}")
    
    lines.append("\n" + "-" * 40)
    lines.append("What's at Risk")
    lines.append("-" * 40)
    lines.append(summary.risk_avoided)
    
    lines.append("\n" + "-" * 40)
    lines.append("Business Impact")
    lines.append("-" * 40)
    lines.append(summary.revenue_impact)
    
    lines.append("\n" + "-" * 40)
    lines.append("Cost Consideration")
    lines.append("-" * 40)
    lines.append(summary.cost_tradeoff)
    
    lines.append("\n" + "-" * 40)
    lines.append("Recommended Action")
    lines.append("-" * 40)
    lines.append(f">> {summary.recommendation_plain}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_timeline(timeline: TimelineRecommendation) -> str:
    """Format timeline recommendation as readable output."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("TIMELINE-AWARE ESCALATION")
    lines.append("=" * 80)
    
    lines.append(f"\nCurrent Phase: {timeline.current_phase.upper()}")
    lines.append("-" * 40)
    lines.append(f"Security: {timeline.current_security}")
    lines.append(f"Cost: {timeline.current_cost}")
    
    if timeline.time_to_escalation > 0:
        lines.append(f"\n>> NEXT ESCALATION in {timeline.time_to_escalation} minutes")
        lines.append(f"   Phase: {timeline.next_phase.upper()}")
        lines.append("-" * 40)
        lines.append(f"   Security: {timeline.next_security}")
        lines.append(f"   Cost: {timeline.next_cost}")
        lines.append(f"\n   Trigger: {timeline.escalation_trigger}")
    else:
        lines.append(f"\n>> {timeline.escalation_trigger}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_comparison(comparison: Dict[str, Any]) -> str:
    """Format comparison results with raw, normalized, and weighted scores."""
    lines = []
    lines.append("=" * 100)
    lines.append("IR-COST REFEREE - DETERMINISTIC SCORING MODEL")
    lines.append("=" * 100)
    
    # Context summary
    ctx = comparison["context"]
    lines.append(f"\nAlert: {ctx['alert_type']} | Severity: {ctx['severity']}/10 | Duration: {ctx['incident_duration_minutes']}min ({ctx['incident_phase']})")
    lines.append(f"Asset: {ctx['asset_type']} | Criticality: {ctx['business_criticality']}")
    lines.append(f"Cost Spike: {ctx['cloud_cost_spike_percent']}% | Traffic: {ctx['traffic_pattern']}")
    lines.append(f"SLA: {ctx['sla_importance']} | Time: {ctx['time_context']}")
    
    # Weights explanation
    w = comparison["weights"]
    lines.append(f"\nWEIGHTS: Security={w['security_risk']:.0%}, Uptime={w['business_downtime']:.0%}, "
                 f"Cost={w['cost_impact']:.0%}, DataLoss={w['data_loss_probability']:.0%}")
    lines.append(f"Reason: {comparison['weight_explanation']}")
    
    # Security options - Raw scores
    lines.append("\n" + "-" * 100)
    lines.append("SECURITY OPTIONS - RAW SCORES")
    lines.append("-" * 100)
    lines.append(f"{'Option':<25} {'SecRisk':>10} {'Downtime':>12} {'Cost':>12} {'DataLoss':>10}")
    lines.append("-" * 100)
    
    for key, opt in comparison["security_options"].items():
        s = opt["scores"]
        lines.append(
            f"{opt['name']:<25} {s['security_risk']:>10.1f} "
            f"{s['business_downtime']:>12.1f}min {s['cost_impact']:>11.0f}$ "
            f"{s['data_loss_probability']:>10.1f}%"
        )
    
    # Security options - Normalized (0-10)
    lines.append("\n" + "-" * 100)
    lines.append("SECURITY OPTIONS - NORMALIZED (0-10 scale, lower is better)")
    lines.append("-" * 100)
    lines.append(f"{'Option':<25} {'SecRisk':>8} {'Downtime':>10} {'Cost':>8} {'DataLoss':>10} {'Total':>8}")
    lines.append("-" * 100)
    
    for key, opt in comparison["security_options"].items():
        n = opt["normalized_scores"]
        lines.append(
            f"{opt['name']:<25} {n['security_risk']:>8.1f} "
            f"{n['business_downtime']:>10.1f} {n['cost_impact']:>8.1f} "
            f"{n['data_loss_probability']:>10.1f} {n['total']:>8.1f}"
        )
    
    # Security options - Weighted
    lines.append("\n" + "-" * 100)
    lines.append("SECURITY OPTIONS - WEIGHTED SCORES (lower is better)")
    lines.append("-" * 100)
    lines.append(f"{'Option':<25} {'SecRisk':>8} {'Downtime':>10} {'Cost':>8} {'DataLoss':>10} {'TOTAL':>8}")
    lines.append("-" * 100)
    
    for key, opt in comparison["security_options"].items():
        w = opt["weighted_scores"]
        lines.append(
            f"{opt['name']:<25} {w['security_risk']:>8.2f} "
            f"{w['business_downtime']:>10.2f} {w['cost_impact']:>8.2f} "
            f"{w['data_loss_probability']:>10.2f} {w['total']:>8.2f}"
        )
    
    # Cost options - Raw scores
    lines.append("\n" + "-" * 100)
    lines.append("COST OPTIONS - RAW SCORES")
    lines.append("-" * 100)
    lines.append(f"{'Option':<25} {'SecRisk':>10} {'Downtime':>12} {'Cost':>12} {'DataLoss':>10}")
    lines.append("-" * 100)
    
    for key, opt in comparison["cost_options"].items():
        s = opt["scores"]
        lines.append(
            f"{opt['name']:<25} {s['security_risk']:>10.1f} "
            f"{s['business_downtime']:>12.1f}min {s['cost_impact']:>11.0f}$ "
            f"{s['data_loss_probability']:>10.1f}%"
        )
    
    # Cost options - Normalized (0-10)
    lines.append("\n" + "-" * 100)
    lines.append("COST OPTIONS - NORMALIZED (0-10 scale, lower is better)")
    lines.append("-" * 100)
    lines.append(f"{'Option':<25} {'SecRisk':>8} {'Downtime':>10} {'Cost':>8} {'DataLoss':>10} {'Total':>8}")
    lines.append("-" * 100)
    
    for key, opt in comparison["cost_options"].items():
        n = opt["normalized_scores"]
        lines.append(
            f"{opt['name']:<25} {n['security_risk']:>8.1f} "
            f"{n['business_downtime']:>10.1f} {n['cost_impact']:>8.1f} "
            f"{n['data_loss_probability']:>10.1f} {n['total']:>8.1f}"
        )
    
    # Cost options - Weighted
    lines.append("\n" + "-" * 100)
    lines.append("COST OPTIONS - WEIGHTED SCORES (lower is better)")
    lines.append("-" * 100)
    lines.append(f"{'Option':<25} {'SecRisk':>8} {'Downtime':>10} {'Cost':>8} {'DataLoss':>10} {'TOTAL':>8}")
    lines.append("-" * 100)
    
    for key, opt in comparison["cost_options"].items():
        w = opt["weighted_scores"]
        lines.append(
            f"{opt['name']:<25} {w['security_risk']:>8.2f} "
            f"{w['business_downtime']:>10.2f} {w['cost_impact']:>8.2f} "
            f"{w['data_loss_probability']:>10.2f} {w['total']:>8.2f}"
        )
    
    lines.append("\n" + "=" * 100)
    return "\n".join(lines)


# =============================================================================
# BUILT-IN SCENARIO SIMULATOR
# =============================================================================

SIMULATOR_SCENARIOS = [
    {
        "name": "Scenario 1: High Severity Malware on Revenue-Critical Service",
        "description": (
            "Active malware detected on a production API gateway that handles payment processing. "
            "This is a revenue-critical service with strict SLA requirements. "
            "The incident has been developing for 35 minutes."
        ),
        "tradeoff_focus": "Security vs. Uptime - aggressive containment risks service disruption",
        "context": AlertContext(
            alert_type="malware",
            severity=9,
            asset_type="api_gateway",
            business_criticality="critical",
            cloud_cost_spike_percent=140.0,
            traffic_pattern="anomalous",
            sla_importance="high",
            time_context="business_hours",
            incident_duration_minutes=35,
        ),
    },
    {
        "name": "Scenario 2: Medium Severity Anomaly with Large Cost Spike",
        "description": (
            "Unusual compute activity detected with a 400% cost spike. "
            "Could be cryptomining or legitimate traffic surge. "
            "Medium business criticality, but costs are accumulating rapidly."
        ),
        "tradeoff_focus": "Cost vs. Risk - aggressive cost cuts may impact legitimate workloads",
        "context": AlertContext(
            alert_type="cost_anomaly",
            severity=5,
            asset_type="compute",
            business_criticality="medium",
            cloud_cost_spike_percent=400.0,
            traffic_pattern="sustained_high",
            sla_importance="medium",
            time_context="off_hours",
            incident_duration_minutes=90,
        ),
    },
    {
        "name": "Scenario 3: Low Severity Phishing with Normal Cost",
        "description": (
            "Phishing attempt detected targeting internal users. "
            "No systems compromised yet, costs are normal. "
            "Low immediate risk but requires monitoring."
        ),
        "tradeoff_focus": "Monitoring vs. Action - overreaction wastes resources, underreaction risks escalation",
        "context": AlertContext(
            alert_type="intrusion",
            severity=3,
            asset_type="compute",
            business_criticality="low",
            cloud_cost_spike_percent=105.0,
            traffic_pattern="normal",
            sla_importance="low",
            time_context="business_hours",
            incident_duration_minutes=15,
        ),
    },
    {
        "name": "Scenario 4: PII Database Under Attack (Guardrail Demo)",
        "description": (
            "Data exfiltration detected on customer database containing PII. "
            "Medium severity but high data protection requirements. "
            "Demonstrates PII_DATA_PROTECTION_PRIORITY guardrail."
        ),
        "tradeoff_focus": "Data Protection vs. Cost - guardrails enforce PII protection priority",
        "context": AlertContext(
            alert_type="data_exfil",
            severity=6,
            asset_type="database",
            business_criticality="medium",
            cloud_cost_spike_percent=180.0,
            traffic_pattern="anomalous",
            sla_importance="medium",
            time_context="off_hours",
            incident_duration_minutes=25,
        ),
    },
]


def run_scenario_simulator():
    """
    Run all predefined scenarios automatically.
    Demonstrates different trade-offs without external input.
    """
    print("\n" + "=" * 100)
    print("  IR-COST REFEREE - SCENARIO SIMULATOR")
    print("  Demonstrating Trade-Off Analysis Across Different Incident Types")
    print("=" * 100)
    
    for i, scenario in enumerate(SIMULATOR_SCENARIOS, 1):
        print("\n" + "#" * 100)
        print(f"  {scenario['name']}")
        print("#" * 100)
        
        print(f"\n{scenario['description']}")
        print(f"\n>> Trade-off Focus: {scenario['tradeoff_focus']}")
        
        # Run analysis
        result = referee_analyze(scenario["context"], force_full=True)
        
        # Print executive summary and verdict only (skip detailed scores for brevity)
        print(format_business_impact(result["business_impact"] if isinstance(result["business_impact"], BusinessImpactSummary) 
              else BusinessImpactSummary(**result["business_impact"])))
        
        # Print condensed verdict
        v = result["verdict"]
        print("\n" + "-" * 60)
        print("VERDICT SUMMARY")
        print("-" * 60)
        print(f"Security: {v['recommended_security']}")
        print(f"Cost: {v['recommended_cost']}")
        print(f"Confidence: {v['confidence'].upper()}")
        print(f"\n{v['balance_summary']}")
        
        # Print key trade-off insight
        print("\n" + "-" * 60)
        print("KEY TRADE-OFF INSIGHT")
        print("-" * 60)
        _print_tradeoff_insight(scenario["context"], v)
        
        print("\n" + "=" * 100)
    
    # Summary comparison
    print("\n" + "#" * 100)
    print("  SCENARIO COMPARISON SUMMARY")
    print("#" * 100)
    print("\n{:<50} {:<25} {:<25}".format("Scenario", "Security Action", "Cost Action"))
    print("-" * 100)
    
    for scenario in SIMULATOR_SCENARIOS:
        result = referee_analyze(scenario["context"], force_full=True)
        v = result["verdict"]
        short_name = scenario["name"].split(":")[1].strip()[:45]
        print(f"{short_name:<50} {v['recommended_security']:<25} {v['recommended_cost']:<25}")
    
    print("\n" + "=" * 100)
    print("  Simulator complete. Each scenario demonstrates different weight distributions")
    print("  based on severity, cost spike, business criticality, and SLA importance.")
    print("=" * 100 + "\n")


def _print_tradeoff_insight(context: AlertContext, verdict: Dict[str, Any]) -> None:
    """Print a human-readable trade-off insight for the scenario."""
    if context.severity >= 8:
        print("HIGH SEVERITY prioritizes security over cost.")
        print(f"- Security weight increased due to severity {context.severity}/10")
        print(f"- Cost optimization is secondary until threat is contained")
        if verdict["recommended_security"] in ["Isolate Immediately", "Failover to Backup"]:
            print(f"- Aggressive action ({verdict['recommended_security']}) accepted despite potential downtime")
    
    elif context.cloud_cost_spike_percent >= 300:
        print("LARGE COST SPIKE prioritizes cost control.")
        print(f"- Cost spike of {context.cloud_cost_spike_percent:.0f}% demands attention")
        print(f"- Security measures balanced against budget impact")
        if verdict["recommended_cost"] in ["Scale Down Immediately", "Enable Spot Instances"]:
            print(f"- Cost action ({verdict['recommended_cost']}) prioritized to stop bleeding")
    
    elif context.severity <= 4 and context.cloud_cost_spike_percent <= 150:
        print("LOW RISK scenario favors monitoring over action.")
        print(f"- Severity {context.severity}/10 doesn't warrant aggressive response")
        print(f"- Cost spike {context.cloud_cost_spike_percent:.0f}% is within tolerance")
        print(f"- Overreaction would waste resources and cause unnecessary disruption")
    
    else:
        print("BALANCED scenario requires careful trade-off analysis.")
        print(f"- No single factor dominates the decision")
        print(f"- Weights distributed based on multiple context factors")


# =============================================================================
# CLI DEMO
# =============================================================================

DEMO_SCENARIOS = {
    "intrusion": AlertContext(
        alert_type="intrusion",
        severity=7,
        asset_type="database",
        business_criticality="high",
        cloud_cost_spike_percent=180.0,
        traffic_pattern="anomalous",
        sla_importance="high",
        time_context="business_hours",
        incident_duration_minutes=45,  # Developing phase
    ),
    "cost_anomaly": AlertContext(
        alert_type="cost_anomaly",
        severity=3,
        asset_type="compute",
        business_criticality="medium",
        cloud_cost_spike_percent=300.0,
        traffic_pattern="sustained_high",
        sla_importance="low",
        time_context="off_hours",
        incident_duration_minutes=10,  # Early phase
    ),
    "ddos": AlertContext(
        alert_type="ddos",
        severity=9,
        asset_type="api_gateway",
        business_criticality="critical",
        cloud_cost_spike_percent=250.0,
        traffic_pattern="anomalous",
        sla_importance="high",
        time_context="business_hours",
        incident_duration_minutes=90,  # Prolonged phase
    ),
    "data_exfil": AlertContext(
        alert_type="data_exfil",
        severity=10,
        asset_type="database",
        business_criticality="critical",
        cloud_cost_spike_percent=50.0,
        traffic_pattern="anomalous",
        sla_importance="high",
        time_context="off_hours",
        incident_duration_minutes=5,  # Early phase - just detected
    ),
    "malware": AlertContext(
        alert_type="malware",
        severity=9,
        asset_type="compute",
        business_criticality="critical",
        cloud_cost_spike_percent=120.0,
        traffic_pattern="anomalous",
        sla_importance="high",
        time_context="business_hours",
        incident_duration_minutes=30,  # Developing phase
    ),
    "critical_db": AlertContext(
        alert_type="intrusion",
        severity=10,
        asset_type="database",
        business_criticality="critical",
        cloud_cost_spike_percent=200.0,
        traffic_pattern="anomalous",
        sla_importance="high",
        time_context="business_hours",
        incident_duration_minutes=150,  # Prolonged phase - urgent
    ),
    "prolonged": AlertContext(
        alert_type="intrusion",
        severity=6,
        asset_type="api_gateway",
        business_criticality="high",
        cloud_cost_spike_percent=220.0,
        traffic_pattern="sustained_high",
        sla_importance="high",
        time_context="business_hours",
        incident_duration_minutes=250,  # Critical duration - over 4 hours
    ),
    # Guardrail test scenarios
    "guardrail_pii": AlertContext(
        alert_type="data_exfil",
        severity=5,
        asset_type="database",
        business_criticality="medium",
        cloud_cost_spike_percent=200.0,
        traffic_pattern="anomalous",
        sla_importance="medium",
        time_context="off_hours",
        incident_duration_minutes=20,
    ),
    "guardrail_malware": AlertContext(
        alert_type="malware",
        severity=4,
        asset_type="compute",
        business_criticality="low",
        cloud_cost_spike_percent=110.0,
        traffic_pattern="normal",
        sla_importance="low",
        time_context="off_hours",
        incident_duration_minutes=10,
    ),
}


def print_banner():
    """Print CLI banner."""
    print("\n" + "=" * 80)
    print("  IR-COST REFEREE - Decision Support System")
    print("  Week 6: The Referee")
    print("=" * 80)


def print_menu():
    """Print scenario menu."""
    print("\nAvailable demo scenarios:")
    print("-" * 60)
    for key, ctx in DEMO_SCENARIOS.items():
        fast_path = check_fast_path(ctx)
        phase = get_incident_phase(ctx.incident_duration_minutes)
        marker = " [FAST-PATH]" if fast_path else ""
        print(f"  {key:<15} - {ctx.alert_type} (sev {ctx.severity}/10, {ctx.incident_duration_minutes}min/{phase}){marker}")
    print(f"  {'cloud':<15} - Generate from mock cloud cost data")
    print(f"  {'simulate':<15} - Run built-in scenario simulator")
    print(f"  {'custom':<15} - Enter your own values")
    print(f"  {'quit':<15} - Exit")
    print("-" * 60)


def get_custom_context() -> AlertContext:
    """Prompt user for custom context values."""
    print("\nEnter custom alert context:")
    print("(Press Enter to use default values shown in brackets)\n")
    
    alert_type = input("  Alert type [intrusion]: ").strip() or "intrusion"
    
    severity_str = input("  Severity 1-10 [5]: ").strip() or "5"
    severity = max(1, min(10, int(severity_str)))
    
    asset_type = input("  Asset type [database/api_gateway/compute/storage]: ").strip() or "database"
    
    criticality = input("  Business criticality [low/medium/high/critical]: ").strip() or "medium"
    
    cost_spike_str = input("  Cloud cost spike % [150]: ").strip() or "150"
    cost_spike = float(cost_spike_str)
    
    traffic = input("  Traffic pattern [normal/spike/sustained_high/anomalous]: ").strip() or "normal"
    
    sla = input("  SLA importance [low/medium/high]: ").strip() or "medium"
    
    time_ctx = input("  Time context [business_hours/off_hours/weekend/maintenance]: ").strip() or "business_hours"
    
    duration_str = input("  Incident duration in minutes [30]: ").strip() or "30"
    duration = max(0, int(duration_str))
    
    return AlertContext(
        alert_type=alert_type,
        severity=severity,
        asset_type=asset_type,
        business_criticality=criticality,
        cloud_cost_spike_percent=cost_spike,
        traffic_pattern=traffic,
        sla_importance=sla,
        time_context=time_ctx,
        incident_duration_minutes=duration,
    )


def run_demo():
    """Run interactive CLI demo."""
    print_banner()
    
    while True:
        print_menu()
        choice = input("\nSelect scenario: ").strip().lower()
        
        if choice == "quit" or choice == "q":
            print("\nExiting IR-Cost Referee. Stay secure!\n")
            break
        
        if choice == "simulate":
            run_scenario_simulator()
            input("\nPress Enter to continue...")
            continue
        
        if choice == "cloud":
            # Generate from mock cloud data
            print("\nCloud Cost Scenario Types:")
            print("  1. normal  - Typical usage")
            print("  2. spike   - Sudden cost increase")
            print("  3. gradual - Slowly increasing costs")
            print("  4. anomaly - Unusual pattern (cryptomining, DDoS, etc.)")
            
            scenario_type = input("\nSelect type [anomaly]: ").strip().lower() or "anomaly"
            asset = input("Asset type [compute/database/api_gateway/storage]: ").strip().lower() or "compute"
            
            # Generate cloud data
            cloud_data = generate_mock_cloud_costs(scenario_type, asset)
            print(format_cloud_costs(cloud_data))
            
            # Create alert context from cloud data
            context = create_alert_from_cloud_data(cloud_data, asset)
            print(f"\n>>> Generated alert from cloud data <<<")
            
        elif choice == "custom":
            context = get_custom_context()
        elif choice in DEMO_SCENARIOS:
            context = DEMO_SCENARIOS[choice]
        else:
            print(f"\nUnknown option: '{choice}'. Try again.")
            continue
        
        print(f"\n>>> Running analysis for: {context.alert_type} <<<")
        result = referee_analyze(context)
        print(result["formatted_output"])
        
        # If fast-path was triggered, offer full analysis
        if result.get("fast_path"):
            full = input("\nRun full analysis anyway? (y/n): ").strip().lower()
            if full == "y":
                print("\n>>> Running FULL analysis (fast-path bypassed) <<<")
                result = referee_analyze(context, force_full=True)
                print(result["formatted_output"])
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    force_full = "--force-full" in sys.argv or "-f" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    
    if len(args) > 0:
        scenario = args[0].lower()
        if scenario == "simulate":
            run_scenario_simulator()
        elif scenario in DEMO_SCENARIOS:
            print_banner()
            print(f"\n>>> Running: {scenario} <<<")
            if force_full:
                print("(--force-full: bypassing fast-path)")
            result = referee_analyze(DEMO_SCENARIOS[scenario], force_full=force_full)
            print(result["formatted_output"])
        elif scenario == "help":
            print("\nUsage: python referee_engine.py [scenario] [--force-full]")
            print("\nScenarios:")
            print("  intrusion      - Database intrusion (severity 7)")
            print("  cost_anomaly   - Cost spike anomaly (severity 3)")
            print("  ddos           - DDoS attack (severity 9)")
            print("  data_exfil     - Data exfiltration (severity 10)")
            print("  malware        - Malware detection (severity 9)")
            print("  critical_db    - Critical database intrusion (severity 10)")
            print("  prolonged      - Long-running incident (250 min)")
            print("  guardrail_pii  - PII protection guardrail demo")
            print("  guardrail_malware - Malware guardrail demo")
            print("  simulate       - Run built-in scenario simulator")
            print("\nOptions:")
            print("  --force-full, -f  Bypass fast-path and run full analysis")
            print("\nRun without arguments for interactive mode.")
        else:
            print(f"Unknown scenario: {scenario}")
            print("Available: intrusion, cost_anomaly, ddos, data_exfil, malware, critical_db,")
            print("           prolonged, guardrail_pii, guardrail_malware, simulate")
    elif "--help" in sys.argv or "-h" in sys.argv:
        print("\nUsage: python referee_engine.py [scenario] [--force-full]")
        print("\nScenarios: intrusion, cost_anomaly, ddos, data_exfil, malware, critical_db, prolonged")
        print("           simulate - Run built-in scenario simulator (3 predefined scenarios)")
        print("\nOptions:")
        print("  --force-full, -f  Bypass fast-path and run full analysis")
        print("\nRun without arguments for interactive mode.")
    else:
        # Interactive mode
        run_demo()
