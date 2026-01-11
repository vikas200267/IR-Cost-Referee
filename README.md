# IR-Cost Referee

A neutral decision-making system for evaluating infrastructure and cost-related options. Built for Week 6 analysis, this tool compares alternatives by examining trade-offs, constraints, and reasoning—without recommending a single "best" answer.

## What It Does

IR-Cost Referee helps teams make informed decisions by:

- **Comparing multiple options** fairly and transparently
- **Mapping constraints** (budget, timeline, technical, compliance)
- **Quantifying trade-offs** so decision-makers understand what's gained and lost
- **Showing reasoning** at every step for auditability
- **Avoiding bias** by presenting all perspectives equally

## The Problem

Traditional incident response and cost management tools give single answers: "Do this." But real decisions involve trade-offs:

- Isolating a threat immediately is secure but causes downtime
- Monitoring without action is cheap but risky
- Scaling down saves money but may impact performance

IR-Cost Referee doesn't hide these trade-offs. It surfaces them.

## How Kiro Accelerated Development

This project was built using Kiro's structured approach:

1. **Rapid Scaffolding** - Complete folder structure and file templates generated in one pass
2. **Prompt-Driven Architecture** - Three-stage prompt system (identity → engine → verdict) defined before implementation
3. **Iterative Refinement** - Each feature added incrementally with validation after each change
4. **Code Quality** - Diagnostics checked after every modification to catch issues early

The `.kiro/` directory contains the prompts and iteration notes that guided development.

## Project Structure

```
IR-Cost Referee/
├── .kiro/
│   ├── prompts/
│   │   ├── stage1_referee_identity.md      # Neutrality principles
│   │   ├── stage2_decision_engine.md       # Analysis framework
│   │   └── stage3_final_verdict.md         # Verdict structure
│   ├── iterations/
│   │   └── refinement_notes.md             # Development log
│   └── screenshots/
├── src/
│   └── referee_engine.py                   # Core implementation
└── README.md
```

## How to Run

**Interactive mode:**
```bash
python src/referee_engine.py
```

**Run specific scenario:**
```bash
python src/referee_engine.py intrusion
python src/referee_engine.py ddos
python src/referee_engine.py cost_anomaly
python src/referee_engine.py data_exfil
```

**Show help:**
```bash
python src/referee_engine.py --help
```

## Output Includes

1. **Comparison Table** - All options with scores for security risk, downtime, cost, and data loss
2. **Referee Verdict** - Recommended options with plain English explanations
3. **Rejection Reasons** - Why other options were not selected
4. **Sensitivity Analysis** - How the verdict changes if conditions shift

## Key Principles

- **Neutrality First** - All options presented with equal detail
- **Constraint Awareness** - Explicit mapping of what each option satisfies/violates
- **Trade-off Transparency** - Clear naming of what's gained and lost
- **Reasoning Visibility** - Every conclusion is auditable
- **No Single Answer** - The tool informs decisions; it doesn't make them

## Example

```
Alert: intrusion | Severity: 7/10
Asset: database | Criticality: high
Cost Spike: 180% | Traffic: anomalous

RECOMMENDED SECURITY RESPONSE: Block Source IPs
- Offers minimal downtime impact, reasonable cost
- Given high severity, security was prioritized

RECOMMENDED COST ACTION: Maintain Current State
- Offers acceptable security risk, minimal downtime
- High severity means budget flexibility decreases

WHY OTHER OPTIONS WERE NOT CHOSEN:
- Isolate Immediately: excessive downtime (122 min)
- Monitor and Alert: security risk too high (86)
- Failover to Backup: cost too high ($5400)
```

---

**Built with Kiro** | Week 6: The Referee
