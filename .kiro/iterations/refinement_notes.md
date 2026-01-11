# Refinement Notes & Iteration Log

## Purpose

This document tracks refinements to the IR-Cost Referee system as it's tested and improved through Week 6 analysis cycles.

---

## Iteration 1: Initial Framework

**Focus Area**: Core comparison logic

**Changes Made**:
- Established base scoring for security and cost options
- Defined input parameters (severity, cost spike, traffic pattern, etc.)
- Created option dictionaries with base metrics

**Rationale**: Needed a foundation that could evaluate multiple options without hardcoding preferences.

**Lessons Learned**: Base scores alone aren't enough—context multipliers are essential for realistic comparisons.

---

## Iteration 2: Dynamic Weighting

**Focus Area**: Context-aware weight calculation

**Changes Made**:
- Added `_compute_weights()` function that adapts to severity and cost spike
- High severity shifts weight toward security risk and data loss
- High cost spike shifts weight toward cost impact
- SLA importance affects downtime weighting

**Rationale**: A static weighting system would always favor the same options. Real decisions depend on context.

**Impact**: Verdicts now change based on input conditions, not just option scores.

---

## Iteration 3: Explanation Layer

**Focus Area**: Plain English reasoning

**Changes Made**:
- Added `_explain_selection()` for why an option was chosen
- Added `_explain_rejections()` for why others were not
- Explanations reference actual scores and context values

**Rationale**: A verdict without explanation isn't useful. Decision-makers need to understand the reasoning.

**Lessons Learned**: Explanations should cite specific numbers, not just general statements.

---

## Iteration 4: Sensitivity Analysis

**Focus Area**: What-if scenarios

**Changes Made**:
- Added four sensitivity scenarios (severity +2, cost spike x2, traffic normalizes, duration >2h)
- Each scenario re-runs the verdict logic with modified inputs
- Explanations describe whether and why the verdict changes

**Rationale**: Decisions aren't static. Showing how recommendations shift under different conditions builds trust.

**Impact**: Users can now anticipate how escalation or de-escalation affects the recommended response.

---

## Iteration 5: CLI Demo

**Focus Area**: Usability for demonstration

**Changes Made**:
- Added interactive menu with preset scenarios
- Added command-line argument support for direct scenario execution
- Added custom input mode for ad-hoc testing

**Rationale**: The engine needed a simple way to demonstrate its capabilities without requiring code changes.

---

## Known Limitations

- Analysis assumes stable market conditions
- Vendor pricing may change during evaluation period
- Technical requirements may evolve mid-project
- Sensitivity analysis covers four scenarios; more could be added

## Future Improvements

- [ ] Add more granular asset types
- [ ] Integrate real-time cost data feeds
- [ ] Add scenario modeling for changing requirements
- [ ] Develop automated constraint validation
- [ ] Add export to JSON/CSV for reporting

---

## Kiro's Role in Development

Kiro accelerated this project by:

1. **Scaffolding**: Generated complete folder structure and file templates in one pass
2. **Prompt Architecture**: Defined three-stage prompt system (identity → engine → verdict)
3. **Iterative Refinement**: Each feature was added incrementally with validation
4. **Code Quality**: Diagnostics checked after each change to catch issues early
5. **Documentation**: README and iteration notes created alongside code

The structured approach—prompts first, then implementation, then refinement—kept the project focused on neutrality and trade-off clarity rather than feature creep.

---

## Version History

| Version | Date | Key Changes | Status |
|---------|------|-------------|--------|
| 1.0 | Week 6 | Initial framework with comparison logic | Complete |
| 1.1 | Week 6 | Added verdict layer with explanations | Complete |
| 1.2 | Week 6 | Added sensitivity analysis | Complete |
| 1.3 | Week 6 | Added CLI demo | Complete |
