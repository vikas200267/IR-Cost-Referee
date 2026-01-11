# Stage 3: Final Verdict & Recommendation Framework

## Verdict Structure

The final verdict does NOT pick a winner. Instead, it provides decision-makers with the information needed to choose.

### Section 1: Executive Summary

```
Analysis Date: [date]
Options Evaluated: [count]
Constraints Identified: [count]
Key Finding: [one sentence neutral summary]
```

### Section 2: Constraint Satisfaction Matrix

Present as a table showing which options satisfy which constraints:

```
| Constraint | Option A | Option B | Option C | Critical? |
|------------|----------|----------|----------|-----------|
| Budget     | ✓        | ✓        | ✗        | Yes       |
| Timeline   | ✓        | ✗        | ✓        | Yes       |
| Performance| ✗        | ✓        | ✓        | No        |
```

### Section 3: Detailed Trade-off Analysis

For each pair of options, explicitly state what's gained and lost:

```
Option A vs Option B:

Choosing A over B means:
- GAIN: [benefit with metric]
- GAIN: [benefit with metric]
- LOSE: [cost with metric]
- LOSE: [cost with metric]
- NET IMPACT: [neutral assessment]

Choosing B over A means:
- GAIN: [benefit with metric]
- GAIN: [benefit with metric]
- LOSE: [cost with metric]
- LOSE: [cost with metric]
- NET IMPACT: [neutral assessment]
```

### Section 4: Risk Assessment

Present risks without implying which is "better":

```
Option A Risks:
- [Risk type]: [description] - Mitigation: [approach]
- [Risk type]: [description] - Mitigation: [approach]

Option B Risks:
- [Risk type]: [description] - Mitigation: [approach]
- [Risk type]: [description] - Mitigation: [approach]
```

### Section 5: Decision Scenarios

Help decision-makers by showing which option fits different priorities:

```
IF your priority is cost minimization:
→ Option A saves $[amount] annually
→ Trade-off: [specific limitation]
→ Constraint impact: [assessment]

IF your priority is performance:
→ Option B achieves [metric]
→ Trade-off: [specific limitation]
→ Constraint impact: [assessment]

IF your priority is risk reduction:
→ Option C reduces [risk]
→ Trade-off: [specific limitation]
→ Constraint impact: [assessment]
```

### Section 6: Missing Information

Identify what would improve the decision:

```
To make a more confident decision, you would need:
- [Information type]: [why it matters]
- [Information type]: [why it matters]
- [Information type]: [why it matters]
```

### Section 7: Implementation Considerations

For each option, outline what success looks like:

```
Option A Implementation:
- Timeline: [duration]
- Key milestones: [list]
- Success metrics: [list]
- Rollback plan: [approach]

Option B Implementation:
- Timeline: [duration]
- Key milestones: [list]
- Success metrics: [list]
- Rollback plan: [approach]
```

## Verdict Language Guidelines

### DO:
- "Option A provides [benefit] at the cost of [trade-off]"
- "If [priority] matters most, Option B aligns better because [reason]"
- "The key difference is [trade-off], which affects [dimension]"
- "This decision depends on whether [assumption] holds true"

### DON'T:
- "Option A is better because..."
- "You should choose Option B"
- "Option C is the obvious choice"
- "Option A is clearly superior"

## Final Checklist

- [ ] All options presented with equal detail and depth
- [ ] Trade-offs explicitly quantified where possible
- [ ] Constraints clearly mapped to each option
- [ ] Risks identified for all options
- [ ] Decision scenarios provided for different priorities
- [ ] Missing information flagged
- [ ] No language implying a preferred option
- [ ] Implementation paths clear for all options
- [ ] Reasoning is transparent and auditable
