# Reliability Constraint - Visual Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         VNE-Sim Simulator                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────┐          ┌──────────────────┐               │
│  │  parameters.   │          │  Configuration   │               │
│  │   json         │◄────────►│   (NEW: 3 params)│               │
│  └────────────────┘          └──────────────────┘               │
│         │                                                         │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Substrate Network (SN) Creation             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │   │
│  │  │ Snodes  │  │ Sedges   │  │ Reliability (NEW)  │   │   │
│  │  │ - cpu   │  │ - bw     │  │ - Range: [0.85-99] │   │   │
│  │  │ - index │  │ - latency│  │ - Random assign    │   │   │
│  │  └──────────┘  └──────────┘  └────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                         │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │      VNR Generator & VNRs Creation                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │   │
│  │  │ VNFs    │  │ Vedges   │  │ Reliability (NEW)  │   │   │
│  │  │ - cpu   │  │ - bw     │  │ - Range: [0.90-99] │   │   │
│  │  │ - index │  │ - latency│  │ - Random assign    │   │   │
│  │  └──────────┘  └──────────┘  └────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                         │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           VNR Placement (By Solvers)                   │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │ Node Mapping (VNF → Snode)                       │  │   │
│  │  │ Edge Mapping (Vedge → Path in SN)                │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                         │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │        Reward Calculation (NEW: Includes Reliability)   │   │
│  │                                                           │   │
│  │  1. Calculate R2C Ratio                                 │   │
│  │  2. Calculate Load Balance Factor                       │   │
│  │  3. Calculate Placement Reliability (NEW)               │   │
│  │     = ∏(VNF) × ∏(Node) × ∏(Edge)                       │   │
│  │  4. Combine with weights:                              │   │
│  │     reward = (1-w)×[R2C+balance] + w×reliability       │   │
│  │                                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Solver Learning & Optimization               │   │
│  │  - GNNDRL  - GNNDQL  - FirstFit  - GRASP (all updated)│   │
│  │  - Learn to maximize final reward                      │   │
│  │  - Balance cost, load, and reliability                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Three-Objective Optimization

```
┌─────────────────────────────────────┐
│   Solver Optimization Objective     │
│                                     │
│  Maximize:                          │
│  ┌───────────────────────────────┐  │
│  │ 1. R2C Ratio (Cost Efficiency)│  │ 85% weight
│  │    (higher is better)          │  │ (if w=0.15)
│  ├───────────────────────────────┤  │
│  │ 2. Load Balance Factor         │  │
│  │    (even distribution)         │  │
│  ├───────────────────────────────┤  │
│  │ 3. Reliability (NEW)           │  │ 15% weight
│  │    (high-reliability paths)    │  │ (if w=0.15)
│  └───────────────────────────────┘  │
│                                     │
│  Final Reward = Weighted Sum        │
│  of these three metrics             │
└─────────────────────────────────────┘

            Weight Control
    ┌─────────────────────────────┐
    │ vnr_reliability_weight = 0.15│ ◄─── Configurable!
    │                              │
    │ 0.0  → Pure cost+balance     │
    │ 0.15 → Balanced (DEFAULT)    │
    │ 0.5  → High reliability      │
    │ 1.0  → Pure reliability      │
    └─────────────────────────────┘
```

## Data Flow Diagram

```
parameters.json
     │
     ├─► reliability_range_sn [0.85, 0.99]
     │   │
     │   └─► SN.__init__()
     │       │
     │       └─► Random assign to:
     │           - Snodes.reliability
     │           - Sedges.reliability
     │
     ├─► reliability_range_vnr [0.90, 0.99]
     │   │
     │   └─► VNR.__init__()
     │       │
     │       └─► Random assign to:
     │           - Vnfs.reliability
     │           - Vedges.reliability
     │
     └─► vnr_reliability_weight = 0.15
         │
         └─► Solver.__init__()
             │
             └─► Used in getReward():
                 final_reward = (1-w)×basic + w×reliability
```

## Reliability Calculation Flow

```
VNR Placement Successful
       │
       ▼
calculateReliability(vnr, sn)
       │
       ├─► Calculate Node Reliability
       │   └─ For each vnr.nodemapping[i]:
       │      node_rel *= sn.snode[i].reliability
       │
       ├─► Calculate VNF Reliability
       │   └─ For each vnf in vnr.vnode:
       │      vnf_rel *= vnf.reliability
       │
       ├─► Calculate Path Reliability
       │   └─ For each vedge.spc (path):
       │      path_rel *= sn.sedege[j].reliability
       │
       └─► Final Reliability
           = node_rel × vnf_rel × path_rel
           = Value between 0 and 1
           │
           ▼
        Used in getReward()
```

## Reward Function Evolution

### Before (Original)
```
reward = sigma × R2C + (1 - sigma) × e^(-p_load) × balance_factor
         └─────────────────────────────────────────────────────┘
                    2 objectives: Cost + Load
```

### After (With Reliability)
```
basic_reward = sigma × R2C + (1 - sigma) × e^(-p_load) × balance_factor
                                  ▲
                                  │
final_reward = (1 - w) × basic_reward + w × reliability
                └──────────┬─────────┘    └────────┬────────┘
                    85% weight                  15% weight
                  Cost + Load               Reliability
                  (if w=0.15)               (if w=0.15)
```

## Configuration Impact Matrix

```
┌─────────────────────┬──────────────┬──────────────┬──────────────┐
│ Configuration       │ Reliability  │ Acceptance   │ Cost         │
│                     │ Focus        │ Ratio        │ Efficiency   │
├─────────────────────┼──────────────┼──────────────┼──────────────┤
│ w = 0.0  (Disabled) │ Very Low     │ Very High    │ Very High    │
│ w = 0.05 (Low)      │ Low          │ High         │ High         │
│ w = 0.15 (Default)  │ Medium       │ Medium       │ Medium       │
│ w = 0.5  (High)     │ Very High    │ Low          │ Low          │
│ w = 1.0  (Pure Rel) │ Maximum      │ Very Low     │ Very Low     │
└─────────────────────┴──────────────┴──────────────┴──────────────┘

Higher reliability_weight → Prefer reliable nodes, fewer placements
Lower reliability_weight → Maximize cost efficiency, more placements
```

## Class Hierarchy (Modified)

```
Solver (Base Class) ◄──── NEW: reliability_weight param
   │
   ├─► GNNDQN ◄────── Updated: +reliability_weight
   │   └─► Uses: calculateReliability() + modified getReward()
   │
   ├─► GNNDRL ◄────── Updated: +reliability_weight
   │   └─► Uses: calculateReliability() + modified getReward()
   │
   ├─► FirstFit ◄───── Updated: +reliability_weight
   │   └─► Uses: calculateReliability() + modified getReward()
   │
   ├─► GNNDRL2 ◄────── Updated: +reliability_weight
   │   └─► Uses: calculateReliability() + modified getReward()
   │
   └─► GNNDRLPPO ◄─── Updated: +reliability_weight
       └─► Uses: calculateReliability() + modified getReward()

Grasp ◄─ Also inherits from Solver
   └─► Updated: +reliability_weight
```

## Component Changes Matrix

```
┌──────────────┬──────────────┬─────────────────────────────────┐
│ Component    │ Type         │ Changes                         │
├──────────────┼──────────────┼─────────────────────────────────┤
│ parameters.  │ Config File  │ + 3 new parameters              │
│ json         │              │                                 │
├──────────────┼──────────────┼─────────────────────────────────┤
│ node.py      │ Data Model   │ + reliability attr (Vnf, Snode) │
├──────────────┼──────────────┼─────────────────────────────────┤
│ edege.py     │ Data Model   │ + reliability attr (Vedege,    │
│              │              │   Sedege)                       │
├──────────────┼──────────────┼─────────────────────────────────┤
│ vnr.py       │ Generator    │ + reliability init for VNF/     │
│              │              │   Vedges                        │
├──────────────┼──────────────┼─────────────────────────────────┤
│ substrate.py │ Generator    │ + reliability init for Snode/   │
│              │              │   Sedges                        │
├──────────────┼──────────────┼─────────────────────────────────┤
│ generator.py │ Generator    │ + reliability_range param       │
├──────────────┼──────────────┼─────────────────────────────────┤
│ solver.py    │ Core Logic   │ + calculateReliability()        │
│              │              │ ~ modified getReward()          │
│              │              │ ~ updated all subclasses        │
├──────────────┼──────────────┼─────────────────────────────────┤
│ grasp_solver │ Solver       │ ~ updated constructor           │
│ .py          │              │                                 │
├──────────────┼──────────────┼─────────────────────────────────┤
│ main.py      │ Orchestration│ + config loading                │
│              │              │ + param passing                 │
└──────────────┴──────────────┴─────────────────────────────────┘
```

## Key Methods

```
calculateReliability(vnr, sn)
    ├─ Input: VNR (with node/edge mapping), SN
    ├─ Process: Multiply all reliability values
    └─ Output: Overall reliability [0, 1]

getReward(vnr, sn)
    ├─ OLD: reward = sigma×R2C + (1-sigma)×load_balance
    ├─ NEW: reward = (1-w)×[basic] + w×[reliability]
    └─ Output: (r2c, p_load, final_reward)
```

## Usage Workflow

```
User
  │
  ├─► 1. Edit parameters.json
  │   - Add 3 reliability parameters
  │   - No code changes needed!
  │
  ├─► 2. Run main.py
  │   - Loads parameters
  │   - Creates SN with reliability
  │   - Generates VNRs with reliability
  │   - Creates solvers with reliability_weight
  │
  ├─► 3. Simulation runs
  │   - Solvers learn to optimize reliability
  │   - Reward includes reliability component
  │   - Results track reliability metrics
  │
  └─► 4. Review results
      - Check reliability of placements
      - Analyze cost vs. reliability trade-off
      - Compare with different weight values
```

## Decision Tree: Choosing reliability_weight

```
Does your use case require
high reliability?
    │
    ├─ YES, mission-critical
    │  │
    │  └─► Use: 0.4 - 0.7
    │      └─ Example: Healthcare, Finance
    │
    ├─ MAYBE, standard service
    │  │
    │  └─► Use: 0.15 - 0.3 (DEFAULT 0.15)
    │      └─ Example: Enterprise, Cloud
    │
    └─ NO, cost-focused
       │
       └─► Use: 0.01 - 0.1
           └─ Example: Research, Development
```

---

**This visual overview shows the complete architecture and impact of the reliability constraint implementation.**
