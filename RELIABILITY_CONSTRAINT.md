# Reliability Constraint Implementation

## Overview
This document describes the implementation of a **reliability constraint** for Virtual Network Requests (VNRs) and Substrate Networks (SNs) in the VNE-Sim simulator. The reliability is now integrated as an objective function to optimize across all solvers.

## Parameters Added to `parameters.json`

### Substrate Network (SN) Reliability
```json
"reliability_range_sn": [0.85, 0.99]
```
- Defines the range of reliability values for substrate nodes and edges
- Range: 0.85 to 0.99 (85% to 99% availability)

### Virtual Network Request (VNR) Reliability
```json
"reliability_range_vnr": [0.90, 0.99]
```
- Defines the range of reliability values for VNF nodes and virtual edges
- Range: 0.90 to 0.99 (90% to 99% availability)

### Reliability Weight in Objective Function
```json
"vnr_reliability_weight": 0.15
```
- Controls the balance between:
  - Revenue-to-Cost (R2C) ratio
  - Load balancing factor
  - Reliability
- Default: 0.15 (15% of the reward is based on reliability)
- Higher values prioritize reliability over other metrics

## Code Modifications

### 1. **node.py** - Node Classes
Added `reliability` attribute to all node types:

#### Vnf (Virtual Network Function)
```python
self.reliability = 0.95
```
- Randomly initialized during VNR generation within `reliability_range_vnr`
- Represents the probability that the VNF operates correctly

#### Snode (Substrate Node)
```python
self.reliability = 0.95
```
- Randomly initialized during SN creation within `reliability_range_sn`
- Represents the probability that the substrate node operates correctly

### 2. **edege.py** - Edge Classes
Added `reliability` attribute to all edge types:

#### Vedege (Virtual Edge)
```python
self.reliability = 0.95
```
- Randomly initialized during VNR generation within `reliability_range_vnr`
- Represents the reliability requirement for the virtual edge

#### Sedege (Substrate Edge)
```python
self.reliability = 0.95
```
- Randomly initialized during SN creation within `reliability_range_sn`
- Represents the reliability of the substrate edge

### 3. **vnr.py** - VNR Generator
Modified the VNR class constructor:
```python
def __init__(self, vnf_range, cpu_range, bw_range, lt_range, flavor_size, 
             duration, mtbs, reliability_range=None):
```

- Added optional `reliability_range` parameter (defaults to [0.90, 0.99])
- VNF and Vedge reliability values are randomly assigned during generation

### 4. **substrate.py** - Substrate Network Generator
Modified the SN class constructor:
```python
def __init__(self, num_nodes, cpu_range, bw_range, lt_range, topology, 
             reliability_range=None):
```

- Added optional `reliability_range` parameter (defaults to [0.85, 0.99])
- Snode and Sedge reliability values are randomly assigned during generation

### 5. **generator.py** - VNR Generator
Added reliability_range parameter:
```python
def __init__(self, vnr_classes, mlt, mtbs, mtba, vnfs_range, vcpu_range, 
             vbw_range, vlt_range, flavor_tab, p_flavors, nb_solvers, 
             reliability_range=None):
```

- Passes `reliability_range` to VNR instances during generation
- Allows for configurable reliability ranges per scenario

### 6. **solver.py** - Core Solver Classes

#### Base Solver Class
Added `reliability_weight` parameter:
```python
def __init__(self, sigma, rejection_penalty, reliability_weight=0.15):
    self.reliability_weight = reliability_weight
```

#### New Method: `calculateReliability()`
Calculates the overall reliability of a VNR placement:
```python
def calculateReliability(self, vnr, sn):
    """
    Overall Reliability = 
        Product of all VNF reliabilities × 
        Product of all Substrate Node reliabilities × 
        Product of all Substrate Edge reliabilities
    """
```

The reliability is computed as:
1. **Node Reliability**: Product of reliability values of all mapped substrate nodes
2. **VNF Reliability**: Product of reliability values of all VNFs in the VNR
3. **Path Reliability**: Product of reliability values of all substrate edges used in the mapping

#### Modified Method: `getReward()`
Updated reward function to include reliability:

**Previous Formula:**
```
Reward = sigma × R2C + (1 - sigma) × e^(-p_load) × balance_factor
```

**New Formula:**
```
basic_reward = sigma × R2C + (1 - sigma) × e^(-p_load) × balance_factor
overall_reliability = calculateReliability(vnr, sn)
final_reward = (1 - reliability_weight) × basic_reward + reliability_weight × overall_reliability
```

This balances three optimization objectives:
- **R2C (Revenue-to-Cost)**: Efficient resource utilization
- **Load Balancing**: Even distribution of load across substrate nodes
- **Reliability**: High-reliability path and node selection

#### All Solver Subclasses Updated:
- `GNNDQN`: Added `reliability_weight=0.15` parameter
- `GNNDRL`: Added `reliability_weight=0.15` parameter
- `FirstFit`: Added `reliability_weight=0.15` parameter
- `GNNDRL2`: Added `reliability_weight=0.15` parameter
- `GNNDRLPPO`: Added `reliability_weight=0.15` parameter

### 7. **grasp_solver.py** - GRASP Solver
Modified Grasp class constructor:
```python
def __init__(self, sigma, rejection_penalty, max_iter=20, alpha=0.3, 
             reliability_weight=0.15):
    super().__init__(sigma, rejection_penalty, reliability_weight)
```

### 8. **main.py** - Main Simulation Script
Added configuration parameters:
```python
reliability_range_sn = json_object.get('reliability_range_sn', [0.85, 0.99])
reliability_range_vnr = json_object.get('reliability_range_vnr', [0.90, 0.99])
vnr_reliability_weight = json_object.get('vnr_reliability_weight', 0.15)
```

Updated SN creation:
```python
old_subNet = SN(numnodes, cpu_range, bw_range, lt_range, topology, reliability_range_sn)
```

Updated Generator creation:
```python
generator = Generator(vnr_classes, MLT, MTBS, MTBA[j], vnfs_range, vcpu_range, 
                      vbw_range, vlt_range, flavor_tab, p_flavors, 
                      len(solvers_inputs), reliability_range_vnr)
```

Updated Solver instantiation (all solver types):
```python
# FirstFit
FirstFit(sigma, rejection_penalty, vnr_reliability_weight)

# GNNDRL
GNNDRL(..., vnr_reliability_weight)

# GNNDRL2
GNNDRL2(..., None, None, None, vnr_reliability_weight)

# GNNDRLPPO
GNNDRLPPO(..., vnr_reliability_weight)
```

## How It Works

### 1. VNR Generation
- Each VNF is assigned a random reliability value from `reliability_range_vnr`
- Each virtual edge is assigned a random reliability value from `reliability_range_vnr`

### 2. SN Creation
- Each substrate node is assigned a random reliability value from `reliability_range_sn`
- Each substrate edge is assigned a random reliability value from `reliability_range_sn`

### 3. Placement Evaluation
When a VNR is successfully placed:
1. Calculate the basic reward (R2C + load balancing)
2. Calculate the overall reliability of the placement using `calculateReliability()`
3. Combine both metrics using the `reliability_weight` parameter

### 4. Learning
DRL agents learn to select placements that:
- Maximize revenue-to-cost ratio
- Balance load across nodes
- Prioritize high-reliability paths and nodes (based on `reliability_weight`)

## Configuration Examples

### Scenario 1: High Reliability Priority (50% weight)
```json
{
  "reliability_range_sn": [0.90, 0.99],
  "reliability_range_vnr": [0.95, 0.99],
  "vnr_reliability_weight": 0.50
}
```
The solver will prioritize selecting high-reliability nodes and edges.

### Scenario 2: Default Balanced Approach (15% weight)
```json
{
  "reliability_range_sn": [0.85, 0.99],
  "reliability_range_vnr": [0.90, 0.99],
  "vnr_reliability_weight": 0.15
}
```
The solver balances all three objectives equally.

### Scenario 3: Cost-Optimized (5% weight)
```json
{
  "reliability_range_sn": [0.80, 0.99],
  "reliability_range_vnr": [0.85, 0.99],
  "vnr_reliability_weight": 0.05
}
```
The solver prioritizes cost and load balancing while still considering reliability.

## Backward Compatibility
- All parameters have sensible defaults
- Existing code will work without modification
- To enable reliability optimization, simply add the new parameters to `parameters.json`

## Testing
To verify the reliability constraint works correctly:

1. Check that reliability values are assigned to all nodes and edges
2. Verify that `calculateReliability()` returns values between 0 and 1
3. Confirm that rewards include reliability component
4. Monitor if agents learn to prefer high-reliability paths in training

## Future Enhancements
- Add per-VNR reliability SLAs (Service Level Agreements)
- Implement reliability-aware load balancing
- Add reliability recovery mechanisms
- Support for backup paths with different reliability levels
