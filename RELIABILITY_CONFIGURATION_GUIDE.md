# Reliability Configuration Examples

This file provides configuration examples for different reliability scenarios and use cases.

## Example 1: Balanced Reliability (Default)

**Use case:** Standard production environment with moderate reliability requirements

```json
{
  "reliability_range_sn": [0.85, 0.99],
  "reliability_range_vnr": [0.90, 0.99],
  "vnr_reliability_weight": 0.15
}
```

**Characteristics:**
- Substrate nodes: 85-99% reliability
- VNF/edges: 90-99% reliability
- Reward: 85% based on R2C + load balancing, 15% based on reliability
- Best for: Balanced cost vs. reliability trade-off

---

## Example 2: High Reliability Requirements

**Use case:** Critical services (healthcare, finance, emergency response)

```json
{
  "reliability_range_sn": [0.95, 0.99],
  "reliability_range_vnr": [0.95, 0.99],
  "vnr_reliability_weight": 0.50
}
```

**Characteristics:**
- Substrate nodes: 95-99% reliability
- VNF/edges: 95-99% reliability  
- Reward: 50% based on R2C + load balancing, 50% based on reliability
- Best for: Mission-critical applications requiring high availability

---

## Example 3: Cost-Optimized with Minimum Reliability

**Use case:** Non-critical services with loose SLAs

```json
{
  "reliability_range_sn": [0.80, 0.95],
  "reliability_range_vnr": [0.85, 0.95],
  "vnr_reliability_weight": 0.05
}
```

**Characteristics:**
- Substrate nodes: 80-95% reliability
- VNF/edges: 85-95% reliability
- Reward: 95% based on R2C + load balancing, 5% based on reliability
- Best for: Cost-sensitive deployments

---

## Example 4: Strict Reliability SLA

**Use case:** Telecom/network services with strict uptime requirements

```json
{
  "reliability_range_sn": [0.99, 0.9999],
  "reliability_range_vnr": [0.99, 0.9999],
  "vnr_reliability_weight": 0.70
}
```

**Characteristics:**
- Substrate nodes: 99-99.99% reliability
- VNF/edges: 99-99.99% reliability
- Reward: 30% based on R2C + load balancing, 70% based on reliability
- Best for: Telecom infrastructure, critical network services
- Note: May use specialized, higher-cost hardware

---

## Example 5: Progressive Reliability Tiers

**Scenario:** Multi-tier cloud with different SLA levels

### Tier 1: Standard Service
```json
{
  "reliability_range_sn": [0.90, 0.97],
  "reliability_range_vnr": [0.92, 0.97],
  "vnr_reliability_weight": 0.10
}
```

### Tier 2: Premium Service
```json
{
  "reliability_range_sn": [0.95, 0.99],
  "reliability_range_vnr": [0.96, 0.99],
  "vnr_reliability_weight": 0.30
}
```

### Tier 3: Ultra-Premium Service
```json
{
  "reliability_range_sn": [0.99, 0.9999],
  "reliability_range_vnr": [0.99, 0.9999],
  "vnr_reliability_weight": 0.60
}
```

---

## Example 6: Geographic Redundancy

**Use case:** Multi-region deployment with failover

```json
{
  "reliability_range_sn": [0.98, 0.9999],
  "reliability_range_vnr": [0.98, 0.9999],
  "vnr_reliability_weight": 0.40
}
```

**Characteristics:**
- High reliability ranges for both SN and VNR
- Moderate reliability weight to balance with load distribution
- Expects backup/redundancy mechanisms to be configured elsewhere

---

## Example 7: Development/Testing Environment

**Use case:** Non-production testing environment

```json
{
  "reliability_range_sn": [0.70, 0.95],
  "reliability_range_vnr": [0.75, 0.95],
  "vnr_reliability_weight": 0.01
}
```

**Characteristics:**
- Lower reliability requirements
- Minimal reliability weight (nearly ignored)
- Maximum focus on cost efficiency
- Best for: Testing placement algorithms

---

## How to Modify Reliability Settings

### To increase reliability focus:
1. **Increase `reliability_weight`**: Higher values (0.5-0.9) prioritize reliability
2. **Narrow reliability ranges**: Use tighter min-max ranges (e.g., [0.95, 0.99])
3. **Increase minimum reliability**: Require nodes with higher baseline reliability

### To decrease reliability focus:
1. **Decrease `reliability_weight`**: Lower values (0.01-0.1) minimize reliability impact
2. **Expand reliability ranges**: Use broader ranges (e.g., [0.80, 0.99])
3. **Decrease minimum reliability**: Allow nodes with lower baseline reliability

---

## Impact on Learning

The reliability constraint affects solver learning behavior:

### High Reliability Weight (0.5-0.9)
- Agents learn to prefer high-reliability nodes
- May sacrifice some cost efficiency for reliability
- Longer convergence time in learning
- More stable, predictable placements

### Low Reliability Weight (0.01-0.15)
- Agents prioritize cost and load balancing
- Reliability becomes secondary consideration
- Faster convergence in learning
- More aggressive optimization

### Zero Reliability Weight
```python
"vnr_reliability_weight": 0.00
```
- Reliability constraint completely disabled
- Equivalent to original behavior (before reliability implementation)
- Pure R2C + load balancing optimization

---

## Monitoring and Tuning

### Metrics to Monitor
1. **Average Placement Reliability**: Should match your target
2. **Acceptance Ratio**: Higher reliability weight may reduce acceptance
3. **Revenue per VNR**: Check if optimization is cost-effective
4. **Node Load Distribution**: Should remain balanced

### Adjustment Strategy
1. Start with default (0.15 weight)
2. Monitor reliability metrics for 1-2 iterations
3. Increase weight if reliability is too low
4. Decrease weight if acceptance ratio drops too much
5. Find optimal balance for your use case

---

## Common Configurations for Industry Scenarios

### Cloud Provider
```json
{
  "reliability_range_sn": [0.95, 0.9999],
  "reliability_range_vnr": [0.95, 0.99],
  "vnr_reliability_weight": 0.20
}
```

### Edge Computing Provider
```json
{
  "reliability_range_sn": [0.90, 0.99],
  "reliability_range_vnr": [0.92, 0.98],
  "vnr_reliability_weight": 0.25
}
```

### Telecom Operator
```json
{
  "reliability_range_sn": [0.999, 0.9999],
  "reliability_range_vnr": [0.999, 0.9999],
  "vnr_reliability_weight": 0.75
}
```

### Research/Academic Network
```json
{
  "reliability_range_sn": [0.85, 0.98],
  "reliability_range_vnr": [0.90, 0.98],
  "vnr_reliability_weight": 0.12
}
```

---

## Combining with Other Parameters

The reliability constraint works alongside existing parameters:

```json
{
  "beta": 0.5,                          // Network generation parameter
  "cpu_range": [9, 28],
  "bw_range": [28, 55],
  "lt_range": [1, 9],
  "reliability_range_sn": [0.85, 0.99],  // NEW
  
  "vnfs_range": [3, 7],
  "vcpu_range": [1, 11],
  "vbw_range": [1, 11],
  "vlt_range": [16, 25],
  "reliability_range_vnr": [0.90, 0.99], // NEW
  "vnr_reliability_weight": 0.15,         // NEW
  
  "solvers": [
    {
      "name": "FirstFit",
      "sigma": 0.5,
      "type": "FF"
    }
  ]
}
```

All existing parameters continue to work as before, with reliability being additive.
