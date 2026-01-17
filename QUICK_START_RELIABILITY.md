# Quick Start Guide - Reliability Constraint

## What Was Added

A **reliability constraint** has been added to the VNE-Sim simulator. This allows optimization of virtual network placements based on node and edge reliability, in addition to traditional cost and load balancing metrics.

## Quick Setup (5 minutes)

### Step 1: Update parameters.json

Add these three lines to your `parameters.json` configuration:

```json
{
  ...existing configuration...
  
  "reliability_range_sn": [0.85, 0.99],
  "reliability_range_vnr": [0.90, 0.99],
  "vnr_reliability_weight": 0.15,
  
  ...rest of configuration...
}
```

**Meaning:**
- Substrate nodes/edges will have random reliability between 85% and 99%
- VNF nodes/edges will have random reliability between 90% and 99%  
- Reliability accounts for 15% of the solver's optimization objective

### Step 2: Run the Simulator

No code changes needed! Just run normally:

```bash
python main.py
```

The simulator will automatically:
1. Assign reliability values to all nodes and edges
2. Calculate placement reliability during VNR placement
3. Include reliability in the reward/objective function

### Step 3: Observe Results

The solvers will now learn to prefer:
- High-reliability substrate nodes
- High-reliability substrate edges
- VNRs with high reliability requirements on high-reliability nodes

## Configuration Presets

Use these pre-configured settings for common scenarios:

### For Critical Services (High Reliability)
```json
{
  "reliability_range_sn": [0.95, 0.99],
  "reliability_range_vnr": [0.95, 0.99],
  "vnr_reliability_weight": 0.50
}
```

### For Cost-Sensitive Services (Low Reliability)
```json
{
  "reliability_range_sn": [0.80, 0.95],
  "reliability_range_vnr": [0.85, 0.95],
  "vnr_reliability_weight": 0.05
}
```

### For Balanced Scenarios (Default)
```json
{
  "reliability_range_sn": [0.85, 0.99],
  "reliability_range_vnr": [0.90, 0.99],
  "vnr_reliability_weight": 0.15
}
```

See `RELIABILITY_CONFIGURATION_GUIDE.md` for more detailed examples.

## How It Works (Technical)

### Reliability Calculation
When a VNR is placed, the overall reliability is:

```
Reliability = (∏ VNF reliabilities) × (∏ Node reliabilities) × (∏ Edge reliabilities)
```

All reliabilities are between 0 and 1, so the product gives a value between 0 and 1.

### Reward Function
The solver's reward for a placement is:

```
Reward = (1 - weight) × [R2C + LoadBalancing] + weight × Reliability
       = (1 - 0.15) × [R2C + LoadBalancing] + 0.15 × Reliability
       = 0.85 × [R2C + LoadBalancing] + 0.15 × Reliability
```

Where:
- **R2C** = Revenue-to-Cost ratio (higher is better)
- **LoadBalancing** = Measure of even distribution (higher is better)
- **Reliability** = Overall reliability (higher is better)
- **weight** = `vnr_reliability_weight` from configuration

### Optimization Process
The DRL agents learn to maximize this reward, which means they learn to:
1. Efficiently use resources (high R2C)
2. Balance load across nodes (load balancing)
3. Prefer high-reliability nodes and edges (reliability)

The balance between these three is controlled by `vnr_reliability_weight`.

## Files Modified

| File | Changes |
|------|---------|
| `parameters.json` | Added 3 new configuration parameters |
| `node.py` | Added `reliability` attribute to Vnf and Snode |
| `edege.py` | Added `reliability` attribute to Vedege and Sedege |
| `vnr.py` | Modified VNR class to initialize reliability values |
| `substrate.py` | Modified SN class to initialize reliability values |
| `generator.py` | Added reliability_range parameter to VNR generator |
| `solver.py` | Added `calculateReliability()` method and modified `getReward()` |
| `grasp_solver.py` | Updated Grasp class constructor |
| `main.py` | Updated to pass reliability parameters to SN and Solvers |

## Key Methods

### calculateReliability(vnr, sn)
Computes the overall reliability of a VNR placement in the substrate network.

**Returns:** Float between 0 and 1 representing the reliability

```python
reliability = solver.calculateReliability(vnr, substrate_network)
print(f"Placement reliability: {reliability:.4f}")  # e.g., 0.8234
```

### getReward(vnr, sn)
Modified to include reliability component in the reward calculation.

**Returns:** Tuple of (r2c, p_load, reward)

```python
r2c, p_load, reward = solver.getReward(vnr, substrate_network)
print(f"Reward: {reward:.4f}")  # e.g., 0.7523
```

## Backward Compatibility

✅ **100% backward compatible**

If you don't add the new parameters to `parameters.json`:
- Reliability will use defaults (ranges will be [0.90, 0.99] for VNR and [0.85, 0.99] for SN)
- `reliability_weight` will be 0.15 (15%)
- Everything works exactly as before

## Advanced Usage

### Disable Reliability Constraint
Set `vnr_reliability_weight` to 0.0:
```json
"vnr_reliability_weight": 0.0
```

This makes the reward function ignore reliability entirely, reverting to original behavior.

### Custom Reliability Ranges
Modify the ranges to match your infrastructure:

```json
{
  "reliability_range_sn": [0.99, 0.999],      // Very high reliability substrate
  "reliability_range_vnr": [0.95, 0.999],     // High reliability VNRs
  "vnr_reliability_weight": 0.40
}
```

### Monitor Reliability in Output
The reliability metrics will appear in solver logs and results. Look for:
- Average placement reliability
- Reliability vs. cost trade-offs
- Acceptance ratio changes

## Troubleshooting

### Reliability always 0?
- Check that VNRs and SNs were created with reliability values
- Verify `calculateReliability()` is being called in `getReward()`
- Ensure mappings (nodemapping, edgemapping) are set before calculating

### Solvers ignoring reliability?
- Check `vnr_reliability_weight` is not 0.0
- Verify weight is being passed to solver constructors
- Check that reliability values are assigned (should be 0.85-0.99)

### Parameters not loading?
- Use `.get()` method with defaults (already done in main.py)
- Verify JSON syntax is correct in parameters.json
- Check file is in the same directory as main.py

## Next Steps

1. **Run simulations** with different `vnr_reliability_weight` values
2. **Analyze results** to see how reliability affects placement quality
3. **Compare** reliability vs. cost metrics
4. **Tune** weight value for your specific use case
5. **See RELIABILITY_CONFIGURATION_GUIDE.md** for detailed examples

## Documentation

For more detailed information, see:
- `RELIABILITY_CONSTRAINT.md` - Full technical documentation
- `IMPLEMENTATION_SUMMARY.md` - Overview of changes
- `RELIABILITY_CONFIGURATION_GUIDE.md` - Configuration examples and scenarios
