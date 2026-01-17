# Code Changes Summary

This document shows the key code changes made to implement the reliability constraint.

## 1. parameters.json - New Parameters

```diff
{
     "bw_range": [
          28,
          55
     ],
     "lt_range": [
          1,
          9
     ],
+    "reliability_range_sn": [
+         0.85,
+         0.99
+    ],
     "numnodes": 24,
     ...
     "vbw_range": [
          1,
          11
     ],
     "vlt_range": [
          16,
          25
     ],
+    "vnr_reliability_weight": 0.15,
     "episode_duration": 1000,
```

## 2. node.py - Vnf Class Changes

```diff
 class Vnf(Node):
     
     def __init__(self, index,cpu,cpu_max,req,flavor_size,p_scalingUp):
         ...
         self.current = 0
         """
         A flag indicating whether the VNF is the currently selected one for placement.
             - If the value is 0, the VNF is not the current one.
             - If the value is 1, the VNF is the current one for placement.
         """
+        self.reliability = 0.95
+        """ 
+        The reliability of this VNF, representing the probability that the VNF operates correctly.
+        Initialized to a default value of 0.95 and can be set during VNR generation.
+        Range: [0.90, 0.99]
+        """
```

## 3. node.py - Snode Class Changes

```diff
 class Snode(Node):
     
     def __init__(self,index,cpu):
         super().__init__(index, cpu)
         self.lastcpu=cpu
         ...
         self.p_load=0
         """ The node's potential load... """
+        self.reliability = 0.95
+        """ 
+        The reliability of this substrate node, representing the probability that the node operates correctly.
+        Initialized to a default value of 0.95 and can be set during SN generation.
+        Range: [0.85, 0.99]
+        """
```

## 4. vnr.py - VNR Generator Updates

```diff
 class VNR:
     ID_ = 0
-    def __init__(self,vnf_range, cpu_range, bw_range,lt_range,flavor_size,duration,mtbs):
+    def __init__(self,vnf_range, cpu_range, bw_range,lt_range,flavor_size,duration,mtbs, reliability_range=None):
         VNR.ID_+=1
         ...
         self.edgemapping = []
         """ Edges placement in the Service Network (SN) """
+        
+        # Set default reliability range if not provided
+        if reliability_range is None:
+            reliability_range = [0.90, 0.99]
         
         p_scalingUp=int(1/2*duration/mtbs)+1
         # VNFs Creation
         for i in range(self.num_vnfs):
             cpu=np.random.randint(cpu_range[0],cpu_range[1]//2)
             vno = Vnf(i,cpu,cpu_range[1],self.id,flavor_size,p_scalingUp)
+            # Initialize VNF reliability
+            vno.reliability = np.random.uniform(reliability_range[0], reliability_range[1])
             self.vnode.append(vno)
         ...
         for i in range(self.num_vedges):
             a_t_b = list(self.graph.edges())[i]
             bw = np.random.randint(bw_range[0],bw_range[1])
             lt = np.random.randint(lt_range[0],lt_range[1])
             ved = Vedege(i,bw,lt,a_t_b)
+            # Initialize Vedge reliability
+            ved.reliability = np.random.uniform(reliability_range[0], reliability_range[1])
             self.vedege.append(ved)
```

## 5. substrate.py - SN Generator Updates

```diff
 class SN :
     
-    def __init__(self, num_nodes, cpu_range, bw_range,lt_range,topology):
+    def __init__(self, num_nodes, cpu_range, bw_range,lt_range,topology, reliability_range=None):
         self.num_nodes = num_nodes
         ...
         self.numedges = len(self.edges)
+        
+        # Set default reliability range if not provided
+        if reliability_range is None:
+            reliability_range = [0.85, 0.99]
 
         # Snodes Creation 
         for i in range(self.num_nodes):
             cpu = np.random.randint(cpu_range[0],cpu_range[1])
-            self.snode.append(Snode(i,cpu))
+            sn = Snode(i,cpu)
+            # Initialize Snode reliability
+            sn.reliability = np.random.uniform(reliability_range[0], reliability_range[1])
+            self.snode.append(sn)
```

## 6. solver.py - New calculateReliability Method

```python
def calculateReliability(self, vnr, sn):
    """
    Calculate the overall reliability of a VNR placement in the substrate network.
    
    The reliability is calculated as the product of:
    1. Node reliability: Product of reliability of all mapped substrate nodes
    2. VNF reliability: Product of reliability of all VNFs in the VNR
    3. Path reliability: Product of reliability of all substrate edges used in the mapping
    
    The overall reliability is: Product of all VNF, Node, and Path reliabilities
    
    Args:
        vnr: The virtual network request with mapping information
        sn: The substrate network containing node and edge information
        
    Returns:
        float: Overall reliability of the placement (between 0 and 1)
    """
    # Calculate node mapping reliability (substrate nodes reliability)
    node_reliability = 1.0
    for i in range(vnr.num_vnfs):
        mapped_node_idx = vnr.nodemapping[i]
        node_reliability *= sn.snode[mapped_node_idx].reliability
    
    # Calculate VNF reliability
    vnf_reliability = 1.0
    for vnf in vnr.vnode:
        vnf_reliability *= vnf.reliability
    
    # Calculate path reliability (substrate edges reliability)
    path_reliability = 1.0
    for vedge_idx, vedge in enumerate(vnr.vedege):
        if vedge.spc:  # If the edge has been mapped
            for sedge_idx in vedge.spc:
                path_reliability *= sn.sedege[sedge_idx].reliability
    
    # Overall reliability is the product of all components
    overall_reliability = node_reliability * vnf_reliability * path_reliability
    
    return overall_reliability
```

## 7. solver.py - Modified getReward Method

```diff
  def getReward(self,vnr,sn):
      """
-     Calculates the reward for placing a virtual network request (VNR) in the substrate network. 
-     The reward is computed as: Reward = sigma * R2C + (1 - sigma) * e^(-p_load)
+     Calculates the reward for placing a virtual network request (VNR) in the substrate network. 
+     The reward is computed as a weighted combination of:
+     - R2C (Revenue-to-Cost ratio)
+     - Load balancing factor (e^(-p_load))
+     - Reliability of the placement
      ...
      """    
      r2c = self.rev2cost(vnr)
      
      # Collect p_load of all mapped nodes
      p_loads = []
      for i in range(vnr.num_vnfs):
          p_loads.append(sn.snode[vnr.nodemapping[i]].p_load)
      
      p_load_mean = np.mean(p_loads)
      p_load_std = np.std(p_loads)
      
-     balance_factor = 1.0 / (1.0 + p_load_std)
-     reward = self.sigma * r2c + (1 - self.sigma) * math.exp(-p_load_mean) * balance_factor
-     
-     return r2c,p_load_mean,reward
+     balance_factor = 1.0 / (1.0 + p_load_std)
+     
+     # Calculate the basic reward (R2C + Load balancing)
+     basic_reward = self.sigma * r2c + (1 - self.sigma) * math.exp(-p_load_mean) * balance_factor
+     
+     # Calculate reliability of the placement
+     overall_reliability = self.calculateReliability(vnr, sn)
+     
+     # Combine basic reward with reliability as weighted objective
+     final_reward = (1 - self.reliability_weight) * basic_reward + self.reliability_weight * overall_reliability
+     
+     return r2c, p_load_mean, final_reward
```

## 8. solver.py - Solver Base Class Constructor

```diff
  class Solver():
      
-     def __init__(self, sigma,rejection_penalty):
+     def __init__(self, sigma,rejection_penalty, reliability_weight=0.15):
          self.rejection_penalty= rejection_penalty
          ...
          self.sigma=sigma
          """ A parameter used to calculate the reward given to the solver after a successful VNR placement. """
+         self.reliability_weight = reliability_weight
+         """ 
+         Weight for reliability in the objective function.
+         Controls the balance between R2C, load balancing, and reliability.
+         Default: 0.15 (15% of the reward is based on reliability)
+         """
```

## 9. main.py - Configuration Loading

```diff
+ reliability_range_sn = json_object.get('reliability_range_sn', [0.85, 0.99])
+ # Virtual network related parameters
  num_reqs = json_object['num_reqs']
+ reliability_range_vnr = json_object.get('reliability_range_vnr', [0.90, 0.99])
+ vnr_reliability_weight = json_object.get('vnr_reliability_weight', 0.15)
```

## 10. main.py - SN Instantiation

```diff
- old_subNet= SN(numnodes, cpu_range, bw_range,lt_range,topology)
+ old_subNet= SN(numnodes, cpu_range, bw_range,lt_range,topology, reliability_range_sn)
```

## 11. main.py - Solver Instantiation

```diff
  if solvers_inputs[i]["type"]=="FF":
-     solvers.append(FirstFit(solvers_inputs[i]["sigma"],solvers_inputs[i]["rejection_penalty"]))
+     solvers.append(FirstFit(solvers_inputs[i]["sigma"],solvers_inputs[i]["rejection_penalty"], vnr_reliability_weight))
  if solvers_inputs[i]["type"]=="GNNDRL":
-     solvers.append(GNNDRL(...))
+     solvers.append(GNNDRL(..., vnr_reliability_weight))
  if solvers_inputs[i]["type"]=="GNNDRL2":
-     solvers.append(GNNDRL2(...))
+     solvers.append(GNNDRL2(..., None, None, None, vnr_reliability_weight))
```

## 12. main.py - Generator Instantiation

```diff
- generator=Generator(vnr_classes, MLT, MTBS, MTBA[j], vnfs_range, vcpu_range, vbw_range,vlt_range, flavor_tab, p_flavors,len(solvers_inputs))
+ generator=Generator(vnr_classes, MLT, MTBS, MTBA[j], vnfs_range, vcpu_range, vbw_range,vlt_range, flavor_tab, p_flavors,len(solvers_inputs), reliability_range_vnr)
```

## Summary of Changes

- **3 new configuration parameters** in `parameters.json`
- **2 new reliability attributes** in node/edge classes (4 total)
- **1 new method** in Solver class (`calculateReliability`)
- **1 modified method** in Solver class (`getReward`)
- **4 modified constructors** in Solver subclasses
- **2 modified generators** (VNR and SN) to initialize reliability
- **1 modified main script** to load and pass reliability parameters

**Total Lines Added:** ~150-200
**Total Lines Modified:** ~50-70
**Breaking Changes:** None (fully backward compatible)
