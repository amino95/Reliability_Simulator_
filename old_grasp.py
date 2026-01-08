# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 20:33:50 2022

@author: amirgu1
"""

import json
import numpy as np
import networkx as nx
from copy import deepcopy

from costComputation import CostComputation as CC
from reliabilityComputation import ReliabilityComputation as RC

from substrateNet import SubstrateNetwork
from virtualNetReq import VirtualNetworkRequest
from filtering import Filtering
from pathSelector import PathSelector
from updateResources import UpdateResources


class GraspPlacement:
    def __init__(self, SubNet, VNR, VNE=True):
        self.SubNet = SubNet
        self.VNE = VNE
        self.nIteration, self.Lambda, self.Individuals, self.lsActive, self.lsDepth = self.config_loader()
        self.RemainingVNFs = list(VNR.G_VNR.nodes())
        self.nodemapping = {}  # {VNF:SN, ...}
        self.edgemapping = {}  # {EDGE:PATH, ...}
        self.subNet_copy = deepcopy(self.SubNet)

    def config_loader(self, config_file="config.json"):
        """
            Loads algo params from config file
                ○ Input : config_file
		           ○ Output : algo params (nIteration, Lambda, Individuals)
        """
        with open(config_file) as json_data_file:
            data = json.load(json_data_file)
            nIteration = data["AlgoParams"]["GRASP"]["General"]["nIterations"]
            Lambda = data["AlgoParams"]["GRASP"]["General"]["Lambda"]
            Individuals = data["AlgoParams"]["GRASP"]["General"]["Individuals"]
            lsActive = data["AlgoParams"]["GRASP"]["LocalSearch"]["Active"]
            lsDepth = data["AlgoParams"]["GRASP"]["LocalSearch"]["Depth"]
        return nIteration, Lambda, Individuals, lsActive, lsDepth

    def updateConfig(self, lamda, config_file="config.json"):
        """
            Updates lambda values to automate the execution of the algortihm with different values of lamda
        """
        with open(config_file, 'r+') as json_data_file:
            data = json.load(json_data_file)
            data["AlgoParams"]["GRASP"]["General"]["Lambda"] = lamda
            json_data_file.seek(0)
            json.dump(data, json_data_file)
            json_data_file.truncate()
        return

        # Objective function to minimize

    def EvaluateSolution(self, subNet, VNR, nodemapping, LinksPlacementSol):
        """
            Evaluate solution and return the cost of the placement of this sol
                ○ Input : solution
		           ○ Output : cost of the solution
        """
        costCompute = CC(subNet, VNR, nodemapping, LinksPlacementSol, self.VNE)
        cost = costCompute.computeCost()
        return cost

    def create_partial_solution(self, VNR, FeasableSubNodes, vnf):
        """
            Creates partial placement solution for given VNF based on the feasible Nodes
                ○ Input : list of FeasableSubNodes, the given VNF
		           ○ Output : boolean for placed sol or not and solution couple (nodemapping, LinksPlacementSol)
        """
        # copy of the actual solutions
        nodemapping = deepcopy(self.nodemapping)
        LinksPlacementSol = deepcopy(self.edgemapping)
        subNet_copy_global = deepcopy(self.subNet_copy)
        filtering = Filtering(VNR, subNet_copy_global)

        # select a random node placement and remove it
        selectSubNode = np.random.choice(FeasableSubNodes)
        FeasableSubNodes.remove(selectSubNode)

        # place the vnf
        nodemapping[vnf] = selectSubNode

        # place links of the vnf with already placed vnfs
        PathsSol = True

        for i in nodemapping:
            # check if a link exist with that vnf
            if VNR.G_VNR.has_edge(vnf, i):
                # place that link
                edge = (vnf, i)
                FeasableSubLinks = filtering.GeneralSubLinksFilter(edge)
                # print("FeasableSubLinks :", FeasableSubLinks)
                if len(FeasableSubLinks) == 0:
                    # no possible path with the selected node
                    PathsSol = False
                    break
                else:
                    subCopy = deepcopy(subNet_copy_global)
                    for e in subCopy.SN_G.edges():
                        if e not in FeasableSubLinks:
                            subCopy.SN_G.remove_edge(e[0], e[1])
                            # compute the shortest paths - all_simple_paths need to be optimized
                    paths = nx.all_simple_paths(subCopy.SN_G, source=nodemapping[edge[0]],
                                                target=nodemapping[edge[1]])
                    paths = list(paths)
                    # print("all simple paths :", paths)
                    # select the best path (load balacing ...)
                    if len(paths) == 0:
                        # No path found
                        PathsSol = False
                        # print("no paths !!!")
                        break
                    else:
                        pathSel = PathSelector(subCopy, paths)
                        path = pathSel.get_shortestpathHops()  # Enhancement: a random choice from shortest paths will be more efficient here

                        LinksPlacementSol[edge] = path
                        # remove the consumed bandwidth from the selected path
                        upR = UpdateResources(VNR, subNet_copy_global)
                        # print(nodemapping, "----", LinksPlacementSol, "----", edge)
                        upR.updateVLinkResources(edge, LinksPlacementSol)

        if PathsSol:
            # print("LinksPlacementSol :", LinksPlacementSol)
            relCheck = self.checkReliability(VNR, LinksPlacementSol)
            # print("relCheck :", relCheck)
            if relCheck:
                return True, (nodemapping, LinksPlacementSol)
            else:
                # print("Error ! reliability not satisfied !")
                return False, ({}, {})
        else:
            # print("no paths !")
            return False, ({}, {})

    def ConstructGreedyRandomizedSolution(self, VNR):
        """
            Construction of the solution and then selects the best solution from the Restricled Candidate So lutions
                ○ Input : {}
		           ○ Output : boolean for finding a valid solution and the selected best solution with a minimum cost
        """
        # self.RemainingVNFs = list(VNR.G_VNR.nodes())
        if len(self.RemainingVNFs) == 0:
            # print("129", "no remaining node")
            return False, ({}, {}), 0.0
        vnf = np.random.choice(
            self.RemainingVNFs)  # Enhancement: a random choice from neighboring placed VNFs will be more efficient here
        # print("VNF ", vnf)
        self.RemainingVNFs.remove(vnf)
        # filter nodes with not enough resources
        filtering = Filtering(VNR, self.subNet_copy)
        FeasableSubNodes = filtering.GeneralSubNodesFilter(vnf)
        # print("FeasableSubNodes :", FeasableSubNodes)
        if (len(self.nodemapping) != 0 and self.VNE):
            for i in FeasableSubNodes:
                if i in self.nodemapping.values():
                    FeasableSubNodes.remove(i)
        if len(FeasableSubNodes) == 0:
            # print("141", "no feasable nodes")
            return False, ({}, {}), 0.0
        else:
            # case when there are at least one solution
            """
             here I commented the individuals variable to see the effect of exploring
             all the feasible substrate nodes 
            """
            individ = self.Individuals
            candidateSolutions = []
            while len(FeasableSubNodes) != 0 and individ > 0:
                Valid, candidateSolution = self.create_partial_solution(VNR, FeasableSubNodes, vnf)
                if not Valid:
                    # print("150", "no feasable candidate solution")
                    # individ -= 1
                    continue
                candidateSolutions.append(candidateSolution)
                individ -= 1
            if len(candidateSolutions) == 0:
                # no placement solutions found
                return False, ({}, {}), 0.0
            else:
                # construct the restricted list candidates
                RCL = []
                SolsCost = []
                for sol in candidateSolutions:
                    SolsCost.append(self.EvaluateSolution(self.subNet_copy, VNR, sol[0], sol[1]))
                # sort the solutions by cost
                SolsCost, candidateSolutions = zip(*sorted(zip(SolsCost, candidateSolutions), key=lambda x: x[0]))
                # get the worst cost
                worstCost = SolsCost[0]
                # select the best cost
                bestCost = SolsCost[-1]

                # add to RCL the solutions with f < fmin + lambda  (fmax-fmin)
                # problème de maximization : f >= fmax + lambda (fmin - fmax)
                for i in range(len(candidateSolutions)):
                    if SolsCost[i] >= (bestCost + self.Lambda * (worstCost - bestCost)):
                        RCL.append(candidateSolutions[i])
                # print("RCL : ", RCL)
                assert len(RCL) > 0, "RCL is empty"

                solidx = np.random.choice(len(RCL), size=1)
                Solution = RCL[solidx[0]]
        return True, Solution, bestCost

    def kNeighbours(self, G, start, k):
        """
            Function to return the list of k-th order neighbors of the start node
        """
        nbrs = set([start])
        for l in range(k):
            nbrs = set((nbr for n in nbrs for nbr in G[n]))
        return nbrs

    def checkReliability(self, VNR, LinksPlacementSol):

        relComp = RC(self.SubNet.SN_G)

        for path in LinksPlacementSol.values():
            # print(path)
            v = []
            r = relComp.computeReliability(path)

            # we have the same reliability value for all the edges in the VNR, so we can take the first element
            relVNR = list(nx.get_edge_attributes(VNR.G_VNR, 'Reliability').values())[0]
            if r >= relVNR:
                v.append(True)
            else:
                v.append(False)
                return False
        return True

    def localSearch(self, VNR, constructedSol, constructedSolCost):
        """
           Local search phase of the GRASP algorithm to try to find a better solution in the neighborhood of the constructed solution
               ○ Input : constrcuted Solution returned by the construction phase function
	           ○ Output : the best solution in the neighborhood of the constructed solution and its cost
        """
        # print("################ Local Search Phase ####################")
        # search in the neighbourhood of the solution if we can find a better wolution with  a better cost
        # solution : ({1: 8, 2: 15, 0: 0}, {(0, 2): [0, 17, 16, 23, 15]})
        # nodesSol : {1: 8, 2: 15, 0: 0}
        # {(2, 0): [1, 5, 17, 18], (2, 1): [1, 5, 2]}
        G = self.subNet_copy.SN_G
        # print("Constructed Solution :", constructedSol)
        bestSolution, bestCost = [], 0

        for vn_sn in constructedSol[0].items():

            vnf_id, sn_id = vn_sn[0], vn_sn[1]
            # print("#### Local search for VNF", vnf_id)
            if self.lsDepth == 1:
                neighborsList = self.kNeighbours(G, sn_id, 1)
            elif self.lsDepth == 2:
                neighborsList = self.kNeighbours(G, sn_id, 2)

            elif self.lsDepth == 3:
                neighborsList = self.kNeighbours(G, sn_id, 3)

            else:
                print("Local Search Depth is undefined !")
            # print("neighbors of", sn_id,"=", neighborsList)

            if not neighborsList:
                # print("no neighbors for this node !")
                continue
            # filtering neighbors
            filtering = Filtering(VNR, self.subNet_copy)
            feasableSubNodesNeighbors = filtering.FilteringNeighborsLocalSearch(vnf_id, neighborsList)
            if not feasableSubNodesNeighbors:
                # print("No feasable SubNode Neighbors for this node !")
                # have to do something to prevent the error and stop the placement
                bestCost = constructedSolCost
                # print("bestCost =", bestCost)
                bestSolution = constructedSol
                # print("bestSolution :", bestSolution)

                continue
            else:
                # try to place on filtered neighbors
                # calculate cost
                localCandidateSolutions = []
                while len(feasableSubNodesNeighbors) != 0:
                    Valid, localCandidateSolution = self.create_partial_solution(VNR, feasableSubNodesNeighbors, vnf_id)
                    if not Valid:
                        # print("230", "no feasable local candidate solution")

                        continue
                    localCandidateSolutions.append(localCandidateSolution)
                if len(localCandidateSolutions) == 0:
                    # no placement solutions found
                    bestSolution = constructedSol
                    return False, constructedSolCost, bestSolution
                else:

                    localSolsCost = []
                    for sol in localCandidateSolutions:
                        cost = self.EvaluateSolution(self.subNet_copy, VNR, sol[0], sol[1])
                        localSolsCost.append(cost)

                    # chosing best sol with max cost
                    maxCost = max(localSolsCost)
                    maxCostIdx = localSolsCost.index(maxCost)
                    bestCost = maxCost
                    bestSolution = localCandidateSolutions[maxCostIdx]

        return True, bestCost, bestSolution

    def runPlacement(self, VNR):
        """
           Main function of the GRASP algorithm
               ○ Input : {}
	           ○ Output : final solution in the booleans for finding paths and nodes solutions
        """
        # nx.draw(VNR.G_VNR,  with_labels=True)

        NodesPlaced, LinksPlaced = False, False
        bestCost = 0

        if self.lsActive:
            # print("Local Search is ON !")
            for i in range(len(VNR.G_VNR.nodes)):
                # print("########## Iteration", i, "#################")
                Valid, constructedSol, constructedSolCost = self.ConstructGreedyRandomizedSolution(VNR)
                # print(Valid, constructedSol, constructedSolCost)
                if Valid and constructedSol:
                    NodesPlaced, LinksPlaced = True, True
                    self.nodemapping = constructedSol[0]
                    # print("NodesSol", self.nodemapping)
                    self.edgemapping = constructedSol[1]

                    # print("LinksSol", self.edgemapping)
                else:

                    # print("No solution found")
                    return NodesPlaced, {}, LinksPlaced, {}, bestCost, self.SubNet

            Valid, bestCost, bestSolution = self.localSearch(VNR, constructedSol, constructedSolCost)
            # print(Valid, bestCost, bestSolution)
            if Valid:
                self.nodemapping = bestSolution[0]
                # print("nodemapping:", nodemapping)
                self.edgemapping = bestSolution[1]
                # print("LinksPlacementSol:", LinksPlacementSol)
                upR = UpdateResources(VNR, self.SubNet)
                # print(nodemapping, "----", LinksPlacementSol, "----", edge)
                upR.updateLinksResources(self.edgemapping)
                upR.updateNodesResources(self.nodemapping)

            # return NodesPlaced, self.nodemapping, LinksPlaced, self.edgemapping, bestCost, self.SubNet
            results = {'success': success, 'nodemapping': [], 'edgemapping': [], 'nb_vnfs': len(VNR.G_VNR.nodes), "nb_vls": 0, 'R2C': bestCost,
                       'p_load': 0, 'reward': self.rejection_penalty, 'sn': sb, 'cause': cause, 'nb_iter': None}
            return results
        else:
            # print("Local Search is OFF !")
            # print("lambda =", self.Lambda)
            for i in range(len(VNR.G_VNR.nodes)):

                # print("########## Iteration", i, "#################")
                Valid, constructedSol, constructedSolCost = self.ConstructGreedyRandomizedSolution(VNR)
                # print(Valid, constructedSol, constructedSolCost)
                if Valid and constructedSol:
                    NodesPlaced, LinksPlaced = True, True
                    bestCost = constructedSolCost
                    self.nodemapping = constructedSol[0]
                    # print("NodesSol", self.nodemapping)
                    self.edgemapping = constructedSol[1]
                    upR = UpdateResources(VNR, self.SubNet)
                    # print(nodemapping, "----", LinksPlacementSol, "----", edge)
                    upR.updateLinksResources(self.edgemapping)
                    upR.updateNodesResources(self.nodemapping)
                    # print("LinksSol", self.edgemapping)
                    bestCost = constructedSolCost
                else:

                    # print("No solution found")
                    return NodesPlaced, {}, LinksPlaced, {}, bestCost, self.SubNet
            #return NodesPlaced, self.nodemapping, LinksPlaced, self.edgemapping, bestCost, self.SubNet
            results = {'success': success, 'nodemapping': [], 'edgemapping': [], 'nb_vnfs': 0, "nb_vls": 0, 'R2C': 0,
                       'p_load': 0, 'reward': self.rejection_penalty, 'sn': sb, 'cause': cause, 'nb_iter': None}
            return results

if __name__ == "__main__":
    SEED_VALUE = 30
    LAMBDA_VALUES = [0.5]

    for lamda in LAMBDA_VALUES:
        subNet = SubstrateNetwork()

        totCPU, totRAM = subNet.totalResourcesSystem(subNet.nDetails)
        # print("Total Subnet resources before placement :", totCPU, totRAM, totDisk)
        success = 0
        print("######### GRASP with lambda =", lamda)
        costs = []
        for i in range(10):
            # print("VNR", i)
            virtNet = VirtualNetworkRequest(seed=SEED_VALUE)
            gp = GraspPlacement(subNet, virtNet)
            gp.updateConfig(lamda)
            # print("edges details :", virtNet.getEdgesDetails())
            totCPU, totRAM = virtNet.totalResourcesSystem(virtNet.nDetails)
            # print("Total VirtNet resources :", totCPU, totRAM, totDisk)
            NodesPlaced, nodemapping, LinksPlaced, LinksPlacementSol, subNet = [], [], [], [], None

            NodesPlaced, nodemapping, LinksPlaced, LinksPlacementSol, bestCost, subNet = gp.runPlacement(virtNet)
            # print(NodesPlaced, nodemapping, LinksPlaced, LinksPlacementSol, bestCost)
            # print('bestCost :',bestCost)
            costs.append(bestCost)
            if LinksPlaced and NodesPlaced and bestCost > 0.0:
                success += 1
                totCPU, totRAM = subNet.totalResourcesSystem(subNet.nDetails)

            else:
                print("Failed VNR placement.")

                totCPU, totRAM = subNet.totalResourcesSystem(subNet.nDetails)
                # print("Total Subnet resources After final placement :", totCPU, totRAM, totDisk)
                # break
        print("Success :", success)
        print("Costs :", costs)
        print("non zero elements in costs :", np.count_nonzero(costs))

        print("mean costs :", round(np.mean(costs), 2))