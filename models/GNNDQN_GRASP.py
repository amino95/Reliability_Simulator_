
import sys, os
from pathlib import Path

# Add the parent directory to sys.path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add it to the system path
if current_dir not in sys.path:
    sys.path.append(current_dir)

from observation import Observation
from solver import Solver
from DQNAgent import DQNAgent
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
import numpy as np



class GNNDQN_GRASP(Solver):
    def __init__(self, sigma, rejection_penalty, gamma,
                 learning_rate, memory_size, batch_size, epsilon, eps_min, eps_dec,
                 num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out,
                 num_actions, max_iteration_gp, alpha, max_iteration):
        super().__init__(sigma, rejection_penalty)

        # Agent DRL qui fournit les Q-values pour la heuristique
        self.agent = DQNAgent(gamma, learning_rate, epsilon, memory_size, batch_size,
                              num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out,
                              num_actions, eps_min, eps_dec)
        # Cette variable est lue par solver.py
        # Fixez-la à 1 si vous ne voulez qu'un seul essai,
        # ou plus (ex: 5) pour profiter de l'aspect GRASP
        self.max_iteration = 5
        self.saved_reward = None
        # Indique au solver que cet algorithme supporte les itérations
        self.iteration = True
        self.max_iteration_gp = max_iteration_gp  # Nombre de tentatives GRASP par VNR
        self.alpha = alpha  # Paramètre de relaxation (0 = pur greedy, 1 = pur aléatoire)
        self.saved_transition = None

    # ------------------------------------------------------------------
    #  Phase de Construction : Sélection avec RCL
    # ------------------------------------------------------------------
    def get_rcl_action(self, observation, sn, vnr, vnf_idx):
        """
        Calcule les Q-values via le GNN et retourne une action choisie dans la RCL.
        """
        # 1. Obtenir les Q-values de tous les nœuds substrats (actions)
        q_values = self.agent.get_q_values(observation)
        cpu_req = vnr.vnode[vnf_idx].cpu

        # 2. Filtrer les candidats valides (Contrainte CPU)
        candidates = []
        for i in range(len(sn.snode)):
            if sn.snode[i].lastcpu >= cpu_req:
                candidates.append({'id': i, 'q': q_values[i].item()})

        if not candidates:
            return -1

        # 3. Définir les bornes pour la RCL
        q_max = max(c['q'] for c in candidates)
        q_min = min(c['q'] for c in candidates)

        # Seuil GRASP : q >= q_max - alpha * (q_max - q_min)
        threshold = q_max - self.alpha * (q_max - q_min)

        # 4. Construire la liste restreinte (RCL)
        rcl = [c['id'] for c in candidates if c['q'] >= threshold]

        # 5. Choix aléatoire dans la RCL
        return random.choice(rcl)

    def calculate_r2c(self, vnr):
        revenue = sum(vn.cpu for vn in vnr.vnode) + sum(ve.bandwidth for ve in vnr.vedege)

        cost_cpu = sum(vn.cpu for vn in vnr.vnode)
        cost_bw = 0
        for ve in vnr.vedege:
            # Si spc est vide (même nœud), coût = bandwidth. Sinon, bandwidth * nb_liens
            hops = len(ve.spc) if len(ve.spc) > 0 else 1
            cost_bw += ve.bandwidth * hops

        return revenue / (cost_cpu + cost_bw) if (cost_cpu + cost_bw) > 0 else 0


    def calculate_reliability(self, sn, vnr):
        """
        Calcule la fiabilité globale de la requête virtuelle.
        """
        total_reliability = 1.0

        # 1. Fiabilité des Nœuds
        # On multiplie la fiabilité de chaque nœud physique hébergeant une VNF
        # for vnf_idx, sn_node_id in vnr.nodemapping.items():
        #     # On récupère la fiabilité du nœud physique (ex: 0.99)
        #     node_rel = sn.snode[sn_node_id].rel
        #     total_reliability *= node_rel

        # 2. Fiabilité des Liens (Edges)
        # Un lien virtuel est fiable si TOUS les liens physiques de son chemin (spc) le sont
        for v_edge in vnr.vedege:
            if hasattr(v_edge, 'spc') and v_edge.spc:
                for s_edge_idx in v_edge.spc:
                    edge_rel = sn.sedege[s_edge_idx].rel
                    total_reliability *= edge_rel

        return total_reliability

    def calculate_reward(self, vnr):
        """
        Calcule la récompense en favorisant les solutions à haute fiabilité.
        Revenue = Somme(CPU) + Somme(BW)
        Cost = Somme(CPU_utilisé) + Somme(BW * Longueur_Chemin)
        Reward = Revenue * (Reliability^2) / Cost_Factor
        """
        # 1. Calcul du Revenu (ce que la VNR demande)
        revenue = sum(vn.cpu for vn in vnr.vnode) + sum(ve.bandwidth for ve in vnr.vedege)

        # 2. Calcul du Coût Réel (ce que le substrat consomme réellement)
        # Plus le chemin est long, plus le coût est élevé
        cost_cpu = sum(vn.cpu for vn in vnr.vnode)
        cost_bw = 0
        for ve in vnr.vedege:
            # On utilise la longueur du chemin (spc) déterminée par edegemapping
            hops = len(ve.spc) if len(ve.spc) > 0 else 1
            cost_bw += ve.bandwidth * hops

        total_cost = cost_cpu + cost_bw

        # 3. Récupération de la fiabilité (calculée par votre méthode dédiée)
        # On suppose que self.calculate_reliability(sn, vnr) a été appelée
        # ou on la recalcule ici si nécessaire.
        rel = getattr(vnr, 'reliability', 0.5)  # Valeur par défaut si non trouvé

        # 4. Formule de Récompense Optimisée
        # On utilise rel^2 pour pénaliser fortement les faibles fiabilités
        if total_cost > 0:
            # Le ratio (revenue/cost) encourage l'efficacité spectrale
            # Le multiplicateur rel^2 oriente l'agent vers la survie du service
            reward = (revenue / total_cost) * (rel ** 2) * 10
        else:
            reward = 0

        return reward
    def edegemapping(self, sn, vnr, idx, nodes_mapped, ve2seindex):
        """
        Effectue le mapping des liens virtuels vers les chemins physiques
        en utilisant le plus court chemin (Shortest Path).
        """
        success = True
        # Récupère les voisins du nœud VNR actuel qui sont déjà placés
        neighbors = vnr.vnode[idx].neighbors
        mapped = nodes_mapped
        intersection = np.intersect1d(neighbors, mapped)

        for i in intersection:
            # Assurer l'ordre des indices pour retrouver l'arête dans NetworkX
            s, d = (idx, i) if idx < i else (i, idx)

            edge_list = list(vnr.graph.edges())
            try:
                index = edge_list.index((s, d))
            except ValueError:
                # Sécurité si l'ordre (s,d) n'est pas celui du graphe
                index = edge_list.index((d, s))

            bw_required = vnr.vedege[index].bandwidth
            from_substrate_node = vnr.nodemapping[s]
            to_substrate_node = vnr.nodemapping[d]

            # 1. Générer le graphe résiduel (ne contient que les liens avec assez de BW)
            # Cette fonction doit filtrer les liens physiques incapables de supporter bw_required
            g_residual = self.Sn2_networkxG(sn.snode, sn.sedege, bw_required)

            # 2. Calcul du plus court chemin
            path_indices, cost = self.shortpath(g_residual, from_substrate_node, to_substrate_node, weight=None)

            # Si aucun chemin n'est trouvé, le mapping de la VNR échoue
            if not path_indices:
                return False

            # 3. Mise à jour des ressources physiques (Substrate)
            for j in path_indices:
                # Réduction de la bande passante résiduelle sur le lien physique
                sn.sedege[j].lastbandwidth -= bw_required

                # Enregistrement du mapping pour la libération future des ressources
                sn.sedege[j].vedegeindexs.append([vnr.id, index])

                # Mise à jour de la BW totale sur les nœuds adjacents (souvent utilisé par le GNN)
                node_indices = sn.sedege[j].nodeindex
                sn.snode[node_indices[0]].lastbw -= bw_required
                sn.snode[node_indices[1]].lastbw -= bw_required

            # 4. Enregistrement du chemin dans la requête (VNR)
            vnr.vedege[index].spc = path_indices
            ve2seindex[index] = path_indices

        return success

    def choose_action_grasp(self, observation, vnf_cpu, sn_last_cpus, candidates_limit=3):
        # 1. Obtenir les scores (Q-values) depuis le réseau de politique
        with torch.no_grad():
            # Dans DQNAgent, le réseau s'appelle généralement policy_net
            # On utilise forward_batch car le modèle attend une liste d'observations
            try:
                # Tentative avec policy_net (standard Klesaux)
                q_values = self.agent.policy_net.forward_batch([observation])
            except AttributeError:
                # Si policy_net n'existe pas, on essaie 'gnn' ou 'model'
                # Ceci permet une compatibilité si vous avez changé le nom
                model = getattr(self.agent, 'policy_net', getattr(self.agent, 'model', None))
                if model is None:
                    raise AttributeError("L'agent n'a ni 'policy_net' ni 'model'. Vérifiez DQNAgent.__init__")
                q_values = model.forward_batch([observation])

            # On extrait les valeurs et on s'assure qu'elles sont sur le CPU pour numpy/random
            q_values = q_values.flatten().cpu()

        # 2. Filtrer les nœuds physiques capables d'accueillir la VNF
        feasible_indices = [
            i for i, cpu in enumerate(sn_last_cpus)
            if cpu >= vnf_cpu
        ]

        if not feasible_indices:
            return -1  # Échec : aucun nœud n'a assez de CPU

        # 3. Extraire les Q-values des nœuds éligibles
        feasible_q_values = q_values[feasible_indices]

        # 4. Construire la RCL (Restricted Candidate List)
        k = min(candidates_limit, len(feasible_indices))
        _, top_k_local_indices = torch.topk(feasible_q_values, k)

        # Mappe les indices locaux vers les indices réels du substrat
        rcl = [feasible_indices[idx] for idx in top_k_local_indices.tolist()]

        # 5. Sélection aléatoire (GRASP)
        return random.choice(rcl)

    def prepare_final_result(self, sn, vnr, ve2seindex, current_iter, success):
        if success:
            # On remplit les données nécessaires
            for i in range(len(vnr.vedege)):
                vnr.vedege[i].spc = ve2seindex[i]

            rel = self.calculate_reliability(sn, vnr)
            vnr.reliability = rel

            return {
                'sn': sn,
                'success': True,
                'reward': self.calculate_reward(vnr),
                'reliability': rel,
                'R2C': self.calculate_r2c(vnr),
                'nodemapping': vnr.nodemapping,
                'edgemapping': {i: ve2seindex[i] for i in range(len(vnr.vedege))},
                'nb_iter': current_iter,
                'nb_vnfs': len(vnr.vnode), 'nb_vls': len(vnr.vedege),
                'p_load': sum(n.p_load for n in sn.snode) / len(sn.snode),
                'cause': None
            }
        else:
            # --- NETTOYAGE ET RETOUR D'ÉCHEC SÉCURISÉ ---
            for node in sn.snode:
                node.vnodeindexs = [item for item in node.vnodeindexs if item[0] != vnr.id]

            return {
                'sn': sn,
                'success': False,
                'cause': 'mapping_failed',
                'reward': 0,
                'reliability': 0,
                'R2C': 0,
                'nb_vnfs': len(vnr.vnode),
                'nb_vls': len(vnr.vedege),
                'nb_iter': current_iter,
                # AJOUTEZ CES DEUX LIGNES POUR LE CONTROLLER :
                'nodemapping': {},
                'edgemapping': {}
            }
    # ------------------------------------------------------------------
    #  Mapping Global (Multi-itérations)
    # --
    def mapping(self, sn, vnr, vnrss=None):
        """
        Mapping complet avec GRASP, Snapshot transactionnel et synchronisation SPC/CPU.
        """
        # --- 1. SNAPSHOT INITIAL (Sauvegarde de l'état avant toute modification) ---
        snapshot_node_cpu = [node.lastcpu for node in sn.snode]
        snapshot_node_bw = [node.lastbw for node in sn.snode]
        snapshot_edge_bw = [edge.lastbandwidth for edge in sn.sedege]

        # --- 2. BOUCLE D'ITÉRATIONS GRASP ---
        for current_iter in range(1, self.max_iteration + 1):

            # --- RESET TOTAL DE L'ITÉRATION ---
            # Nettoyage des Nœuds physiques
            for i, node in enumerate(sn.snode):
                node.lastcpu = snapshot_node_cpu[i]
                node.lastbw = snapshot_node_bw[i]
                node.vnodeindexs = [item for item in node.vnodeindexs if item[0] != vnr.id]

            # Nettoyage des Liens physiques (Corrige "bad edege palcement")
            for j, edge in enumerate(sn.sedege):
                edge.lastbandwidth = snapshot_edge_bw[j]
                edge.vedegeindexs = [item for item in edge.vedegeindexs if item[0] != vnr.id]

            # Reset des attributs de la VNR
            vnr.nodemapping = {}
            for vn in vnr.vnode:
                vn.sn_host = -1
            for ve in vnr.vedege:
                ve.spc = []

            # --- 3. PLACEMENT DES NOEUDS (VNFs) ---
            node_success = True
            for vnf_idx in range(len(vnr.vnode)):
                obs = Observation(sn, vnr, vnf_idx, vnr.nodemapping)

                # Appel GRASP (vérifiez que sn_last_cpus est passé)
                sn_node_id = self.choose_action_grasp(obs, vnr.vnode[vnf_idx].cpu, [n.lastcpu for n in sn.snode])

                if sn_node_id == -1:
                    node_success = False
                    break

                # Allocation physique
                s_node = sn.snode[sn_node_id]
                v_node = vnr.vnode[vnf_idx]

                s_node.lastcpu -= v_node.cpu
                s_node.vnodeindexs.append([vnr.id, vnf_idx])

                # Enregistrement logique
                vnr.nodemapping[vnf_idx] = sn_node_id
                v_node.sn_host = sn_node_id

            if not node_success:
                continue  # Échec nœud, itération suivante

            # --- 4. PLACEMENT DES LIENS (EDGES) ---
            # Utilisation de la méthode de la classe mère
            success_edge, ve2seindex = super().edegemapping(sn.snode, sn.sedege, vnr, vnr.nodemapping)

            if success_edge:
                # --- SUCCÈS TOTAL DE L'ITÉRATION ---

                # 1. Synchronisation SPC pour le test d'intégration
                for i in range(len(vnr.vedege)):
                    vnr.vedege[i].spc = ve2seindex[i]

                # 2. SYNCHRONISATION lastbw DES NOEUDS (Correction de l'erreur)
                # On recalcule snode.lastbw basé sur la nouvelle valeur des liens connectés
                for n in sn.snode:
                    total_link_bw = 0
                    for e in sn.sedege:
                        if n.index in e.nodeindex:
                            total_link_bw += e.lastbandwidth
                    n.lastbw = total_link_bw  # On force la cohérence demandée par le test

                # 3. Mise à jour du p_load
                for i, snode in enumerate(sn.snode):
                    relevant_vnfs = [item[1] for item in snode.vnodeindexs if item[0] == vnr.id]
                    for vnf_idx in relevant_vnfs:
                        v_node = vnr.vnode[vnf_idx]
                        snode.p_load = (v_node.p_maxCpu + snode.p_load * snode.cpu) / snode.cpu

                return self.prepare_final_result(sn, vnr, ve2seindex, current_iter, success=True)
        # --- 5. ÉCHEC FINAL ---
        for i, node in enumerate(sn.snode):
            node.lastcpu = snapshot_node_cpu[i]
            node.lastbw = snapshot_node_bw[i]  # On restaure la valeur initiale ici aussi
            node.vnodeindexs = [item for item in node.vnodeindexs if item[0] != vnr.id]

        for j, edge in enumerate(sn.sedege):
            edge.lastbandwidth = snapshot_edge_bw[j]
            edge.vedegeindexs = [item for item in edge.vedegeindexs if item[0] != vnr.id]

        return self.prepare_final_result(sn, vnr, None, self.max_iteration, success=False)