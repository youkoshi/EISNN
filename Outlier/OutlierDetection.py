import numpy as np

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy import stats
from scipy.special import expit  

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.svm import OneClassSVM

import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches

import joblib

import re
import os
from loguru import logger

from collections import defaultdict


## DTW Calculation
def dtw_calculation_r(s1, s2):
    '''==================================================
        Calculate DTW for two sequence (1D) in reverse order
        Parameter: 
            s1: 1D complex sequence 1
            s2: 1D complex sequence 2
        Returen:
            dtw_matrix: DTW distance matrix (n x n)
            dtw_sequence: DTW Path (point by point)
        Note:
            * Sequence can be 1D complex sequence
        ==================================================
    '''
    len_s1, len_s2 = np.shape(s1)[0], np.shape(s2)[0]
    dtw_matrix = np.full((len_s1 + 1, len_s2 + 1), np.inf)
    dtw_trace = np.zeros((len_s1 + 1, len_s2 + 1, 2))
    dtw_matrix[len_s1, len_s1] = 0
    for i in reversed(range(0, len_s1)):
        for j in reversed(range(0, len_s2)):
            cost = np.abs(s1[i] - s2[j])
            penalty = [ dtw_matrix[i + 1, j],
                        dtw_matrix[i, j + 1],
                        dtw_matrix[i + 1, j + 1]]
            i_p = np.argmin(penalty)
            dtw_matrix[i, j] = cost + penalty[i_p]
            if i_p == 0: dtw_trace[i, j] = [i + 1, j]  
            elif i_p==1: dtw_trace[i, j] = [i, j + 1]
            elif i_p==2: dtw_trace[i, j] = [i + 1, j + 1]
    dtw_sequence = []
    i,j = 0, 0
    while  i!=len_s1-1 and j!=len_s1-1:
        dtw_sequence.append([i, j]) 
        i,j = int(dtw_trace[i, j, 0]), int(dtw_trace[i, j, 1])


    return [dtw_matrix[:-1,:-1], np.array(dtw_sequence)]


def DTWPairwase(data):
    '''==================================================
        Calculate DTW for n sequence with length m
        Parameter: 
            data: n x m Data, can be complex 2D Matrix
        Returen:
            dtw_dist_value: DTW distance matrix (n x n)
            dtw_dist_trace: DTW Path (k x 2)
        Note:
            * Sequence can be complex matrix
        ==================================================
    '''
    num_samples = np.shape(data)[0]
    # num_data = np.shape(data)[1]
    dtw_dist_value = np.zeros((num_samples,num_samples))
    # dtw_dist_matrix = defaultdict(lambda: defaultdict(list))
    dtw_dist_trace = defaultdict(lambda: defaultdict(list))

    for i in range(num_samples):
        # dtw_dist_matrix[i][i] = np.zeros((np.shape(data[i])[0],np.shape(data[i])[0]))
        dtw_dist_value[i,i] = 0
        for j in range(i + 1, num_samples):
            distance, dtw_sequence = dtw_calculation_r(data[i], data[j])
            # dtw_dist_matrix[i][j] = distance
            # dtw_dist_matrix[j][i] = distance
            dtw_dist_trace[i][j] = [dtw_sequence[:,0],dtw_sequence[:,1]]
            dtw_dist_trace[j][i] = [dtw_sequence[:,1],dtw_sequence[:,0]]

            ## Reverse Order DTW
            dtw_dist_value[i,j] = distance[0,0]
            dtw_dist_value[j,i] = distance[0,0]

            ## Normal Order DTW
            # dtw_dist_value[i,j] = distance[-1,-1]
            # dtw_dist_value[j,i] = distance[-1,-1]
    return dtw_dist_value, dtw_dist_trace

## DTW Phase Space
def dtw_phase_extract(dtw_trace_inst):
    '''==================================================
        Extract Phase from DTW Path
        Parameter: 
            dtw_trace_inst: DTW Path (k x 2)
        Returen:
            dtw_phz_norm: DTW Path Phase in [-1,1]
        Note:
            * Reverse order DTW based Phase Calculation
        ==================================================
    '''
    _x = dtw_trace_inst[0]
    _y = dtw_trace_inst[1]

    _x_diff = np.diff(_x[:])
    _y_diff = np.diff(_y[:])

    phz_scale = _x_diff + _y_diff
    phz_value = np.array(_x[1:] - _y[1:])

    dtw_phz_norm = np.sum(phz_scale * phz_value) / (_x[-1]**2)      # _x[-1] & _y[-1] = len(curve)-1

    return dtw_phz_norm

def DTWManifoldGeneration(dtw_dist_value, dtw_dist_trace):
    '''==================================================
        Enbedding ordering in DTW Manifold 
        Parameter: 
            dtw_dist_value: DTW distance matrix (n x n)
            dtw_dist_trace: DTW Path (k x 2)
        Returen:
            dtw_manifold_vec: (sequence) i->j vector in complex number (n x n)
            dtw_manifold_dist: (manifold) i->j distance (n x n)
        Note:
            * dtw_manifold_vec[i,j] = - dtw_manifold_vec[j,i]
            * dtw_manifold_dist[i,j] = dtw_manifold_dist[j,i]
        ==================================================
    '''
    num_samples = np.shape(dtw_dist_value)[0]
    dtw_manifold_imag = np.zeros((num_samples,num_samples,2))
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            _amp = dtw_dist_value[i, j]
            _phz = dtw_phase_extract(dtw_dist_trace[i][j])
            dtw_manifold_imag[i, j] = [_amp,_phz]
            dtw_manifold_imag[j, i] = [-_amp,_phz]  # Flip 

    dtw_manifold_vec = dtw_manifold_imag[:,:,0] * np.exp(-1j * dtw_manifold_imag[:,:,1] * np.pi / 2)

    dtw_manifold_dist = np.zeros((num_samples,num_samples))
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            dtw_manifold_dist[i,j] = np.sum(np.abs(dtw_manifold_vec[i,:] - dtw_manifold_vec[j,:]))
            dtw_manifold_dist[j,i] = dtw_manifold_dist[i,j]

    return dtw_manifold_vec, dtw_manifold_dist


## DTW Outlier Detection
class DTW_TREE:
    def __init__(self, id, value = np.inf):
        self.id = id
        self.value = value
        self.l = None
        self.r = None
        self.p = None

    def display(self):
        lines, *_ = self._display_aux()
        for line in lines: print(line)
    def _display_aux(self):
        if self.l is None and self.r is None:
            # _s = f"{self.id:03d}:{self.value:03d}"
            _s = f"{self.id:03d}"
            _w = len(_s)
            _h = 1
            _m = _w//2
            return [_s], _w, _h, _m
        elif self.r is None:
            _sr, _wr, _hr, _mr = self.l._display_aux()
            # _s = f"{self.id:03d}:{self.value:03d}"
            _s = f"{self.id:03d}"
            _w = len(_s)
            
            _line_0 = (_mr + 1) * ' ' + (_wr - _mr - 1) * '_' + _s
            _line_1 = _mr * ' ' + '/' + (_wr - _mr - 1 + _w) * ' '
            _lins_s = [line + _w * ' ' for line in _sr]
            return [_line_0, _line_1] + _lins_s, _wr + _w, _hr + 2, _wr + _w // 2
        elif self.l is None:
            _sl, _wl, _hl, _ml = self.r._display_aux()
            # _s = f"{self.id:03d}:{self.value:03d}"
            _s = f"{self.id:03d}"
            _w = len(_s)
            _line_0 = _s + _ml * '_' + (_wl - _ml) * ' '
            _line_1 = (_w + _ml) * ' ' + '\\' + (_wl - _ml - 1) * ' '
            _lins_s = [_w * ' ' + line for line in _sl]
            return [_line_0, _line_1] + _lins_s, _wl + _w, _hl + 2, _w // 2
        else:
            _sl, _wl, _hl, _ml = self.l._display_aux()
            _sr, _wr, _hr, _mr = self.r._display_aux()
            # _s = f"{self.id:03d}:{self.value:03d}"
            _s = f"{self.id:03d}"
            _w = len(_s)
            first_line = (_ml + 1) * ' ' + (_wl - _ml - 1) * '_' + _s + _mr * '_' + (_wr - _mr) * ' '
            second_line = _ml * ' ' + '/' + (_wl - _ml - 1 + _w + _mr) * ' ' + '\\' + (_wr - _mr - 1) * ' '
            if _hl < _hr:
                _sl += [_wl * ' '] * (_hr - _hl)
            elif _hr < _hl:
                _sr += [_wr * ' '] * (_hl - _hr)
            zipped_lines = zip(_sl, _sr)
            lines = [first_line, second_line] + [a + _w * ' ' + b for a, b in zipped_lines]
            return lines, _wl + _wr + _w, max(_hl, _hr) + 2, _wl + _w // 2


def dtw_tree_merge(dtw_tree_A, dtw_tree_B, id):
    '''==================================================
        Merge dtw tree by order
       ==================================================
    '''
    if dtw_tree_A.value < dtw_tree_B.value:
        dtw_tree_root = DTW_TREE(id, dtw_tree_A.value)
        dtw_tree_root.l = dtw_tree_A
        dtw_tree_root.r = dtw_tree_B
    else:
        dtw_tree_root = DTW_TREE(id, dtw_tree_B.value)
        dtw_tree_root.l = dtw_tree_B
        dtw_tree_root.r = dtw_tree_A
    dtw_tree_A.p = dtw_tree_root
    dtw_tree_B.p = dtw_tree_root
    return dtw_tree_root


def dtw_leaf_ordering(dtw_tree):
    '''==================================================
        Returen lead order
       ==================================================
    '''
    if dtw_tree is None: return []
    if dtw_tree.l is None and dtw_tree.r is None: return [dtw_tree.value]
    return dtw_leaf_ordering(dtw_tree.l) + dtw_leaf_ordering(dtw_tree.r)


def DTWOrderingTreeBuild(dtw_manifold_dist):
    '''==================================================
        Build DTW Tree based on linkage
        Parameter: 
            dtw_manifold_dist: (manifold) i->j distance (n x n)
        Returen:
            dtw_node_list:order list of manifold sequence (n x 1)
        Note:
            * None of outlier has been removed here
        ==================================================
    '''
    num_samples = np.shape(dtw_manifold_dist)[0]
    # logger.warning(f"{num_samples}")
    _Z = linkage(squareform(dtw_manifold_dist), method='single', optimal_ordering=True)  # 使用 Ward 方式进行层次聚类
    dtw_node_list = []
    for i in range(num_samples):
        dtw_node_list.append(DTW_TREE(i,i))

    _node_cnt = num_samples 
    for i in _Z:
        dtw_node_list.append(dtw_tree_merge(dtw_node_list[int(i[0])], dtw_node_list[int(i[1])],_node_cnt))
        _node_cnt = _node_cnt + 1

    return dtw_node_list

def all_longest_increasing_subsequences(seq):
    '''==================================================
        Returen all longest increasing subseq in the same length
       ==================================================
    '''
    if not seq:
        return []
    
    n = len(seq)
    dp = [1] * n
    prev = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i):
            if seq[j] < seq[i]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    prev[i] = [j]
                elif dp[j] + 1 == dp[i]:
                    prev[i].append(j)
    
    max_len = max(dp)
    end_indices = [i for i in range(n) if dp[i] == max_len]
    
    def backtrack(i):
        if not prev[i]:
            return [[seq[i]]]
        res = []
        for j in prev[i]:
            for subseq in backtrack(j):
                res.append(subseq + [seq[i]])
        return res

    result = []
    for i in end_indices:
        result.extend(backtrack(i))
    
    return result


def longest_sequence_dist(dist_matrix, sequence):
    '''==================================================
        Calculate spacing with in the sequence
       ==================================================
    '''
    _dist = 0
    for i in range(1,len(sequence)):
        _dist = _dist + dist_matrix[i-1,i]
    return _dist

def DTWTreeAnomalyDetection(dtw_manifold_dist, dtw_node_list):
    '''==================================================
        Anomaly Detection Based on DTW Order Tree
        Parameter: 
            dtw_manifold_dist: (manifold) i->j distance (n x n)
            dtw_node_list:order list of manifold sequence (n x 1)
        Returen:
            leaf_full: (nx1) Full manifold sequence (not in order)
            leaf_optimal_seq: (nnx1) manifold sequence without obvious anomaly (in order)
        Note:
            * Obvious outliers are removed here
        ==================================================
    '''
    leaf_full = dtw_leaf_ordering(dtw_node_list[-1])
    # Find All longest_increasing_subsequences
    leaf_ordering = all_longest_increasing_subsequences(leaf_full)

    optimal_seq_dist = longest_sequence_dist(dtw_manifold_dist, leaf_ordering[0])
    optimal_seq_id = 0
    for i in range(1, len(leaf_ordering)):
        _dist = longest_sequence_dist(dtw_manifold_dist, leaf_ordering[i])
        if _dist < optimal_seq_dist:
            optimal_seq_dist = _dist
            optimal_seq_id = i

    leaf_optimal_seq = np.array(leaf_ordering[optimal_seq_id])
    
    # leaf_optimal_len = np.shape(leaf_optimal_seq)[0]
    # leaf_anomaly = np.array([poi for poi in leaf_full if poi not in leaf_optimal_seq])
    # logger.info(f"\n{leaf_full}\n{leaf_ordering}\n{leaf_optimal_seq}\n{leaf_anomaly}")

    return leaf_full, leaf_optimal_seq

## DTW Transient Detection

def DTWManifoldClusterAP(dtw_manifold_vec, leaf_optimal_seq):
    '''==================================================
        Cluster based on manifold & rough-cleaned sequence
        Parameter: 
            dtw_manifold_vec: (sequence) i->j vector in complex number (n x n)
            leaf_optimal_seq: (nnx1) manifold sequence without obvious anomaly (in order)
        Returen:
            dtw_seq_vec: dtw_manifold_vec after rough-cleaning, 
                         stack real & imag part (nn x 2nn)
            dtw_cluster: cluster result of rough-cleaned sequence
        Note:
            * Obvious outliers are removed here
        ==================================================
    '''
    leaf_optimal_len = np.shape(leaf_optimal_seq)[0]
    
    dtw_seq_vec = np.zeros((leaf_optimal_len, leaf_optimal_len*2))
    for i in range(leaf_optimal_len):
        dtw_seq_vec[i,:] = np.hstack((dtw_manifold_vec[leaf_optimal_seq[i],leaf_optimal_seq].real, dtw_manifold_vec[leaf_optimal_seq[i],leaf_optimal_seq].imag))

    best_model = None
    best_score = -np.inf
    for seed in [7,42,1999]:
        aff_prop = AffinityPropagation()
        aff_prop.fit(dtw_seq_vec)
        _score = silhouette_score(dtw_seq_vec, aff_prop.labels_)
        if _score > best_score:
            best_score = _score
            best_model = aff_prop.labels_

    dtw_cluster = best_model


    # Increacing Sequence
    def errro_cluster_remover(dtw_cluster, leaf_optimal_seq):
        A = np.array(leaf_optimal_seq)
        B = np.array(dtw_cluster)
        
        C = np.append(np.diff(B),0)
        mask = C < 0
        bad_classes = np.unique(B[mask])

        keep_mask = ~np.isin(B, bad_classes)

        A_filtered = A[keep_mask]
        B_filtered = B[keep_mask]

        unique_vals = np.unique(B_filtered)
        new_labels = np.arange(len(unique_vals))
        mapping = dict(zip(unique_vals, new_labels))

        B_remapped = np.vectorize(mapping.get)(B_filtered)

        return A_filtered,B_remapped

    leaf_optimal_seq, dtw_cluster = errro_cluster_remover(dtw_cluster, leaf_optimal_seq)
    leaf_optimal_len = np.shape(leaf_optimal_seq)[0]
    n_clusters = len(np.unique(dtw_cluster))

    dtw_seq_vec = np.zeros((leaf_optimal_len, leaf_optimal_len*2))
    for i in range(leaf_optimal_len):
        dtw_seq_vec[i,:] = np.hstack((dtw_manifold_vec[leaf_optimal_seq[i],leaf_optimal_seq].real, dtw_manifold_vec[leaf_optimal_seq[i],leaf_optimal_seq].imag))



    # n_clusters = len(np.unique(dtw_cluster))
    # logger.info(f"\n#Cluster: {n_clusters}\n Cluster: {dtw_cluster}")
    return dtw_seq_vec, dtw_cluster, leaf_optimal_seq


def DTWManifoldClusterOPTICS(dtw_manifold_vec, leaf_optimal_seq, xi_default = 0.01):
    '''==================================================
        Cluster based on manifold & rough-cleaned sequence
        Parameter: 
            dtw_manifold_vec: (sequence) i->j vector in complex number (n x n)
            leaf_optimal_seq: (nnx1) manifold sequence without obvious anomaly (in order)
        Returen:
            dtw_seq_vec: dtw_manifold_vec after rough-cleaning, 
                         stack real & imag part (nn x 2nn)
            dtw_cluster: cluster result of rough-cleaned sequence
        Note:
            * Obvious outliers are removed here
        ==================================================
    '''
    leaf_optimal_len = np.shape(leaf_optimal_seq)[0]
    
    dtw_seq_vec = np.zeros((leaf_optimal_len, leaf_optimal_len*2))
    for i in range(leaf_optimal_len):
        dtw_seq_vec[i,:] = np.hstack((dtw_manifold_vec[leaf_optimal_seq[i],leaf_optimal_seq].real, dtw_manifold_vec[leaf_optimal_seq[i],leaf_optimal_seq].imag))

    optics = OPTICS(
        min_samples=2,        # 核心点的最小邻居数，默认5
        xi=xi_default,              # 用于提取簇的最小斜率，默认0.05
        metric='euclidean',   # 在特征空间上用欧氏距离
    )
    optics.fit(dtw_seq_vec)
    labels_opt = optics.labels_   # 噪声点标记为 -1

    dtw_cluster = labels_opt


    # Increacing Sequence
    def errro_cluster_remover(dtw_cluster, leaf_optimal_seq):
        A = np.array(leaf_optimal_seq)
        B = np.array(dtw_cluster)
        
        # Remove -1
        A = A[B != -1]
        B = B[B != -1]

        # Maintain ascending sereis
        C = np.append(np.diff(B),0)

        mask = C < 0
        bad_classes = np.unique(B[mask])

        keep_mask = ~np.isin(B, bad_classes)

        A_filtered = A[keep_mask]
        B_filtered = B[keep_mask]

        unique_vals = np.unique(B_filtered)
        new_labels = np.arange(len(unique_vals))
        mapping = dict(zip(unique_vals, new_labels))

        B_remapped = np.vectorize(mapping.get)(B_filtered)

        return A_filtered,B_remapped

    leaf_optimal_seq, dtw_cluster = errro_cluster_remover(dtw_cluster, leaf_optimal_seq)
    leaf_optimal_len = np.shape(leaf_optimal_seq)[0]
    n_clusters = len(np.unique(dtw_cluster))

    dtw_seq_vec = np.zeros((leaf_optimal_len, leaf_optimal_len*2))
    for i in range(leaf_optimal_len):
        dtw_seq_vec[i,:] = np.hstack((dtw_manifold_vec[leaf_optimal_seq[i],leaf_optimal_seq].real, dtw_manifold_vec[leaf_optimal_seq[i],leaf_optimal_seq].imag))



    # n_clusters = len(np.unique(dtw_cluster))
    # logger.info(f"\n#Cluster: {n_clusters}\n Cluster: {dtw_cluster}")
    return dtw_seq_vec, dtw_cluster, leaf_optimal_seq



def unilateral_T_test(data, x):
    n = len(data)
    if n < 2: return -1
    mean = np.mean(data)
    std = np.std(data, ddof=1)      # ddof = 1 for small sample

    # t_stat = (x - mean) / (std)
    t_stat = (x - mean) / (std / np.sqrt(n))
    # H0： x > μ
    p_value = 1 - stats.t.cdf(t_stat, df=n-1)
    return p_value

def DTWClusterMerge(dtw_cluster, dtw_seq_vec, min_point = 2, p_th = 0.005):
    '''==================================================
        Cluster based on manifold & rough-cleaned sequence
        Parameter: 
            dtw_cluster: (nn x 1)cluster result of rough-cleaned sequence
            dtw_seq_vec: (nn x 1)dtw_manifold_vec after rough-cleaning, 
                         stack real & imag part (nn x 2nn)
        Returen:
            dtw_cluster_revision: Cluster after merging
        ==================================================
    '''
    # Validation of cluster
    n_clusters = len(np.unique(dtw_cluster))
    inner_dist = []
    for i in range(n_clusters):
        _cluster = dtw_seq_vec[dtw_cluster[:]==i,:]
        if np.shape(_cluster)[0] < min_point:
            inner_dist.append([0])
        else: 
            inner_dist.append(np.sum(np.diff(dtw_seq_vec[dtw_cluster[:]==i,:],axis=0)**2, axis=1))

    cluster_connect = [] 

    # Closest Criterion
    # for i in range(n_clusters-1):
    #     _pre = dtw_seq_vec[dtw_cluster[:]==i,:][-1]
    #     _poi = dtw_seq_vec[dtw_cluster[:]==i+1,:][0]
    #     intp_dst = np.sum(np.diff([_pre,_poi], axis=0)**2, axis=1)

    #     p_pre = unilateral_T_test(inner_dist[i], intp_dst)
    #     p_poi = unilateral_T_test(inner_dist[i+1], intp_dst)

    #     if np.max([p_pre,p_poi]) > 0.01 and np.min([p_pre,p_poi]) > 0:    # merge
    #         cluster_connect.append(i)


    # Average Criterion
    for i in range(n_clusters-1):
        _pre = dtw_seq_vec[dtw_cluster[:]==i,:][-1]
        _poi = dtw_seq_vec[dtw_cluster[:]==i+1,:][0]
        intp_dst = np.sum(np.diff([_pre,_poi], axis=0)**2, axis=1)
        _p_value = unilateral_T_test(np.hstack((inner_dist[i],inner_dist[i+1])), intp_dst)
        # print(_p_value)
        if _p_value > p_th:
            cluster_connect.append(i)


    # Merge from high to low
    dtw_cluster_revision = np.array(dtw_cluster)
    for i in cluster_connect[::-1]:
        for j in range(np.shape(dtw_cluster_revision)[0]):
            _x = dtw_cluster_revision[j]
            if _x == i + 1:
                dtw_cluster_revision[j] = i
            elif _x > i + 1:
                dtw_cluster_revision[j] = _x - 1


    # n_clusters_revision = len(np.unique(dtw_cluster_revision))
    # logger.info(f"dtw_cluster_revision: {dtw_cluster_revision}")
    return dtw_cluster_revision

def DTWSingleClusterOutlierDetection(leaf_optimal_seq, dtw_cluster_revision):
    '''==================================================
        Cluster based on manifold & rough-cleaned sequence
        Parameter: 
            leaf_optimal_seq: (nn x 1) manifold sequence without obvious anomaly (in order)
            dtw_cluster_revision: (nn x 1)Cluster after merging 
        Returen:
            eis_seq: Cleaned manifold sequence (nnn x 1)
            eis_cluster: Cleaned manifold Cluster (nnn x 1)
        Note:
            * cluster with single point was removed as outlier
        ==================================================
    '''
    eis_seq = np.array(leaf_optimal_seq)
    eis_cluster = np.array(dtw_cluster_revision)
    eis_cluster_n = len(np.unique(eis_cluster))

    for i in reversed(range(len(eis_seq))):
        if len(eis_cluster[eis_cluster == eis_cluster[i]]) == 1:
            eis_cluster[i:] = eis_cluster[i:] - 1
            eis_cluster = np.delete(eis_cluster, i)
            eis_seq = np.delete(eis_seq, i)
            eis_cluster_n = eis_cluster_n-1
        # elif len(eis_cluster[eis_cluster == eis_cluster[i]]) == 2:
        #     if eis_cluster[i] != 0 and eis_cluster[i] != eis_cluster_n-1:
        #         eis_cluster[i:] = eis_cluster[i:] - 1
                
        #         eis_cluster = np.delete(eis_cluster, i)
        #         eis_seq = np.delete(eis_seq, i)
        #         i = i-1
        #         eis_cluster = np.delete(eis_cluster, i)
        #         eis_seq = np.delete(eis_seq, i)
                
        #         eis_cluster_n = eis_cluster_n-1
                
    # eis_anomaly = np.array([poi for poi in leaf_full if poi not in eis_seq])
    # eis_cluster_n = len(np.unique(eis_cluster))

    return eis_seq, eis_cluster

## Cleaned Data Export


## Overall
def OutlierDetection(chData):
    ch_EIS = chData[:,1,:] + 1j*chData[:,2,:]
    data = np.log(ch_EIS)
    dtw_dist_value, dtw_dist_trace = DTWPairwase(data)
    dtw_manifold_vec, dtw_manifold_dist = DTWManifoldGeneration(dtw_dist_value, dtw_dist_trace)
    dtw_node_list = DTWOrderingTreeBuild(dtw_manifold_dist)
    leaf_full, leaf_optimal_seq = DTWTreeAnomalyDetection(dtw_manifold_dist, dtw_node_list)
    dtw_seq_vec, dtw_cluster, eis_seq = DTWManifoldClusterAP(dtw_manifold_vec, leaf_optimal_seq)
    dtw_cluster_revision = DTWClusterMerge(dtw_cluster, dtw_seq_vec)
    eis_seq, eis_cluster = DTWSingleClusterOutlierDetection(eis_seq, dtw_cluster_revision)

    eis_anomaly = np.array([poi for poi in leaf_full if poi not in eis_seq])
    leaf_anomaly = np.array([poi for poi in leaf_full if poi not in leaf_optimal_seq])
    
    return eis_seq.astype(int), eis_cluster.astype(int), eis_anomaly.astype(int), leaf_anomaly.astype(int)











############################################
# Outlierdetection with Open & Short & Wierd
############################################
def weirdCriterion(model:OneClassSVM, test_data, threshold=0.5):
    '''==================================================
        Define the criterion of weird data
        Parameter: 
            model: trained OneClassSVM model
            test_data: data to be tested [n x 101] - (logZ)
            threshold: threshold of weird data
        Returen:
            weird_mask: True for weird data
        ==================================================
    '''

    _data = np.hstack([test_data.real, test_data.imag])

    _scores = model.decision_function(_data) 

    _probs = expit(_scores * 5)

    weird_mask = _probs > threshold

    return weird_mask


def openCriterion(model:OneClassSVM, test_data, threshold=0.5):
    '''==================================================
        Define the criterion of open data
        Parameter: 
            model: trained OneClassSVM model
            test_data: data to be tested [n x 101] - (logZ)
            threshold: threshold of weird data
        Returen:
            open_mask: True for open data
        ==================================================
    '''
    _data = np.hstack([test_data.real, test_data.imag])
    
    _scores = model.decision_function(_data) 

    _probs = expit(_scores * 5)

    open_mask = _probs > threshold

    return open_mask

def shortCriterion(freq, test_data, threshold = np.log(1e4)):
    '''==================================================
        Define the criterion of short data
        Parameter: 
            freq: frequency of EIS data [101,]
            test_data: data to be tested [n x 101] - (logZ)
            threshold: threshold of short data
        Returen:
            short_mask: True for shorted data
        ==================================================
    '''
    _data = np.hstack([test_data.real, test_data.imag])
    _freq_short_mask = np.zeros(_data.shape[1])
    _freq_short_mask[:_freq_short_mask.shape[0]//2] = freq > 1e4
    _freq_short_mask = _freq_short_mask.astype(bool)

    short_mask = np.all(_data[:,_freq_short_mask] < threshold, axis=1)

    return short_mask





############################################
# Overall
############################################
def OutlierDetection_Ver02(chData, weirdModel = None, mask_flag = False):
    freq_list_weird = np.linspace(0,np.shape(chData)[2]-1,101,dtype=int, endpoint=True)
    freq_list_dtw   = np.linspace(1000,np.shape(chData)[2]-1,101,dtype=int, endpoint=True)

    ch_EIS = chData[:,1,:] + 1j*chData[:,2,:]
    data = np.log(ch_EIS)

    if weirdModel is None:
        # weirdModel = joblib.load("./weirdSVMmodel.pkl")
        # weird_mask = weirdCriterion(weirdModel, data[:,freq_list_weird])
        good_data = data[:,:]
        weird_mask = np.zeros(data.shape[0]).astype(bool)
    else:
        weird_mask = weirdCriterion(weirdModel, data[:,freq_list_weird])
        good_data = data[~weird_mask,:]
    
    good_data = good_data[:,freq_list_dtw]
    # logger.error(f"data.shape: {good_data.shape}")
    dtw_dist_value, dtw_dist_trace = DTWPairwase(good_data)
    dtw_manifold_vec, dtw_manifold_dist = DTWManifoldGeneration(dtw_dist_value, dtw_dist_trace)
    dtw_node_list = DTWOrderingTreeBuild(dtw_manifold_dist)
    leaf_full, leaf_optimal_seq = DTWTreeAnomalyDetection(dtw_manifold_dist, dtw_node_list)
    dtw_seq_vec, dtw_cluster, eis_seq = DTWManifoldClusterOPTICS(dtw_manifold_vec, leaf_optimal_seq)
    dtw_cluster_revision = DTWClusterMerge(dtw_cluster, dtw_seq_vec)
    eis_seq, eis_cluster = DTWSingleClusterOutlierDetection(eis_seq, dtw_cluster_revision)

    seq_full = np.arange(data.shape[0])
    seq_good = seq_full[~weird_mask.astype(bool)]
    eis_seq = seq_good[eis_seq.astype(int)]

    seq_weird = seq_full[weird_mask.astype(bool)]

    eis_anomaly = np.array([poi for poi in seq_full if poi not in eis_seq])
    leaf_anomaly = np.array([poi for poi in leaf_full if poi not in leaf_optimal_seq])
    
    leaf_anomaly = seq_good[leaf_anomaly.astype(int)]
    
    if mask_flag:
        return eis_seq.astype(int), eis_cluster.astype(int), eis_anomaly.astype(int), leaf_anomaly.astype(int), weird_mask.astype(bool)
    else:
        return eis_seq.astype(int), eis_cluster.astype(int), eis_anomaly.astype(int), leaf_anomaly.astype(int), seq_weird.astype(int)



def OpenShortDetection(chData, openModel = None, mask_flag = False):
    
    freq_list = np.linspace(0,np.shape(chData)[2]-1,101,dtype=int, endpoint=True)

    ch_EIS = chData[:,1,:] + 1j*chData[:,2,:]
    data = np.log(ch_EIS)

    ## Open Criterion
    if openModel is None:
        openModel = joblib.load("./openSVMmodel.pkl")
    open_mask = openCriterion(openModel, data[:,freq_list])
    
    ## Short Criterion
    _freq_short = chData[0,0,freq_list]
    short_mask = shortCriterion(_freq_short, data[:,freq_list])
    
    if mask_flag:
        return open_mask, short_mask
    else:
        seq_full    = np.arange(data.shape[0])
        seq_open    = seq_full[open_mask]
        seq_short   = seq_full[short_mask]
        return seq_open, seq_short




############################################
# Plot
############################################
def OutlierDetectionPlot(fig, chData, eis_seq, eis_cluster, eis_anomaly, leaf_anomaly, seq_weird, seq_open, seq_short):
    # fig= plt.figure(figsize=(15,8), constrained_layout=False)
    num_samples = chData.shape[0]
    num_cluster = len(np.unique(eis_cluster))

    seq_open_eis    = np.intersect1d(eis_seq, seq_open)
    seq_short_eis   = np.intersect1d(eis_seq, seq_short)


    axis    = [0] * 11
    axis[0] = fig.add_subplot(3,3,1, projection='3d')   # Original 3D
    axis[1] = fig.add_subplot(3,3,2)                    # Original 2D
    axis[2] = fig.add_subplot(3,3,3)                    # Original 2D
    axis[3] = fig.add_subplot(3,3,4, projection='3d')   # Linkage 3D
    axis[4] = fig.add_subplot(3,3,5)                    # Linkage Sequence
    axis[5] = fig.add_subplot(3,3,6)                    # Linkage Anomaly
    axis[7] = fig.add_subplot(3,3,8)                    # AP Sequence
    axis[8] = fig.add_subplot(3,3,9)                    # AP Anomaly
    
    
    axis[6] = fig.add_subplot(3,3,7)   # Text
    text_axis = axis[6]
    text_axis.axis('off')

    init_elev = 21  # 仰角
    init_azim = 55  # 方位角
    axis[0].view_init(elev=init_elev, azim=init_azim)
    axis[3].view_init(elev=init_elev, azim=init_azim)
    # axis[6].view_init(elev=init_elev, azim=init_azim)


    axis[0].set_title("Original")
    axis[3].set_title("Anomaly Detection")
    # axis[6].set_title("Cluster Analysis")

    
    axis[1].set_title("Original Data")
    axis[4].set_title("After Outlier Detection")
    axis[7].set_title("After Cluster")

    
    axis[2].set_title("Type I Outlier")
    axis[5].set_title("Type II Outlier")
    axis[8].set_title("Open-Short")


    ## Original
    _x = np.arange(num_samples)
    _y = np.log10(chData[0,0,:]).flatten()
    X, Y = np.meshgrid(_x, _y, indexing='ij')
    axis[0].plot_surface(X, Y, np.log10(np.abs(chData[:,1,:]+1j*chData[:,2,:])), cmap='viridis_r', alpha=0.8)


    cmap = plt.colormaps.get_cmap('rainbow_r')
    for i in range(num_samples):
        ch_eis = chData[i,:,:]
        _color = cmap(i/num_samples)
        axis[1].loglog(ch_eis[0,:], np.abs(ch_eis[1,:]+1j*ch_eis[2,:]), color = _color, linewidth=2, label=f"S{i:02d}")

    for i in seq_weird:
        ch_eis = chData[i,:,:]
        _color = cmap(i/num_samples)
        axis[2].loglog(ch_eis[0,:], np.abs(ch_eis[1,:]+1j*ch_eis[2,:]), color = _color, linewidth=2, label=f"S{i:02d}")
    if len(seq_weird) != 0:
        axis[2].legend()
        axis[2].sharex(axis[1])
        axis[2].sharey(axis[1])
    else: axis[2].axis('off')

    ## Anomaly Detection

    _x = np.arange(num_samples)[eis_seq]
    _y = np.log10(chData[0,0,:]).flatten()
    X, Y = np.meshgrid(_x, _y, indexing='ij')
    axis[3].plot_surface(X, Y, np.log10(np.abs(chData[eis_seq,1,:]+1j*chData[eis_seq,2,:])), cmap='viridis_r', alpha=0.8)


    cmap = plt.colormaps.get_cmap('rainbow_r')
    for i in eis_seq:
        ch_eis = chData[i,:,:]
        _color = cmap(i/num_samples)
        axis[4].loglog(ch_eis[0,:], np.abs(ch_eis[1,:]+1j*ch_eis[2,:]), color = _color, linewidth=2, label=f"S{i:02d}")
    axis[4].sharex(axis[1])
    axis[4].sharey(axis[1])


    for i in leaf_anomaly:
        ch_eis = chData[i,:,:]
        _color = cmap(i/num_samples)
        axis[5].loglog(ch_eis[0,:], np.abs(ch_eis[1,:]+1j*ch_eis[2,:]), color = _color, linewidth=2, label=f"S{i:02d}")
    if len(leaf_anomaly) != 0:
        axis[5].legend()
        axis[5].sharex(axis[1])
        axis[5].sharey(axis[1])
    else: axis[5].axis('off')


    ## Cluster Analysis

    # _x = np.arange(num_samples)[eis_seq]
    # _y = np.log10(chData[0,0,:]).flatten()
    # X, Y = np.meshgrid(_x, _y, indexing='ij')
    # axis[6].plot_surface(X, Y, np.log10(np.abs(chData[eis_seq,1,:]+1j*chData[eis_seq,2,:])), cmap='viridis_r', alpha=0.8)


    cmap = plt.colormaps.get_cmap('Set1')
    for i in range(len(eis_seq)):
        _x = eis_seq[i]
        ch_eis = chData[_x,:,:]
        _color = cmap(eis_cluster[i])
        axis[7].loglog(ch_eis[0,:], np.abs(ch_eis[1,:]+1j*ch_eis[2,:]), color = _color, linewidth=2, label=f"{chr(ord('A')+eis_cluster[i])}")

    _legend_handle = []
    for i in range(num_cluster):
        _legend_handle.append(mpatches.Patch(color = cmap(i), label = f"{chr(ord('A')+i)}:{len(eis_cluster[eis_cluster==i])}"))
    axis[7].legend(handles=_legend_handle)

    axis[7].sharex(axis[1])
    axis[7].sharey(axis[1])

    # Open Short

    cmap = plt.colormaps.get_cmap('managua')
    for i in range(len(eis_seq)):
        _x = eis_seq[i]
        if _x in seq_open_eis:     
            _color = cmap(0.0)
            alpha = 1
        elif _x in seq_short_eis:   
            _color = cmap(1.0)
            alpha = 1
        else:                       
            _color = cmap(0.5)
            alpha = 0.2
        ch_eis = chData[_x,:,:]
        axis[8].loglog(ch_eis[0,:], np.abs(ch_eis[1,:]+1j*ch_eis[2,:]), color = _color, linewidth=2, alpha = alpha)

    _legend_handle = []
    _legend_handle.append(mpatches.Patch(color = cmap(0.5), label = f"Norm:{len(eis_seq) - len(seq_open_eis) - len(seq_short_eis)}"))
    _legend_handle.append(mpatches.Patch(color = cmap(0.0), label = f"Open:{len(seq_open_eis)}"))
    _legend_handle.append(mpatches.Patch(color = cmap(1.0), label = f"Short:{len(seq_short_eis)}"))
    axis[8].legend(handles=_legend_handle)
    axis[8].sharex(axis[1])
    axis[8].sharey(axis[1])

    return text_axis