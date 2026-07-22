import random
from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
from scipy import sparse as sp

@dataclass
class SpectralConfig:
    """算法控制参数结构体"""
    top_n: int = 10                   # 指定优化前多少个最大的特征值
    max_iters: int = 500              # 最大迭代次数
    sample_ratio: float = 0.3         # 候选池抽样比例
    add_ratio: float = 0.1            # 每次加入的 edge batch 比例
    target_density_ratio: float = 0.7 # 目标密度比例


def generate_spd_diagonal_decay_target(dim, decay_rate=0.09):
    """生成完全稠密的对称正定（SPD）目标矩阵，并加偏置确保严格正定"""
    W_target = np.zeros((dim, dim))
    np.random.seed(42)
    for i in range(dim):
        for j in range(i + 1, dim):
            distance = abs(i - j)
            weight = np.exp(-decay_rate * distance) * 0.1*np.random.randn()
            weight = max(1e-12, weight)
            W_target[i, j] = weight
            W_target[j, i] = weight
    np.fill_diagonal(W_target, np.sum(W_target, axis=1) + 1.0)
    return W_target

class SpectralSparsification:
    def __init__(self) -> None:
        self.W_target: np.ndarray = np.array(0)
        self.W_backbone: sp.csc_matrix = sp.csc_matrix(0)
        self.config:SpectralConfig = SpectralConfig()
        #
        self.target_evals_top_n:np.ndarray = np.array(0)
        #
        random.seed(42)
    
    def set_graph(self, W_target: np.ndarray):
        self.W_target = W_target
        
    def set_config(self, config: SpectralConfig):
        self.config = config

    def generate_backbone(self):
        """
        骨架初始化
        """
        dim = self.W_target.shape[0]
        
         # 1. Extract the +/- 1 sub-diagonal elements from W_target
        sub_diag_weights = np.diagonal(self.W_target, offset=1)
        
        # 2. Reconstruct indices for the upper and lower sub-diagonals
        idx_u = np.arange(0, dim - 1)
        idx_v = np.arange(1, dim)
        
        # Concatenate to ensure perfect symmetry
        rows = np.concatenate([idx_u, idx_v])
        cols = np.concatenate([idx_v, idx_u])
        data = np.concatenate([sub_diag_weights, sub_diag_weights])
        
        # 3. Initialize directly into CSC format via COO-style constructor assembly
        W_base = sp.csc_matrix((data, (rows, cols)), shape=(dim, dim))
        
        # 4. Inject the full target diagonal to guarantee strict diagonal dominance (SPD)
        W_base.setdiag(np.diagonal(self.W_target))
        self.W_backbone = W_base

    def spectral_optimize(self) -> sp.csc_matrix:
        """
        通过贪心微扰策略，恢复 W_target 权重到 sparse W_base 使前 n 个最大特征值的误差平方和最小。
        所有的调节参数通过 config 结构体传入。
        """
        WEIGHT_MIN = 1e-20
        dim = self.W_backbone.shape[0] # type: ignore
        current_W = self.W_backbone.copy()
        
        # 1. 初始化候选边池
        tri_idx = np.triu_indices(dim, k=1)
        tri_flat_indices = tri_idx[0] * dim + tri_idx[1] # flat to 1 dim
        
        W_base_upper_coo = sp.triu(self.W_backbone, k=1).tocoo()
        base_flat_indices = W_base_upper_coo.row * dim + W_base_upper_coo.col
        is_already_in_base = np.isin(tri_flat_indices, base_flat_indices)
        # 获取 坐标 mask
        mask = (self.W_target[tri_idx] > WEIGHT_MIN) & (~is_already_in_base)
        c_u = tri_idx[0][mask]
        c_v = tri_idx[1][mask]
        all_candidates = set(zip(c_u, c_v))
        
        print(f"初始候选池 {len(all_candidates)}, Wtarget nnz/2 {np.count_nonzero(self.W_target)/2}") 
                    
        # 2. 提前计算目标谱，并提取出前 n 个最大的特征值
        #target_evals_all, target_evecs = la.eigh(self.W_target)
        #self.target_evals_top_n = target_evals_all[-self.config.top_n:] # 升序矩阵中，最后 n 个是最大的
        
        Reff = self._compute_effective_resistances(self.W_target)
        sort_indices = np.argsort(-Reff, axis=None)[:self.config.top_n]
        self.target_evals_top_n = Reff.flat[sort_indices]
        # 特征值的初始搜索向量
        orginal_ratio = self.config.add_ratio
        warm_start_v0 = np.ones(dim)
        best_loss = np.inf
        for iteration in range(self.config.max_iters):
            curr_total = current_W.nnz
            curr_ratio = curr_total / (dim * dim)
            
            print(f"迭代 {iteration}: 当前非零元 {curr_total}, 密度 {curr_ratio:.3f}, "
                  f"目标密度 {self.config.target_density_ratio:.3f}")
            if curr_ratio >= self.config.target_density_ratio or not all_candidates:
                print(" -> 满足终止条件，退出循环, 候选池剩余数量", len(all_candidates))
                break
                
            # 从候选池进行随机采样，均匀采样
            sample_size = max(1, int(len(all_candidates) * self.config.sample_ratio))
            sampled_candidates = random.sample(list(all_candidates), sample_size)
            
            # 矩阵一阶微扰动快速评估谱收益
            # 评估采样边
            U_idx = np.array([u for u, v in sampled_candidates], dtype=np.int32)
            V_idx = np.array([v for u, v in sampled_candidates], dtype=np.int32)
            #gains_spectral = self._calculate_spectral_gains(current_W, (U_idx, V_idx), warm_start_v0, add_sign=1.0)
            gains_spectral = self._calculate_Reffective_gains(current_W, (U_idx, V_idx))
            # 评估 fill—in 收益
            node_degree, node_common = self._calcualte_edge_degree(current_W, U_idx, V_idx)

            # 计算得分，鼓励谱收益和聚团，惩罚度连接
            scores = gains_spectral *(1+ node_common) #/(node_degree + 1)
            
            print(" scores > 0 counts: ", np.sum(scores >0))

            # 批量恢复 Top-N 收益边 (使用来自 config 的 add_ratio)
            if len(scores) > 0:
                current_W, isfinished = self._add_edges(current_W, scores, (U_idx, V_idx), all_candidates)
                if isfinished:
                    current_W = self._refine_weights_by_scale(current_W)
                    return current_W
        
        current_W = self._refine_weights_by_scale(current_W)
        return current_W
    
    def _refine_weights_by_scale(self, current_W: sp.csc_matrix):
        """
        非對角線集體幅值修正 0秒流
        保持對角線剛性絕對不動，只對已選中的非對角線邊權進行集體微調，瞬間鎖定最大特徵值精度
        """
        #
        evals, _ = sp.linalg.eigsh(current_W, k=self.config.top_n, which='LA')
        
        # 2. alpha
        alpha = np.sum(self.target_evals_top_n * evals) / np.sum(evals ** 2)
        ratios = (evals + 1e-30) / (self.target_evals_top_n + 1e-30)
        alpha = np.sum(ratios) / np.sum(ratios ** 2)
        # 安全約束：只允許在 0.8 到 1.2 之間微調
        alpha = np.clip(alpha, 0.8, 1.2)
        print(" 最终修复 rescaling non-diagnoal alpha ", alpha)
        
        # 3. 分层提取：提取非對角線並乘以 alpha
        refined_W = current_W.copy()
        orig_diag = current_W.diagonal()
        refined_W = refined_W * alpha
        
        refined_W.setdiag(orig_diag)
        refined_W.eliminate_zeros()
        return refined_W
    
    def _calcualte_edge_degree(self, spmat:sp.csc_matrix, us:np.ndarray, vs:np.ndarray):
        """
        计算节点度, 评估fill-in
        """
        # 1. Degrees are simply the sum of rows in the adjacency matrix
        # (or the number of non-zero elements per row)
        degrees = np.diff(spmat.indptr)
        deg_u = degrees[us]
        deg_v = degrees[vs]
        deg_product = deg_u * deg_v
        
        # 2. Common neighbors using CSR row slicing
        indptr = spmat.indptr
        indices = spmat.indices
        N = dim
        num_edges = len(us)
        common_counts = np.zeros(num_edges, dtype=np.int32)
        chunk_size = 500000 
        for start_idx in range(0, num_edges, chunk_size):
            end_idx = min(start_idx + chunk_size, num_edges)
            c_us = us[start_idx:end_idx]
            c_vs = vs[start_idx:end_idx]
            num_chunk = len(c_us)
            s_u, e_u = indptr[c_us], indptr[c_us + 1]
            s_v, e_v = indptr[c_vs], indptr[c_vs + 1]
            len_u = e_u - s_u
            len_v = e_v - s_v
            
            # 向量化平摊克隆
            edge_ids_u = np.repeat(np.arange(num_chunk, dtype=np.int32), len_u)
            edge_ids_v = np.repeat(np.arange(num_chunk, dtype=np.int32), len_v)
            
            # 闪电拉出一维邻居大数组（由于直接操作内存物理视图，速度极快）
            cum_lens_u = np.cumsum(len_u)
            offsets_u = np.arange(cum_lens_u[-1], dtype=np.int32) - np.repeat(cum_lens_u - len_u, len_u)
            indices_ptrs_u = np.repeat(s_u, len_u) + offsets_u
            all_neighbors_u = indices[indices_ptrs_u]
            #
            cum_lens_v = np.cumsum(len_v)
            offsets_v = np.arange(cum_lens_v[-1], dtype=np.int32) - np.repeat(cum_lens_v - len_v, len_v)
            indices_ptrs_v = np.repeat(s_v, len_v) + offsets_v
            all_neighbors_v = indices[indices_ptrs_v]
            #
            combined_edge_ids = np.concatenate([edge_ids_u, edge_ids_v])
            combined_neighbors = np.concatenate([all_neighbors_u, all_neighbors_v])
            shift_bits = int(np.ceil(np.log2(dim)))
            combined_ids = (combined_edge_ids << shift_bits) | combined_neighbors
            combined_ids.sort(kind='mergesort')
            matches = combined_ids[:-1] == combined_ids[1:]

            if np.any(matches):
                #matched_edge_ids = combined_ids[1:][matches] // N
                matched_edge_ids = combined_ids[1:][matches] >> shift_bits
                counts = np.bincount(matched_edge_ids, minlength=num_chunk)
                common_counts[start_idx:end_idx] = counts
        
        return deg_product, common_counts
    
    def _calculate_exact_spectral(self, current_W: sp.csc_matrix, sampled_candidates: list):
        """
        Exact Lambda Evaluation:
        Instead of trusting 1st-order math, this inserts an entire candidate batch,
        checks the absolute true eigenvalue loss, and decides whether to accept the batch.
        """
        if len(sampled_candidates) == 0:
            return current_W, False

        # 1. Temporarily assemble ALL sampled edges into a delta matrix
        dim = self.W_target.shape[0]
        U_list, V_list, DATA_list = [], [], []
        for u, v in sampled_candidates:
            # Must use the exact continuous true weight from target matrix
            true_w = self.W_target[u, v]
            U_list.extend([u, v])
            V_list.extend([v, u])
            DATA_list.extend([true_w, true_w])
            
        W_delta = sp.csc_matrix((DATA_list, (U_list, V_list)), shape=(dim, dim))
        
        # 2. Speculatively add them to the matrix
        W_try = current_W + W_delta
        W_try.eliminate_zeros()
        
        # 3. Calculate the ABSOLUTE EXACT eigenvalues of this speculative matrix
        try:
            evals_try, evecs_try = sp.linalg.eigsh(
                W_try, 
                k=self.config.top_n, 
                which='LA', 
                ncv=self.config.top_n * 2,
                maxiter=1000
            )
            # True exact loss
            return evals_try
        except sp.linalg.ArpackNoConvergence:
            # If numerical solvers fail to converge, reject this batch safely
            return None

    def _calculate_spectral_gains(self, current_W:sp.csc_matrix, UV_idx:tuple, warm_start_v0:np.ndarray, add_sign=1.0):
        # 计算当前矩阵的谱, 洛伦兹迭代很快
        evals, evecs = sp.linalg.eigsh (
            current_W, 
            k=self.config.top_n,
            which='LA',              # 最大（Largest Algbra)
            v0=warm_start_v0,        # 传入上一轮的最优残差估计，收敛速度暴增
            ncv=self.config.top_n*2,
            maxiter=1000
        )
        
        # 使用来自 config 的 top_n
        current_loss = np.sum((evals - self.target_evals_top_n)**2)
        warm_start_v0[:] = evecs[:, -1]
        
        # 2. 批量获取这 100 万条边在原图中的连续目标权重 (Shape: (sample_size,))
        U_idx, V_idx = UV_idx
        W_targets = self.W_target[U_idx, V_idx]
        
        # 3. 终极矩阵广播：一瞬间算出所有抽样边对所有 top_n 特征值的一阶贡献
        # evecs[U_idx, :] 的 Shape 是 (sample_size, top_n)
        X_u = evecs[U_idx, :]
        X_v = evecs[V_idx, :]
        delta_1st_all = add_sign * 2 * W_targets[:, None] * X_u * X_v
        #delta_1st_all = add_sign * W_targets[:, None] * (X_u**2 + X_v**2 - X_u * X_v)
        
        # 4. 批量计算预测特征值 (Shape: (sample_size, top_n))
        predicted_evals_all = evals[None, :] + delta_1st_all
        
        # 5. 批量计算所有边对应的预测 Loss (Shape: (sample_size,))
        predicted_losses = np.sum((predicted_evals_all - self.target_evals_top_n[None, :]) ** 2, axis=1)
        
        # 6. 计算收益（Gain）并与边坐标直接绑定排序
        gains = current_loss - predicted_losses
        
        return gains
    
    def _add_edges(self, current_W:sp.csc_matrix, scores:np.ndarray, UVidx:tuple, all_candidates:set):
        dim = self.W_backbone.shape[0] # type: ignore
        curr_total = current_W.nnz
        flag = False
        
        if len(scores) > 0:
            sorted_idx = np.sort(-scores) # 降序排列，取出前n大
            add_size = max(1, int(self.config.add_ratio * len(scores)))
            add_ratio = (curr_total + add_size)/(dim*dim)
            if add_ratio > self.config.target_density_ratio:
                print(f"   Current added edges number reach setting ratio,"
                      f" {self.config.target_density_ratio:.3f}")
                add_size = int(self.config.target_density_ratio*dim*dim) - curr_total
                flag = True
            
            sorted_indices = np.argsort(-scores)
            top_indices = sorted_indices[:add_size]

            top_U = UVidx[0][top_indices]
            top_V = UVidx[1][top_indices]
            top_w = self.W_target[top_U, top_V]

            pure_upper_edges = [
                (int(u), int(v)) if u < v else (int(v), int(u)) 
                for u, v in zip(top_U, top_V)
            ]
            all_candidates.difference_update(pure_upper_edges)
            
            rows = np.concatenate([top_U, top_V])
            cols = np.concatenate([top_V, top_U])
            data = np.concatenate([top_w, top_w])
            # 利用 COO 格式将这一批新边打包成一个独立的稀疏增量矩阵
            delta_W_sparse = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsc()
            # 两个稀疏矩阵直接做加法，在内存连续块中一瞬间完成拓扑合流！
            current_W = current_W + delta_W_sparse
            print(f"   [Batch] 成功恢复边数: {add_size}")
        return current_W, flag
        
        
    def _drop_edges(self, current_W:sp.csc_matrix, collect_gain:list):
        dim = self.W_backbone.shape[0] # type: ignore
        curr_total = current_W.nnz
        
        if len(collect_gain) > 0:
            sorted_gain = sorted(collect_gain, key=lambda x: x[0], reverse=True)
            add_size = max(1, int(self.config.add_ratio * len(collect_gain)))
            count_add = 0
            U_list, V_list, DATA_list = [], [], []
            for gain_val, (u, v) in sorted_gain[:add_size]:
                U_list.extend([u, v])
                V_list.extend([v, u])
                DATA_list.extend([self.W_target[u, v], self.W_target[u, v]])
                count_add += 2 # 对称
                
                drop_ratio = (curr_total + count_add)/(dim*dim)
                if drop_ratio < self.config.target_density_ratio:
                    print(f"   Current drop edges number reach setting ratio, {drop_ratio:.3f}")
                    delta_W_sparse = sp.coo_matrix((DATA_list, (U_list, V_list)), shape=(dim, dim)).tocsc()
                    current_W = (current_W - delta_W_sparse).tocsc()
                    current_W.eliminate_zeros()
                    return current_W, True
            
            # 利用 COO 格式将这一批新边打包成一个独立的稀疏增量矩阵
            delta_W_sparse = sp.coo_matrix((DATA_list, (U_list, V_list)), shape=(dim, dim)).tocsc()
            # 两个稀疏矩阵直接做加法，在内存连续块中一瞬间完成拓扑合流！
            current_W = current_W - delta_W_sparse
            current_W.eliminate_zeros()
            print(f"   [Batch] 成功移除边数: {count_add}")
            return current_W, False
    
    def _calculate_Reffective_gains(self, current_W:sp.csc_matrix, UV_idx:tuple):
        # 计算当前有效电阻Re
        Re_curr = self._compute_effective_resistances(current_W)
        sorted_indices = np.argsort(-Re_curr, axis=None)[: self.config.top_n]
        Re_curr_sorted = Re_curr.flat[sorted_indices]
        
        # 使用来自 config 的 top_n
        current_loss = np.sum((Re_curr_sorted - self.target_evals_top_n)**2)
        
        # 2. 批量获取这 100 万条边在原图中的连续目标权重 (Shape: (sample_size,))
        U_idx, V_idx = UV_idx
        
        # 3. 算出对所有 top_n 特征值的一阶贡献
        #运用 Sherman-Morrison 秩一微扰闭式解
        # 计算公式： \Delta Reff = (w * R^2) / (1 + w * R)
        current_reff = Re_curr[U_idx, V_idx]
        delta_weight = self.W_target[U_idx, V_idx]
        
        # 该边加入后，全图在该通道上能够“挽回/增长”的谱能量差值（得分）
        # reff_decrease = - (w * R * R) / (1.0 + w * R)
        # reff_decrease = -(delta_weight * current_reff * current_reff) / (1.0 + delta_weight * current_reff)
        rho = delta_weight / (1.0 + delta_weight * current_reff + 1e-20) # Shape: (sample_size,)
        
        top_x, top_y = np.unravel_index(sorted_indices, Re_curr.shape) # Shape: (top_n,)
        term1 = Re_curr[top_x[None, :], V_idx[:, None]] # R(x, j)
        term2 = Re_curr[top_y[None, :], U_idx[:, None]] # R(y, i)
        term3 = Re_curr[top_x[None, :], U_idx[:, None]] # R(x, i)
        term4 = Re_curr[top_y[None, :], V_idx[:, None]] # R(y, j)
        # 拓扑传导因子 T = Z_xi - Z_xj - Z_yi + Z_yj
        T = 0.5 * (term1 + term2 - term3 - term4) # Shape: (sample_size, top_n)
        
        # 算出 100 万条新边对 116 个监控端口产生的全局有效电阻下压矩阵
        reff_decrease_all = - rho[:, None] * (T ** 2) # Shape: (sample_size, top_n)
        
        # 4. 批量计算预测特征值 (Shape: (sample_size, top_n))
        predicted_evals_all = Re_curr_sorted[None, :] + reff_decrease_all
        
        # 5. 批量计算所有边对应的预测 Loss (Shape: (sample_size,))
        predicted_losses = np.sum((predicted_evals_all - self.target_evals_top_n[None, :]) ** 2, axis=1)
        
        # 6. 计算收益（Gain）并与边坐标直接绑定排序
        gains = current_loss - predicted_losses
        
        return gains

    def _compute_effective_resistances(self):
        if isinstance(W_current, sp.csc_matrix):
            W_current = W_current.toarray()
        # 1. 构建原稠密图的拉普拉斯矩阵
        W_adj = W_current.copy()
        np.fill_diagonal(W_adj, 0)
        # 提取出对角线接地项（对地电导/磁阻）
        G_ground = np.diag(W_current)

        # 2. 构建完善的接地拉普拉斯矩阵
        # D 仅由跨节点邻接矩阵的行和决定
        D = np.diag(np.sum(W_adj, axis=1))
        # 最终的 L 必须叠加对地项 G_ground！此时 L 变为满秩正定矩阵
        L = D - W_adj + np.diag(G_ground)
        
        # 2. 计算伪逆 (Pinverse)
        L_pinv = la.inv(L)
        
        # Re 计算公式
        diag_pinv = np.diag(L_pinv)
        # 通过广播机制生成 Re_matrix[i, j] = L_pinv[i,i] + L_pinv[j,j] - 2*L_pinv[i,j]
        Re_matrix = diag_pinv[:, None] + diag_pinv[None, :] - 2.0 * L_pinv # type:ignore
        
        # 4. 计算 Spielman-Teng 谱重要性得分 (Score = w * Re)
        # 只有在原图里有权重的边才计算
        weights = W_current
        # valid_mask = weights > 1e-15
        # weights = weights[valid_mask]
        # Re_sampled = Re_matrix[valid_mask]
        scores = weights * Re_matrix
        # 归一化为概率分布
        # self.edge_probs = scores / np.sum(scores)
        # # 将候选边打包保存
        # self.static_candidates = list(zip(u_idx, v_idx))
        # self.static_weights = weights
        return scores

    def spectral_optimize_by_re(self) -> sp.csc_matrix:
        """
        全新的有效电阻抽样迭代优化器：
        通过基于 Re 的采样探索拓扑，通过真实的谱 Cost 进行闭环拦截与损失评估。
        """
        dim = self.W_backbone.shape[0]
        current_W = self.W_backbone.copy()
        
        # 提前提取目标 Top-N 谱
        target_evals_all, _ = la.eigh(self.W_target)
        self.target_evals_top_n = target_evals_all[-self.config.top_n:]
        
        warm_start_v0 = np.ones(dim)
        
        # 计算初始骨架的真实 Loss
        evals, evecs = sp.linalg.eigsh(current_W, k=self.config.top_n, which='LA', v0=warm_start_v0)
        best_loss = np.sum((evals - self.target_evals_top_n) ** 2)
        best_W = current_W.copy()
        
        # 动态维护一个“尚未被选入骨架”的静态索引掩码
        active_candidate_mask = np.ones(len(self.static_candidates), dtype=bool)
        
        for iteration in range(self.config.max_iters):
            curr_ratio = current_W.nnz / (dim * dim)
            if curr_ratio >= self.config.target_density_ratio or not np.any(active_candidate_mask):
                break
                
            # 1. 抽取当前可用的候选边及对应的 Re 概率
            available_indices = np.where(active_candidate_mask)[0]
            current_probs = self.edge_probs[available_indices]
            current_probs /= np.sum(current_probs)  # 重新归一化分布
            
            # 2. 按照 Re 概率进行批次抽样 (无脑乐观加边数量)
            batch_size = max(1, int(len(available_indices) * self.config.add_ratio))
            sampled_meta_indices = np.random.choice(
                available_indices, size=batch_size, replace=False, p=current_probs
            )
            
            # 3. 试探性构建增量矩阵 Delta W
            U_list, V_list, DATA_list = [], [], []
            for idx in sampled_meta_indices:
                u, v = self.static_candidates[idx]
                w = self.static_weights[idx]
                U_list.extend([u, v])
                V_list.extend([v, u])
                DATA_list.extend([w, w])
                
            W_delta = sp.csc_matrix((DATA_list, (U_list, V_list)), shape=(dim, dim))
            W_try = current_W + W_delta
            W_try.eliminate_zeros()
            
            # 4. 通过真实计算特征值，评估这个批次的真实 Loss 贡献
            try:
                evals_try, evecs_try = sp.linalg.eigsh(
                    W_try, k=self.config.top_n, which='LA', v0=warm_start_v0, maxiter=1000
                )
                loss_try = np.sum((evals_try - self.target_evals_top_n) ** 2)
                np.copyto(warm_start_v0, evecs_try[:, -1])  # 更新热启动向量
            except sp.linalg.ArpackNoConvergence:
                # 谱求解若不收敛，直接视作恶性注入，跳过
                continue
            
            # 5. 反馈判定：利用谱 Cost 目标做出接受或回退的抉择
            if loss_try < best_loss:
                # 【接受该批次】：因为真实谱误差下降了！
                current_W = W_try
                best_loss = loss_try
                best_W = current_W.copy()
                # 从候选池中永久剔除这些已经生效的边
                active_candidate_mask[sampled_meta_indices] = False
                print(f"迭代 {iteration}: [接受] 成功注入 {batch_size} 条高Re边, 真实误差降至: {best_loss:.4g}")
            else:
                # 【拒绝并回退】：此批次边对特定 Top-N 产生了谱污染，触发 Rollback
                # current_W 保持原样不加上去，但我们可以选择不剔除这些边，或者把它们暂时锁定
                print(f"迭代 {iteration}: [拒绝] 该批次引发谱漂移 (Loss: {loss_try:.4g} >= 最佳: {best_loss:.4g}), 触发自动回退。")

        return best_W


def evaluate_fill_in_via_ldl(mat: sp.csc_matrix):
    """
    使用 LDLt 分解结构评估两个 CSC 稀疏矩阵的全量 Fill-in 差异。
    注意：输入应当是结构对称的（结构非零元对称）。
    """
    # 对矩阵进行 LDL 分解并统计
    try:
        lu = sp.linalg.splu(mat.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0)
        # L 和 U 的总非零元就是该图拓扑带来的全量 Fill-in 表现
        nnz_factors = lu.L.nnz + lu.U.nnz
        fills = nnz_factors - mat.nnz
    except RuntimeError as e:
        print(f"矩阵 LDL 分解失败（可能是奇异矩阵或未做预排序）: {e}")
        return None
        
    return fills

def test_specsparsify():
    # 加载矩阵图
    print(" load numpy matrix ")
    W_target = np.loadtxt('fipchip_Ldense.dat')
    dim = W_target.shape[0]
    print(f" matrix, dim {dim} "
          f"neg nnz {np.count_nonzero(W_target<1e-20)} "
          f"pos nnz {np.count_nonzero(W_target>1e-20)}"
    )
    
    # === 工程实例化配置结构体 ===
    sconfig = SpectralConfig(
        top_n=10,
        max_iters=500,
        sample_ratio=0.3,
        add_ratio=0.1,
        target_density_ratio=0.75
    )
    
    # 实例化谱稀疏化对象
    sparsifier = SpectralSparsification()
    sparsifier.set_graph(W_target)
    sparsifier.set_config(sconfig)
    
    # 运行算法
    sparsifier.generate_backbone()
    final_W = sparsifier.spectral_optimize()
    # sparsifier.compute_effective_resistances()
    # final_W = sparsifier.spectral_optimize_by_re()
    
    # 精确验证谱恢复结果
    target_evals, target_evecs = la.eigh(W_target)
    final_evals, final_evecs = la.eigh(final_W.toarray())
    
    #
    
    print(f"\n[针对前 {sconfig.top_n} 个最大特征值的优化报告]")
    print(f"\n density ratio ", final_W.nnz / (dim*dim))
    evals_diff_abs = final_evals[-sconfig.top_n:] - target_evals[-sconfig.top_n:]
    evals_diff_rel = evals_diff_abs / (np.abs(target_evals[-sconfig.top_n:]) + 1e-15)
    final_top_n_mse = np.sum((evals_diff_abs)**2)
    cos_matrix = target_evecs.T @ final_evecs
    loss_evecs = cos_matrix.shape[0] - np.sum(cos_matrix ** 2)
    print(" 原始矩阵前n个最大特征值\n", target_evals[-sconfig.top_n:])
    print(" 谱近似矩阵前n个最大特征值\n", final_evals[-sconfig.top_n:])
    print(f"Top-{sconfig.top_n} 总和绝对误差 (MSE): {final_top_n_mse:.4e}\n"
          f"Top-{sconfig.top_n} 平均绝对误差(MSE): {final_top_n_mse/sconfig.top_n:.2e}\n"
          f"Top-{sconfig.top_n} 相对误差:\n {evals_diff_rel}\n"
          f"Top-{sconfig.top_n} 平均相对误差: {np.mean(np.abs(evals_diff_rel)):.4e}\n"
          f" 特征向量几何误差: {loss_evecs:.4e}")

    print(f" 稀疏矩阵fill-in评估 {evaluate_fill_in_via_ldl(final_W)}")

    # 评估 Reff
    Reff = sparsifier._compute_effective_resistances(W_target)
    Reff_sp = sparsifier._compute_effective_resistances(final_W.toarray())
    Reff_bb = sparsifier._compute_effective_resistances(backbone.toarray())
    Reff_sorted = np.argsort(-Reff, axis=None)[:20]
    Reff_sp_sorted = np.argsort(-Reff_sp, axis=None)[:20]
    Reff_bb_sorted = np.argsort(-Reff_bb, axis=None)[:20]
    print(" org shape ", Reff.shape)
    print(" sp shape ", Reff_sp.shape)
    print(" bb shape ", Reff_bb.shape)
    print(" original top Reff \n", Reff.flat[Reff_sorted])
    print(" sparse top Reff \n", Reff_sp.flat[Reff_sp_sorted])
    print(" backbobe top Reff \n", Reff_bb.flat[Reff_bb_sorted])

# ==================== 测试验证 ====================
if __name__ == "__main__":
    test_specsparsify()
