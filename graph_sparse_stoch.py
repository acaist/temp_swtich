import numpy as np
import scipy.linalg as la
from scipy import sparse as sp
from scipy.sparse import csgraph
import random
from dataclasses import dataclass

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
        target_evals_all, target_evecs = la.eigh(self.W_target)
        self.target_evals_top_n = target_evals_all[-self.config.top_n:] # 升序矩阵中，最后 n 个是最大的

        # 特征值的初始搜索向量
        warm_start_v0 = np.ones(dim)
        
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
            
            # 矩阵一阶微扰动快速评估收益
            # 评估采样边
            U_idx = np.array([u for u, v in sampled_candidates], dtype=np.int32)
            V_idx = np.array([v for u, v in sampled_candidates], dtype=np.int32)
            gains = collect_gain = self._calculate_gains(current_W, (U_idx, V_idx), warm_start_v0, add_sign=1.0)
            # 利用 zip 快速组装，只有在最终排序时才接触 Python 对象，速度提升数万倍
            collect_gain = list(zip(gains, sampled_candidates))
          
            # 批量恢复 Top-N 收益边 (使用来自 config 的 add_ratio)
            current_W, isfinished = self._add_edges(current_W, collect_gain, all_candidates)
            if isfinished:
                return current_W

        return current_W
    
    def _calculate_gains(self, current_W:sp.csc_matrix, UV_idx:tuple, warm_start_v0:np.ndarray, add_sign=1.0):
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
        delta_1st_all = add_sign * 2 * W_targets[:, None] * evecs[U_idx, :] * evecs[V_idx, :]
        
        # 4. 批量计算预测特征值 (Shape: (sample_size, top_n))
        # evals[None, :] 会自动沿行方向广播复制
        predicted_evals_all = evals[None, :] + delta_1st_all
        
        # 5. 批量计算所有边对应的预测 Loss (Shape: (sample_size,))
        # 沿着 axis=1 (top_n 轴) 求和，直接吐出 100 万个 Loss 值
        predicted_losses = np.sum((predicted_evals_all - self.target_evals_top_n[None, :]) ** 2, axis=1)
        
        # 6. 计算收益（Gain）并与边坐标直接绑定排序
        gains = current_loss - predicted_losses
                
        return gains
    
    def _add_edges(self, current_W:sp.csc_matrix, collect_gain:list, all_candidates:set):
        dim = self.W_backbone.shape[0] # type: ignore
        curr_total = current_W.nnz
        
        if len(collect_gain) > 0:
            sorted_gain = sorted(collect_gain, key=lambda x: x[0], reverse=True)
            add_size = max(1, int(self.config.add_ratio * len(collect_gain)))
            count_add = 0
            U_list, V_list, DATA_list = [], [], []
            for gain_val, (u, v) in sorted_gain[:add_size]:
                if gain_val < 0:
                    continue
                U_list.extend([u, v])
                V_list.extend([v, u])
                DATA_list.extend([self.W_target[u, v], self.W_target[u, v]])
                all_candidates.remove((u, v))
                count_add += 2 # 对称
                
                add_ratio = (curr_total + count_add)/(dim*dim)
                if add_ratio > self.config.target_density_ratio:
                    print(f"   Current added edges number reach setting ratio, {add_ratio:.3f}")
                    delta_W_sparse = sp.coo_matrix((DATA_list, (U_list, V_list)), shape=(dim, dim)).tocsc()
                    current_W = (current_W + delta_W_sparse).tocsc()
                    return current_W, True
            
            # 利用 COO 格式将这一批新边打包成一个独立的稀疏增量矩阵
            delta_W_sparse = sp.coo_matrix((DATA_list, (U_list, V_list)), shape=(dim, dim)).tocsc()
            # 两个稀疏矩阵直接做加法，在内存连续块中一瞬间完成拓扑合流！
            current_W = current_W + delta_W_sparse
            print(f"   [Batch] 成功恢复边数: {count_add}, 剩余候选池: {len(all_candidates)}")
            return current_W, False


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
        target_density_ratio=0.7
    )
    
    # 实例化谱稀疏化对象
    sparsifier = SpectralSparsification()
    sparsifier.set_graph(W_target)
    sparsifier.set_config(sconfig)
    
    # 运行算法
    sparsifier.generate_backbone()
    final_W = sparsifier.spectral_optimize()
    
    # 精确验证谱恢复结果
    target_evals, _ = la.eigh(W_target)
    final_evals, _ = la.eigh(final_W.toarray())
    
    print(f"\n[针对前 {sconfig.top_n} 个最大特征值的优化报告]")
    evals_diff_abs = final_evals[-sconfig.top_n:] - target_evals[-sconfig.top_n:]
    evals_diff_rel = evals_diff_abs / (np.abs(target_evals[-sconfig.top_n:]) + 1e-15)
    final_top_n_mse = np.sum((evals_diff_abs)**2)
    print(" 原始矩阵前n个最大特征值\n", target_evals[-sconfig.top_n:])
    print(" 谱近似矩阵前n个最大特征值\n", final_evals[-sconfig.top_n:])
    print(f"Top-{sconfig.top_n} 总和绝对误差 (MSE): {final_top_n_mse:.4e}\n"
          f"Top-{sconfig.top_n} 平均绝对误差(MSE): {final_top_n_mse/sconfig.top_n:.2e}\n"
          f"Top-{sconfig.top_n} 相对误差:\n {evals_diff_rel}\n"
          f"Top-{sconfig.top_n} 平均相对误差: {np.mean(np.abs(evals_diff_rel)):.4e}")


# ==================== 测试验证 ====================
if __name__ == "__main__":
    test_specsparsify()
