import numpy as np
import scipy.linalg as la
from scipy import sparse as sp
import random
from dataclasses import dataclass

@dataclass
class SpectralConfig:
    """算法控制参数结构体"""
    top_n: int                        # 指定优化前多少个最大的特征值
    max_iters: int = 1000             # 最大迭代次数
    sample_ratio: float = 0.5         # 候选池抽样比例 (SimplStoch 思想)
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


def spectral_greedy_top_n_largest(W_base, W_target, config: SpectralConfig):
    """
    通过贪心微扰策略，恢复 W_target 权重到 W_base，使【前 n 个最大特征值】的误差平方和最小。
    所有的调节参数通过 config 结构体传入。
    """
    dim = W_base.shape[0]
    current_W = W_base.copy()
    
    # 1. 循环外：初始化候选边池
    all_candidates = set()
    for u in range(dim):
        for v in range(u + 1, dim):
            if current_W[u, v] == 0 and W_target[u, v] > 0:
                all_candidates.add((u, v))
    print(f"初始候选池 {len(all_candidates)}, Wtarget nnz/2 {np.count_nonzero(W_target)/2}") 
                 
    # 2. 提前计算目标谱，并提取出前 n 个最大的特征值
    target_evals_all, _ = la.eigh(W_target)
    target_evals_top_n = target_evals_all[-config.top_n:] # 升序矩阵中，最后 n 个是最大的
    
    # 特征值的初始搜索向量
    warm_start_v0 = np.ones(dim)
    
    for iteration in range(config.max_iters):
        curr_total = np.count_nonzero(current_W)
        curr_ratio = curr_total / (dim * dim)
        
        print(f"迭代 {iteration}: 当前非零元 {curr_total}, 密度 {curr_ratio:.3f}, 目标密度 {config.target_density_ratio:.3f}")
        if curr_ratio >= config.target_density_ratio or not all_candidates:
            print(" -> 满足终止条件，退出循环, 候选池剩余数量", len(all_candidates))
            break
            
        # 3. 计算当前矩阵的谱, Lenvz 迭代很快
        #evals, evecs = la.eigh(current_W)
        current_W_sparse = sp.csc_matrix(current_W)
        evals, evecs = sp.linalg.eigsh(
            current_W_sparse, 
            k=config.top_n,
            which='LA',              # 最大（Largest Algbra)
            v0=warm_start_v0,        # 【热启动】传入上一轮的最优残差估计，收敛速度暴增
            ncv=config.top_n*2,
            maxiter=1000
        )
        
        # 使用来自 config 的 top_n
        current_loss = np.sum((evals - target_evals_top_n)**2)
        warm_start_v0 = evecs[:, -1]
        
        # 4. 从候选池进行随机采样 (使用来自 config 的 sample_ratio)
        sample_size = max(1, int(len(all_candidates) * config.sample_ratio))
        sampled_candidates = random.sample(list(all_candidates), sample_size)
        
        # 5. 评估采样边
        U_idx = np.array([u for u, v in sampled_candidates], dtype=np.int32)
        V_idx = np.array([v for u, v in sampled_candidates], dtype=np.int32)
        
        # . 批量获取这 100 万条边在原图中的连续目标权重 (Shape: (sample_size,))
        W_targets = W_target[U_idx, V_idx]
        
        # . 终极矩阵广播：一瞬间算出所有抽样边对所有 top_n 特征值的一阶贡献
        # evecs[U_idx, :] 的 Shape 是 (sample_size, top_n)
        delta_1st_all = 2 * W_targets[:, None] * evecs[U_idx, :] * evecs[V_idx, :]
        
        # . 批量计算预测特征值 (Shape: (sample_size, top_n))
        # evals[None, :] 会自动沿行方向广播复制
        predicted_evals_all = evals[None, :] + delta_1st_all
        
        # . 批量计算所有边对应的预测 Loss (Shape: (sample_size,))
        # 沿着 axis=1 (top_n 轴) 求和，直接吐出 100 万个 Loss 值
        predicted_losses = np.sum((predicted_evals_all - target_evals_top_n[None, :]) ** 2, axis=1)

        gains = current_loss - predicted_losses
        
        # 利用 zip 快速组装
        collect_gain = list(zip(gains, sampled_candidates))
        
        # 6. 批量恢复 Top-N 收益边 (使用来自 config 的 add_ratio)
        if collect_gain:
            sorted_gain = sorted(collect_gain, key=lambda x: x[0], reverse=True)
            add_size = max(1, int(config.add_ratio * len(collect_gain)))
            count_add = 0
            
            for gain_val, (u, v) in sorted_gain[:add_size]:
                current_W[u, v] = W_target[u, v]
                current_W[v, u] = W_target[u, v]
                all_candidates.remove((u, v))
                count_add += 2 # 对称
                
                add_ratio = (curr_total + count_add)/(dim*dim)
                if add_ratio > sconfig.target_density_ratio:
                    print(f"   Current added edges number reach setting ratio, {add_ratio:.3f}")
                    break
          
            print(f"   [Batch] 成功恢复边数: {count_add}, 剩余候选池: {len(all_candidates)}")
            
    return current_W


# ==================== 测试验证 ====================
if __name__ == "__main__":
    # dim = 1000     
    # W_target = generate_spd_diagonal_decay_target(dim, decay_rate=0.06)
    
    print(" load numpy matrix ")
    W_target = np.loadtxt('fipchip_Ldense.dat')
    dim = W_target.shape[0]
    print(f" matrix, dim {dim} "
          f"neg nnz {np.count_nonzero(W_target<0)} "
          f"pos nnz {np.count_nonzero(W_target>0)}"
    )
    
    # 骨架小图
    W_base = np.zeros((dim, dim))
    np.fill_diagonal(W_base, np.diagonal(W_target)) 
    
    # === 工程优化点：实例化配置结构体 ===
    # 这样调用时参数一目了然，不需要写冗长的参数列表
    sconfig = SpectralConfig(
        top_n=10,
        max_iters=500,
        sample_ratio=0.3,
        add_ratio=0.3,
        target_density_ratio=0.7
    )
    
    # 3. 运行算法（此时入参非常干净）
    final_W = spectral_greedy_top_n_largest(W_base, W_target, config=sconfig)
    
    # 4. 精确验证谱恢复结果
    target_evals, _ = la.eigh(W_target)
    final_evals, _ = la.eigh(final_W)
    
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
