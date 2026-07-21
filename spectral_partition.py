
import numpy as np
import scipy.linalg as la  # 稠密矩阵使用标准 linalg
import time
import matplotlib.pyplot as plt

def create_dense_mesh_graph(grid_size=32):
    """
    生成一个 1024 * 1024 左右的稠密邻接矩阵（模拟二维网格图）
    """
    n = grid_size * grid_size  # 32 * 32 = 1024 节点
    print(f"[Init] 创建邻接全稠密矩阵... 节点数 N = {n}")
    
    adj = np.zeros((n, n), dtype=np.float32)
    for r in range(grid_size):
        for c in range(grid_size):
            node = r * grid_size + c
            if c + 1 < grid_size:
                neighbor = node + 1
                adj[node, neighbor] = 1.0/(np.abs(r - c) + 1)
                adj[neighbor, node] = 1.0/(np.abs(r - c) + 1)
            if r + 1 < grid_size:
                neighbor = node + grid_size
                adj[node, neighbor] = 1.0/(np.abs(r - c) + 1)
                adj[neighbor, node] = 1.0/(np.abs(r - c) + 1)
    return adj

def dense_spectral_bisection_mask(adj_matrix):
    """
    针对稠密矩阵的谱双切：计算 Fiedler 向量并返回当前相对尺寸的布尔掩码
    """
    n = adj_matrix.shape[0]
    
    # 1. 计算度数矩阵 D 和拉普拉斯矩阵 L = D - A
    degrees = np.sum(adj_matrix, axis=1)
    D = np.diag(degrees)
    L = D - adj_matrix
    
    # 2. 稠密矩阵直接使用 eigh（专门针对对称/埃尔米特矩阵，求出的特征值自动升序排列）
    # 我们只需要前 2 个最小特征值对应的特征向量
    try:
        eigenvalues, eigenvectors = la.eigh(L)
        # eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=2, which='SM', sigma=0.0, v0=np.ones(n))
        fiedler_vector = eigenvectors[:, 1]  # 第二小特征值对应的特征向量
    except Exception as e:
        print(f"[Warning] 特征值求解失败，采用降级均分: {e}")
        fiedler_vector = np.arange(n)
        
    # 3. 💥 核心：根据中位数生成相对于当前矩阵尺寸的布尔掩码
    median_val = np.median(fiedler_vector)
    left_mask = fiedler_vector < median_val
    right_mask = ~left_mask
    
    return left_mask, right_mask

def rsb_dense(adj_matrix, global_indices, k):
    """
    递归谱分割（稠密矩阵专用版）
    """
    # 递归终止条件
    if k <= 1 or adj_matrix.shape[0] <= 1:
        return [global_indices.tolist()]
    
    # 1. 获取当前层级的划分掩码
    left_mask, right_mask = dense_spectral_bisection_mask(adj_matrix)
    
    # 防止切出空集导致死循环
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return [global_indices.tolist()]
    
    # 2. 💥 【彻底免疫越界】：提取子矩阵
    # 稠密矩阵（NumPy）使用布尔掩码在两个维度上切片，非常丝滑，绝无 IndexError
    adj_left = adj_matrix[left_mask][:, left_mask]
    adj_right = adj_matrix[right_mask][:, right_mask]
    
    # 3. 💥 【精准追踪】：过滤出属于子树的绝对全局节点编号
    next_global_left = global_indices[left_mask]
    next_global_right = global_indices[right_mask]
    
    # 4. 递归向下传播缩小的矩阵与缩小的绝对索引
    clusters_left = rsb_dense(adj_left, next_global_left, k // 2)
    clusters_right = rsb_dense(adj_right, next_global_right, k - k // 2)
    
    return clusters_left + clusters_right

def plot_block_matrix_with_large_pixels(original_adj, result_clusters, total_nodes, weight_threshold=0.01):
    """
    通过提取非零/大元素，并利用散点图大小(Size)与权重绑定的方式，让关键边无限放大、清晰可见。
    
    参数:
    - original_adj: 4096 * 4096 的原始稠密全连接矩阵
    - result_clusters: RSB 返回的 8 个分区列表
    - weight_threshold: 过滤背景噪声的阈值。大于该值的元素会被放大显示
    """
    # 1. 顺次拼接 8 个分区的节点编号进行行列重排
    reordered_indices = []
    boundary_lines = [] 
    current_idx = 0
    for cluster in result_clusters:
        reordered_indices.extend(cluster)
        current_idx += len(cluster)
        boundary_lines.append(current_idx)
    reordered_indices = np.array(reordered_indices)
    reordered_adj = original_adj[np.ix_(reordered_indices, reordered_indices)]
    
    # 2. 💥 核心变轨：找出所有大于阈值的重要大元素（非零元）的行列索引
    # rows 和 cols 拿到了它们在重排矩阵中的局部绝对坐标
    rows, cols = np.where(reordered_adj >= weight_threshold)
    weights = reordered_adj[rows, cols]
    
    if len(weights) == 0:
        print("[Warning] 没有找到大于该阈值的元素，请调低 weight_threshold")
        return

    # 3. 创建高清晰度画布
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_facecolor('white') # 保持纯白底色，对比度最高
    
    # 4. 💥 精髓：利用 s 参数动态调整圆点尺寸 (Size)
    # 通过将权重乘以一个放大系数（比如 50 或 100），让非零元在图上直接变成清晰可见的粗大圆点！
    # c=weights 配合 cmap 还可以让大元素颜色更鲜艳
    pixel_sizes = weights * 80  # 🌟 调整这个乘数，乘数越大，非零元的点就会变得越粗、越大！
    
    scatter = ax.scatter(cols, rows, s=pixel_sizes, c=weights, cmap='plasma', alpha=0.8, edgecolors='none')
    
    # 添加色带
    cb = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Edge Weight Strength', fontsize=11)
    
    # 5. 画出 8 个分区的红/黑分割边界虚线
    for line in boundary_lines[:-1]:
        ax.axhline(y=line, color='#333333', linestyle='-', linewidth=2.5, alpha=0.9)
        ax.axvline(x=line, color='#333333', linestyle='-', linewidth=2.5, alpha=0.9)
        
    # 6. 视觉修正：因为 matplotlib 坐标轴默认方向，需要反转 Y 轴让矩阵从左上角(0,0)开始
    ax.set_ylim(total_nodes, 0)
    ax.set_xlim(0, total_nodes)
    
    ax.set_title("Highlighted RSB Matrix Layout (Large Non-Zero Pixels)", pad=20, fontsize=13, fontweight='bold')
    ax.set_xlabel("Reordered Node ID", fontsize=11)
    ax.set_ylabel("Reordered Node ID", fontsize=11)
    plt.show()
    
def test_partition():
    # 稠密矩阵
    K_PARTITIONS = 8
    grid_size=64
    #dense_adj = create_dense_mesh_graph(grid_size)
    dense_adj = np.loadtxt('fipchip_Ldense.dat')
    np.fill_diagonal(dense_adj, 0)
    
    total_nodes = dense_adj.shape[0]
    initial_global_nodes = np.arange(total_nodes)
    
    print(f"\n[Start] 启动 1024 阶稠密矩阵 RSB 分割 (k={K_PARTITIONS})...")
    start_time = time.time()
    
    # 执行算法
    result_clusters = rsb_dense(dense_adj, initial_global_nodes, k=K_PARTITIONS)
    
    end_time = time.time()
    print(f"[Success] 稠密矩阵分割完成！总耗时: {end_time - start_time:.4f} 秒")
    
    # 打印每个分区的结果评估
    print("\n[Analysis] 划分平衡度统计:")
    for i, cluster in enumerate(result_clusters):
        print(f"  - 聚类块 {i+1} : 节点数 = {len(cluster)} (节点编号前5个: {cluster[:5]} ...)")


    plot_block_matrix_with_large_pixels(dense_adj*1e9, result_clusters, total_nodes, 1e-15)
