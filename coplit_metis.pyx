# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from scipy.sparse import issparse
cimport metis as mts

def node_nd(A):
    """
    使用 METIS 的嵌套剖分算法（NodeND）对图/稀疏矩阵进行重排序。
    输入 A 必须为一个方阵（通常是对称的稀疏矩阵，不包含对角线自身）。
    返回：
        perm:  排列向量 (用于将矩阵 A 转换为 P * A * P^T)
        iperm: 逆排列向量
    """
    if not issparse(A):
        raise TypeError("Input matrix A must be a scipy sparse matrix.")
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")

    # 1. 获取顶点数 (矩阵的阶数)
    cdef mts.idx_t nvtxs = A.shape[0]
    if nvtxs == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # 2. 转换矩阵格式以获取图的邻接结构
    # METIS 要求输入的图不包含自环（即不包含对角线元素）
    # 强烈建议在传入前执行：A = A - diag(diag(A))
    A_csr = A.tocsr()

    # 3. 严格转换为 C 连续的 int32 数组，确保内存对齐与 idx_t 一致
    cdef cnp.ndarray[mts.idx_t, ndim=1, mode="c"] xadj = np.asarray(A_csr.indptr, dtype=np.int32)
    cdef cnp.ndarray[mts.idx_t, ndim=1, mode="c"] adjncy = np.asarray(A_csr.indices, dtype=np.int32)

    # 4. 分配存储返回结果的 NumPy 内存空间
    cdef cnp.ndarray[mts.idx_t, ndim=1, mode="c"] perm = np.zeros(nvtxs, dtype=np.int32)
    cdef cnp.ndarray[mts.idx_t, ndim=1, mode="c"] iperm = np.zeros(nvtxs, dtype=np.int32)

    # 5. 调用底层 METIS C 接口
    # 对于可选参数 vwgt (顶点权重) 和 options (配置项)，直接传递 NULL 指针使用默认配置
    cdef mts.idx_t status = mts.METIS_NodeND(
        &nvtxs,
        &xadj[0],
        &adjncy[0],
        NULL,       # vwgt = NULL
        NULL,       # options = NULL
        &perm[0],
        &iperm[0]
    )

    # METIS_OK 的常量定义通常为 1
    if status != 1:
        raise RuntimeError(f"METIS_NodeND failed with status code: {status}")

    return perm, iperm
