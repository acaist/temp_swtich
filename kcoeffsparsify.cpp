KcoeffResult KcoeffSparsifier::sparsifyJoint(
    Matrix& Acmp,
    const Matrix& As,
    const KcoeffConfig& cfg) const 
{
    KcoeffResult result;
    const std::size_t n = Acmp.rows();
    if (n == 0) return result;

    if (As.rows() != n || As.cols() != n) {
        throw std::invalid_argument("Matrix dimensions of Acmp and As must match.");
    }

    // 1. 对角线能量叠加处理
    std::vector<double> diagSqrt(n);
    for (std::size_t i = 0; i < n; ++i) {
        double d_total = 0.0;
        if (Acmp(i, i) > 0.0) {
            d_total += Acmp(i, i);
        }
        if (As(i, i) > 0.0) {
            d_total += As(i, i);
        }

        if (d_total < 0.0) {
            throw std::runtime_error("sparsifyJoint: non-positive total diagonal element at index " + std::to_string(i));
        }
        diagSqrt[i] = std::sqrt(d_total);
    }

    // 清空下三角区域，为写入稀疏化的 k 值做准备
    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            Acmp(i, j) = 0.0;
        }
    }

    // 初始化 Gershgorin 过滤的行列和
    std::vector<double> diagSum(n, 0.0);
    const double EPS = 1e-30;

    // 2. 第一步：全盘强行保留 As 所在的边（叠加共存，提前扣除圆盘额度）
    // 访问对称的上三角数据
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i+1; j < n; ++j) {
            // 如果 As 在这里有值，则属于“必须保留”的特权区域
            if (std::fabs(As(i, j)) > cfg.noiseFloor) {
                // 叠加 Acmp 和 As 的值作为总物理强度
                const double totalValue = Acmp(i, j) + As(i, j);
                if (!std::isfinite(totalValue)) {
                    throw std::runtime_error("sparsifyJoint: non-finite matrix element");
                }

                // 直接计算归一化耦合系数
                const double k = totalValue / (diagSqrt[i] * diagSqrt[j] + EPS);
                const double ak = std::fabs(k);

                // 强制写入，不接受 Gershgorin 过滤，直接扣除圆盘额度
                Acmp(j, i) = k; //写入下三角位置
                diagSum[i] += ak;
                diagSum[j] += ak;

                result.edges.push_back(KcoeffEdge{i, j, k});
            }
        }
    }

    // 3. 第二步：收集那些【没有 As 覆盖】的纯 Acmp 候选边进行容量竞争
    const std::size_t cap = cfg.maxEdges == 0 ? n : cfg.maxEdges;
    std::priority_queue<Candidate, std::vector<Candidate>, MinCandidate> heap;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i+1; j < n; ++j) {
            // 如果对称下三角位置被 As填充过，第一步已经处理过了，纯 Acmp 竞争直接跳过
            if (fabs(Acmp(j, i)) > 0.0) {
                continue; 
            }

            // 纯 Acmp 的边放入堆中竞争容量
            ++result.candidates;
            heap.push(Candidate{std::fabs(Acmp(i, j)), i * n + j, Acmp(i, j)});
            
            if (heap.size() > cap) {
                heap.pop();
                ++result.droppedByCapacity;
            }
        }
    }

    // 4. 提取通过容量筛选的纯 Acmp 候选边
    std::vector<Candidate> kept;
    kept.reserve(heap.size());
    while (!heap.empty()) {
        kept.push_back(heap.top());
        heap.pop();
    }

    // 6. 从强到弱处理纯 Acmp 存活下来的补充边（使用受 As 消耗后的残余圆盘额度）
    for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
        const Candidate& c = *it;
        const std::size_t i = c.flat / n;
        const std::size_t j = c.flat % n;
        
        // 此时 c.value 纯粹是 Acmp(i,j) 的值
        const double k = c.value / (diagSqrt[i] * diagSqrt[j] + EPS);
        const double ak = std::fabs(k);

        // 统一进行对角占优（Gershgorin）过滤，此时 rowSum/colSum 已经包含了 As 的特权基底
        if (ak + diagSum[i] < 1.0 && ak + diagSum[j] < 1.0) {
            Acmp(j, i) = k;  // 写入下三角
            diagSum[i] += ak;
            diagSum[j] += ak;
            result.edges.push_back(KcoeffEdge{i, j, k});
        } else {
            ++result.droppedByDominance;
        }
    }

    // 7. 最终将融合后的总对角线值同步回 Acmp，确保自回路能量与非对角线归一化分母保持数学一致
    for (std::size_t i = 0; i < n; ++i) {
        Acmp(i, i) = diagSqrt[i]*diagSqrt[i];
    }

    return result;
}
