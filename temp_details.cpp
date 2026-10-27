
// ---- 纯离散（档位索引向量）工具 ----------------------------------------

// 索引向量 → 表型（按显式类型分派; 越界索引按 clampIndex 钳制）
vector<VarValue> decodeDiscIndices(const vector<VarSpec>& specs, const vector<int>& indices)
{
    vector<VarValue> pheno;
    pheno.reserve(specs.size());

    for (size_t i = 0; i < specs.size(); ++i) {
        const VarSpec& sp = specs[i];
        VarValue vv;
        vv.type = sp.type;
        vv.name = sp.name;

        const int raw = (i < indices.size()) ? indices[i] : 0;
        const int idx = clampIndex(raw, sp.listSize());
        switch (sp.type) {
            case VarType::DISC_DOUBLE: vv.val_dou = sp.list_dou[idx]; break;
            case VarType::DISC_INT:    vv.val_int = sp.list_int[idx]; break;
            case VarType::DISC_STR:    vv.val_str = sp.list_str[idx]; break;
            default:                   break;   // CONT_DOUBLE 不参与索引编码
        }
        pheno.push_back(vv);
    }
    return pheno;
}

// 表型 → 索引向量（容差匹配, 找不到返回 0）
vector<int> encodeDiscIndices(const vector<VarSpec>& specs, const vector<VarValue>& pheno)
{
    vector<int> idx(specs.size(), 0);
    const size_t n = std::min(specs.size(), pheno.size());
    for (size_t i = 0; i < n; ++i) {
        const VarSpec& sp = specs[i];
        switch (sp.type) {
            case VarType::DISC_DOUBLE: idx[i] = indexOfDouble(sp.list_dou, pheno[i].val_dou); break;
            case VarType::DISC_INT:    idx[i] = indexOfInt(sp.list_int, pheno[i].val_int); break;
            case VarType::DISC_STR:    idx[i] = indexOfString(sp.list_str, pheno[i].val_str); break;
            default:                   break;
        }
    }
    return idx;
}

// 离散空间规模（达到 cap 立即返回 cap）
double discSpaceSize(const vector<VarSpec>& specs, double cap)
{
    double space = 1.0;
    for (size_t i = 0; i < specs.size(); ++i) {
        if (!specs[i].isDiscrete())
            continue;
        space *= static_cast<double>(specs[i].listSize());
        if (space >= cap)
            return cap;
    }
    return space;
}

// 结果打印: 各算法统一格式（原 GAMixed::optimize 末尾打印块抽出）
void printBestPara(OptimStatus st, const vector<VarValue>& best, double cost)
{
    std::cout << "\n=== Best Solution ===\n";
    for (size_t i = 0; i < best.size(); ++i) {
        const VarValue& v = best[i];
        std::cout << "Var " << i << " (" << v.name << ") -> ";
        switch (v.type) {
            case VarType::CONT_DOUBLE:
            case VarType::DISC_DOUBLE:
                std::cout << "Value " << v.val_dou << "\n";
                break;
            case VarType::DISC_INT:
                std::cout << "Value " << v.val_int << "\n";
                break;
            case VarType::DISC_STR:
                std::cout << "Value " << v.val_str << "\n";
                break;
        }
    }
    std::cout << "Minimum Cost: " << cost << "\n";
    std::cout << "Status: " << static_cast<int>(st) << "\n";
}
