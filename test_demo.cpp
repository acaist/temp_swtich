#include "MixoptGA.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << msg << " (" << #cond << ")" << std::endl; \
            std::exit(1); \
        } \
    } while(0)

namespace OM = OptMix;

// ==========================================================================
// 测试辅助
// ==========================================================================

// 捕获 optimize() 的控制台输出, 解析 "Minimum Cost: "
struct GAResult {
    double      best_cost;
    std::string raw_output;
};

static GAResult run_and_capture(OM::GAMixed& ga)
{
    std::stringstream buffer;
    std::streambuf* old_buf = std::cout.rdbuf(buffer.rdbuf());
    ga.optimize();
    std::cout.rdbuf(old_buf);

    GAResult result;
    result.raw_output = buffer.str();

    const std::string prefix = "Minimum Cost: ";
    std::string::size_type pos = result.raw_output.find(prefix);
    if (pos != std::string::npos) {
        pos += prefix.size();
        std::string::size_type end = result.raw_output.find('\n', pos);
        result.best_cost = std::atof(result.raw_output.substr(pos, end - pos).c_str());
    } else {
        result.best_cost = -1.0;
    }
    return result;
}

// ---- 入参构造辅助 ----

static OM::VarInput makeContVar(const std::string& name, double lo, double hi)
{
    return OM::VarInput(name, "", std::make_pair(lo, hi));
}

static OM::VarInput makeFormatVar(const std::string& name, const std::string& fmt, double lo, double hi)
{
    return OM::VarInput(name, fmt, std::make_pair(lo, hi));
}

static OM::VarInput makeNumListVar(const std::string& name, const std::vector<double>& vals,
                                   double lo, double hi)
{
    return OM::VarInput(name, "list", std::make_pair(lo, hi), vals);
}

static OM::VarInput makeIntListVar(const std::string& name, const std::vector<int>& vals)
{
    double lo = 0.0, hi = 0.0;
    if (!vals.empty()) {
        lo = static_cast<double>(vals[0]);
        hi = static_cast<double>(vals[0]);
        for (size_t i = 1; i < vals.size(); ++i) {
            if (static_cast<double>(vals[i]) < lo) lo = static_cast<double>(vals[i]);
            if (static_cast<double>(vals[i]) > hi) hi = static_cast<double>(vals[i]);
        }
    }
    return OM::VarInput(name, "list", std::make_pair(lo, hi), std::vector<double>(), vals);
}

static OM::VarInput makeStrVar(const std::string& name, const std::vector<std::string>& vals)
{
    return OM::VarInput(name, "list", std::make_pair(0.0, 0.0),
                        std::vector<double>(), std::vector<int>(), vals);
}

static std::vector<std::string> strList2(const std::string& a, const std::string& b)
{
    std::vector<std::string> v;
    v.push_back(a);
    v.push_back(b);
    return v;
}

static std::vector<std::string> strList3(const std::string& a, const std::string& b, const std::string& c)
{
    std::vector<std::string> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    return v;
}

static std::vector<double> dblList3(double a, double b, double c)
{
    std::vector<double> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    return v;
}

static std::vector<int> intRange(int lo, int hi)
{
    std::vector<int> v;
    for (int i = lo; i <= hi; ++i)
        v.push_back(i);
    return v;
}

// ==========================================================================
// Test 1: 纯连续问题（Sphere）
// ==========================================================================
static void test_continuous_only()
{
    std::cout << "[Test 1] Continuous only (Sphere)\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x1", -5.0, 5.0));
    inputs.push_back(makeContVar("x2", -5.0, 5.0));

    auto sphere = [](const std::vector<OM::VarValue>& v) -> double {
        return v[0].val_dou * v[0].val_dou + v[1].val_dou * v[1].val_dou;
    };

    OM::GAConfig cfg(50, 200, 0.8, 0.1);
    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(sphere), inputs, cfg);
    GAResult r = run_and_capture(ga);

    std::cout << "  best cost = " << r.best_cost << "\n";
    TEST_ASSERT(r.best_cost >= 0.0, "cost >= 0");
    TEST_ASSERT(r.best_cost < 0.5, "cost < 0.5");
    TEST_ASSERT(r.raw_output.find("Best Solution") != std::string::npos, "output contains best solution");
    TEST_ASSERT(ga.getContGeneNum() == 2 && ga.getDiscGeneNum() == 0, "2 continuous variables");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 2: 纯离散字符串变量（自动穷举）
// ==========================================================================
static void test_discrete_only()
{
    std::cout << "[Test 2] Discrete only (string enums, exhaustion)\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeStrVar("model", strList3("A", "B", "C")));
    inputs.push_back(makeStrVar("color", strList3("Red", "Green", "Blue")));

    auto cost = [](const std::vector<OM::VarValue>& v) -> double {
        double c = 0.0;
        if (v[0].val_str != "A") c += 10.0;
        if (v[1].val_str != "Blue") c += 5.0;
        return c;
    };

    OM::GAConfig cfg(30, 100, 0.8, 0.2);
    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
    GAResult r = run_and_capture(ga);

    std::cout << "  best cost = " << r.best_cost << "\n";
    TEST_ASSERT(r.best_cost == 0.0, "global optimum 0");
    TEST_ASSERT(ga.getStatus() == OM::OptimStatus::SUCCESS, "status SUCCESS");
    TEST_ASSERT(ga.getDiscGeneNum() == 2 && ga.getContGeneNum() == 0, "2 discrete variables");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 3: 混合变量（连续 + 字符串）
// ==========================================================================
static void test_mixed_variables()
{
    std::cout << "[Test 3] Mixed variables (continuous + string)\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x", -3.0, 3.0));
    inputs.push_back(makeStrVar("mode", strList2("fast", "slow")));

    auto cost = [](const std::vector<OM::VarValue>& v) -> double {
        const double penalty = (v[1].val_str == "fast") ? 0.0 : 2.0;
        return v[0].val_dou * v[0].val_dou + penalty;
    };

    OM::GAConfig cfg(40, 150, 0.8, 0.1);
    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
    GAResult r = run_and_capture(ga);

    std::cout << "  best cost = " << r.best_cost << "\n";
    TEST_ASSERT(r.best_cost >= 0.0, "cost >= 0");
    TEST_ASSERT(r.best_cost < 0.5, "cost < 0.5");
    TEST_ASSERT(ga.getContGeneNum() == 1 && ga.getDiscGeneNum() == 1, "1 cont + 1 disc");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 4: 边界情况
// ==========================================================================
static void test_edge_cases()
{
    std::cout << "[Test 4] Edge cases\n";

    // 4a: 0 变量
    std::cout << "  4a: no variables\n";
    {
        std::vector<OM::VarInput> inputs;   // 空
        auto const_cost = [](const std::vector<OM::VarValue>&) -> double { return 42.0; };
        OM::GAConfig cfg(10, 10, 0.8, 0.1);
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(const_cost), inputs, cfg);
        GAResult r = run_and_capture(ga);
        TEST_ASSERT(r.best_cost == 42.0, "constant cost 42");
        TEST_ASSERT(ga.getStatus() == OM::OptimStatus::NO_IMPROVEMENT, "42 > desired_err");
    }
    std::cout << "      PASSED\n";

    // 4b: 单个连续变量
    std::cout << "  4b: single continuous\n";
    {
        std::vector<OM::VarInput> inputs;
        inputs.push_back(makeContVar("x", -1.0, 1.0));
        auto cost = [](const std::vector<OM::VarValue>& v) -> double { return v[0].val_dou * v[0].val_dou; };
        OM::GAConfig cfg(20, 50, 0.8, 0.1);
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
        GAResult r = run_and_capture(ga);
        TEST_ASSERT(r.best_cost >= 0.0, "cost >= 0");
        std::cout << "      best cost = " << r.best_cost << " PASSED\n";
    }

    // 4c: 单个字符串变量（穷举）
    std::cout << "  4c: single string\n";
    {
        std::vector<OM::VarInput> inputs;
        inputs.push_back(makeStrVar("c", strList3("opt", "sub", "bad")));
        auto cost = [](const std::vector<OM::VarValue>& v) -> double {
            return (v[0].val_str == "opt") ? 0.0 : 10.0;
        };
        OM::GAConfig cfg(20, 50, 0.8, 0.2);
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
        GAResult r = run_and_capture(ga);
        TEST_ASSERT(r.best_cost == 0.0, "global optimum 0");
        std::cout << "      best cost = " << r.best_cost << " PASSED\n";
    }

    // 4d: 单个整数离散变量（穷举）
    std::cout << "  4d: single integer list\n";
    {
        std::vector<OM::VarInput> inputs;
        inputs.push_back(makeIntListVar("n", intRange(10, 50)));
        auto cost = [](const std::vector<OM::VarValue>& v) -> double {
            return (v[0].val_int == 42) ? 0.0 : 10.0;
        };
        OM::GAConfig cfg(20, 50, 0.8, 0.2);
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
        GAResult r = run_and_capture(ga);
        TEST_ASSERT(r.best_cost == 0.0, "global optimum 0");
        TEST_ASSERT(ga.getBestPara()[0].val_int == 42, "val_int == 42");
        TEST_ASSERT(ga.getBestPara()[0].type == OM::VarType::DISC_INT, "type DISC_INT");
        std::cout << "      best cost = " << r.best_cost << " PASSED\n\n";
    }
}

// ==========================================================================
// Test 5: 逐代回调（flag 刷新契约）
// ==========================================================================
static void test_mutual_ui()
{
    std::cout << "[Test 5] Per-generation callback (flag refresh)\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x1", -5.0, 5.0));
    inputs.push_back(makeContVar("x2", -5.0, 5.0));
    auto sphere = [](const std::vector<OM::VarValue>& v) -> double {
        return v[0].val_dou * v[0].val_dou + v[1].val_dou * v[1].val_dou;
    };

    OM::GAConfig cfg(30, 10, 0.8, 0.1);
    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(sphere), inputs, cfg);

    int calls = 0;
    double last_cost = -1.0;
    int last_gen = -1;
    ga.setUpdateres([&](int igen, std::vector<OM::VarValue>&, double cost) -> bool {
        ++calls;
        last_gen = igen;
        last_cost = cost;
        return true;
    });

    ga.optimize();
    const std::vector<double> traces = ga.getTraces();

    std::cout << "  calls = " << calls << " (traces=" << traces.size()
              << ", last_gen=" << last_gen << ", last_cost=" << last_cost << ")\n";
    TEST_ASSERT(calls >= 1, "callback called at least once");
    TEST_ASSERT(calls == static_cast<int>(traces.size()), "calls == traces size");
    TEST_ASSERT(calls <= 1 + static_cast<int>(cfg.ng_max), "calls <= 1 + ng_max");
    TEST_ASSERT(last_cost >= 0.0, "last cost >= 0");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 6: 提前终止（desired_err）
// ==========================================================================
static void test_early_termination()
{
    std::cout << "[Test 6] Early termination via desired_err\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x1", -5.0, 5.0));
    inputs.push_back(makeContVar("x2", -5.0, 5.0));
    auto sphere = [](const std::vector<OM::VarValue>& v) -> double {
        return v[0].val_dou * v[0].val_dou + v[1].val_dou * v[1].val_dou;
    };

    // 6a: 宽松目标 → 初代评估即停
    std::cout << "  6a: loose desired_err = 100\n";
    {
        OM::GAConfig cfg(50, 200, 0.8, 0.1, 0.01, 100.0);
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(sphere), inputs, cfg);
        GAResult r = run_and_capture(ga);
        TEST_ASSERT(ga.getTraces().size() == 1, "traces == 1");
        TEST_ASSERT(ga.getStatus() == OM::OptimStatus::SUCCESS, "status SUCCESS");
        TEST_ASSERT(r.best_cost < 100.0, "cost < 100");
        std::cout << "      PASSED\n";
    }

    // 6b: 目标不可达 + 关闭停滞检测 → 跑满代数
    std::cout << "  6b: unreachable target, adv_stop off\n";
    {
        OM::GAConfig cfg(50, 200, 0.8, 0.1, 0.01, 0.0, false);
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(sphere), inputs, cfg);
        run_and_capture(ga);
        TEST_ASSERT(ga.getTraces().size() == 1 + 200, "traces == 201");
        TEST_ASSERT(ga.getStatus() == OM::OptimStatus::LOCAL_MIN, "status LOCAL_MIN");
        std::cout << "      PASSED\n";
    }

    // 6c: 短运行
    std::cout << "  6c: unreachable target, short run\n";
    {
        OM::GAConfig cfg(50, 10, 0.8, 0.1, 0.01, 0.0, false);
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(sphere), inputs, cfg);
        run_and_capture(ga);
        TEST_ASSERT(ga.getTraces().size() == 1 + 10, "traces == 11");
        std::cout << "      PASSED\n\n";
    }
}

// ==========================================================================
// Test 7: 穷举分支（纯离散小空间）
// ==========================================================================
static void test_exhaustion()
{
    std::cout << "[Test 7] Exhaustion branch (small discrete space)\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeIntListVar("a", intRange(0, 4)));
    inputs.push_back(makeIntListVar("b", intRange(0, 4)));
    OM::GAConfig cfg(30, 10, 0.8, 0.1);   // 25 < 1.5*30*10 → 穷举

    // 7a: 命中全局最优
    std::cout << "  7a: reachable optimum (3,2)\n";
    {
        auto cost = [](const std::vector<OM::VarValue>& v) -> double {
            return std::fabs(static_cast<double>(v[0].val_int - 3))
                 + std::fabs(static_cast<double>(v[1].val_int - 2));
        };
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
        GAResult r = run_and_capture(ga);
        const std::vector<OM::VarValue> best = ga.getBestPara();
        TEST_ASSERT(best[0].val_int == 3 && best[1].val_int == 2, "global optimum (3,2)");
        TEST_ASSERT(r.best_cost == 0.0, "cost == 0");
        TEST_ASSERT(ga.getStatus() == OM::OptimStatus::SUCCESS, "status SUCCESS");
        std::cout << "      PASSED\n";
    }

    // 7b: 目标不可达 → NO_IMPROVEMENT（穷举证明全局最优）
    std::cout << "  7b: unreachable target -> NO_IMPROVEMENT\n";
    {
        auto cost = [](const std::vector<OM::VarValue>& v) -> double {
            return 1.0 + std::fabs(static_cast<double>(v[0].val_int - 3));
        };
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
        GAResult r = run_and_capture(ga);
        TEST_ASSERT(ga.getStatus() == OM::OptimStatus::NO_IMPROVEMENT, "status NO_IMPROVEMENT");
        TEST_ASSERT(std::fabs(r.best_cost - 1.0) < 1e-12, "exact global optimum 1.0");
        std::cout << "      PASSED\n";
    }

    // 7c: 批量评估 + 穷举
    std::cout << "  7c: batch evaluator + exhaustion\n";
    {
        auto costs = [](const std::vector<std::vector<OM::VarValue> >& pop) -> std::vector<double> {
            std::vector<double> out;
            out.reserve(pop.size());
            for (size_t i = 0; i < pop.size(); ++i) {
                out.push_back(std::fabs(static_cast<double>(pop[i][0].val_int - 3))
                            + std::fabs(static_cast<double>(pop[i][1].val_int - 2)));
            }
            return out;
        };
        OM::GAMixed ga(std::make_shared<OM::ParallelEvaluator>(costs), inputs, cfg);
        ga.optimize();
        const std::vector<OM::VarValue> best = ga.getBestPara();
        TEST_ASSERT(best[0].val_int == 3 && best[1].val_int == 2, "global optimum (3,2)");
        TEST_ASSERT(ga.getStatus() == OM::OptimStatus::SUCCESS, "status SUCCESS");
        std::cout << "      PASSED\n\n";
    }
}

// ==========================================================================
// Test 8: 中止机制（前端终止）
// ==========================================================================
static void test_abort()
{
    std::cout << "[Test 8] Abort mechanism\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x1", -5.0, 5.0));
    inputs.push_back(makeContVar("x2", -5.0, 5.0));
    OM::GAConfig cfg(50, 200, 0.8, 0.1);

    OM::GAMixed* ga_ptr = NULL;
    int eval_count = 0;
    auto cost = [&](const std::vector<OM::VarValue>& v) -> double {
        if (++eval_count >= 30 && ga_ptr != NULL)
            ga_ptr->requestAbort();
        return v[0].val_dou * v[0].val_dou + v[1].val_dou * v[1].val_dou;
    };

    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
    ga_ptr = &ga;
    ga.optimize();

    std::cout << "  status = " << static_cast<int>(ga.getStatus())
              << " (ABORTED=-1), traces = " << ga.getTraces().size() << "\n";
    TEST_ASSERT(ga.getStatus() == OM::OptimStatus::ABORTED, "status ABORTED");
    TEST_ASSERT(ga.getTraces().size() < 1 + cfg.ng_max, "abort before ng_max");
    TEST_ASSERT(ga.getTraces().size() <= 1, "abort during initial evaluation");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 9: 初始解注入（热启动）
// ==========================================================================
static void test_seeding()
{
    std::cout << "[Test 9] Initial solution seeding\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x1", -5.0, 5.0));
    inputs.push_back(makeContVar("x2", -5.0, 5.0));
    auto sphere = [](const std::vector<OM::VarValue>& v) -> double {
        return v[0].val_dou * v[0].val_dou + v[1].val_dou * v[1].val_dou;
    };
    OM::GAConfig cfg(50, 200, 0.8, 0.1);

    // 9a: 注入全局最优 → 初代即成功
    std::cout << "  9a: valid seed (0,0)\n";
    {
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(sphere), inputs, cfg);
        std::vector<OM::VarValue> init;
        init.push_back(OM::VarValue(OM::VarType::CONT_DOUBLE, 0.0, "x1"));
        init.push_back(OM::VarValue(OM::VarType::CONT_DOUBLE, 0.0, "x2"));
        ga.setInitialSolution(init);
        TEST_ASSERT(ga.hasInitialSolution(), "seed accepted");
        ga.optimize();
        TEST_ASSERT(ga.getStatus() == OM::OptimStatus::SUCCESS, "status SUCCESS");
        TEST_ASSERT(ga.getTraces().size() == 1, "traces == 1");
        std::cout << "      PASSED\n";
    }

    // 9b: 非法种子（越界）→ 拒绝并正常运行
    std::cout << "  9b: invalid seed (out of range)\n";
    {
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(sphere), inputs, cfg);
        std::vector<OM::VarValue> bad;
        bad.push_back(OM::VarValue(OM::VarType::CONT_DOUBLE, 99.0, "x1"));
        bad.push_back(OM::VarValue(OM::VarType::CONT_DOUBLE, 0.0, "x2"));
        ga.setInitialSolution(bad);
        TEST_ASSERT(!ga.hasInitialSolution(), "seed rejected");
        ga.optimize();
        TEST_ASSERT(ga.getStatus() == OM::OptimStatus::SUCCESS
                    || ga.getStatus() == OM::OptimStatus::LOCAL_MIN, "run proceeds");
        std::cout << "      PASSED\n\n";
    }
}

// ==========================================================================
// Test 10: 停滞检测（提前终止）
// ==========================================================================
static void test_adv_stop()
{
    std::cout << "[Test 10] Advance stop (stagnation)\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x", -1.0, 1.0));
    auto flat = [](const std::vector<OM::VarValue>&) -> double { return 5.0; };

    // num_small_improve=3 → 连续 3 代无改进则停
    OM::GAConfig cfg(20, 50, 0.8, 0.1, 0.01, 1e-6, true, 1e-3, 3);
    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(flat), inputs, cfg);
    ga.optimize();

    std::cout << "  traces = " << ga.getTraces().size() << " (expect 4)\n";
    TEST_ASSERT(ga.getTraces().size() == 4, "traces == 4");
    TEST_ASSERT(ga.getTraces().size() < 1 + cfg.ng_max, "stopped before ng_max");
    TEST_ASSERT(ga.getStatus() == OM::OptimStatus::LOCAL_MIN, "status LOCAL_MIN");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 11: 可复现性（同 seed）
// ==========================================================================
static void test_determinism()
{
    std::cout << "[Test 11] Determinism (same seed)\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x1", -5.0, 5.0));
    inputs.push_back(makeContVar("x2", -5.0, 5.0));
    auto sphere = [](const std::vector<OM::VarValue>& v) -> double {
        return v[0].val_dou * v[0].val_dou + v[1].val_dou * v[1].val_dou;
    };
    OM::GAConfig cfg(50, 50, 0.8, 0.1);

    OM::GAMixed ga1(std::make_shared<OM::SerialEvaluator>(sphere), inputs, cfg);
    OM::GAMixed ga2(std::make_shared<OM::SerialEvaluator>(sphere), inputs, cfg);
    ga1.optimize();
    ga2.optimize();

    TEST_ASSERT(ga1.getTraces() == ga2.getTraces(), "identical traces");
    TEST_ASSERT(ga1.getStatus() == ga2.getStatus(), "identical status");
    std::cout << "  traces = " << ga1.getTraces().size() << " PASSED\n\n";
}

// ==========================================================================
// Test 12: 边界检测
// ==========================================================================
static void test_boundary()
{
    std::cout << "[Test 12] Boundary detection\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x", -5.0, 5.0));
    auto cost = [](const std::vector<OM::VarValue>& v) -> double {
        const double d = v[0].val_dou + 5.0;
        return d * d + 1.0;   // 全局最优在边界 x=-5, cost=1 > desired_err
    };
    OM::GAConfig cfg(30, 100, 0.8, 0.1);

    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
    std::vector<OM::VarValue> init;
    init.push_back(OM::VarValue(OM::VarType::CONT_DOUBLE, -5.0, "x"));
    ga.setInitialSolution(init);
    ga.optimize();

    std::cout << "  status = " << static_cast<int>(ga.getStatus()) << " (BOUNDARY=2)\n";
    TEST_ASSERT(ga.getStatus() == OM::OptimStatus::BOUNDARY, "status BOUNDARY");
    TEST_ASSERT(ga.getBestPara()[0].val_dou == -5.0, "best at lower bound");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 13: 自定义评估器注入（算法端零改动）
// ==========================================================================
struct CountingEvaluator : public OM::Evaluator
{
    std::shared_ptr<OM::SerialEvaluator> inner;
    int batch_calls;
    std::size_t total_individuals;

    explicit CountingEvaluator(OM::SerialEvaluator::FuncType f)
        : inner(new OM::SerialEvaluator(f)), batch_calls(0), total_individuals(0) {}

    std::vector<double> evaluate(const std::vector<std::vector<OM::VarValue> >& pop)
    {
        ++batch_calls;
        total_individuals += pop.size();
        return inner->evaluate(pop);
    }
    bool isBatch() const { return false; }
};

static void test_custom_evaluator_injection()
{
    std::cout << "[Test 13] Custom evaluator injection\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x1", -5.0, 5.0));
    inputs.push_back(makeContVar("x2", -5.0, 5.0));
    auto sphere = [](const std::vector<OM::VarValue>& v) -> double {
        return v[0].val_dou * v[0].val_dou + v[1].val_dou * v[1].val_dou;
    };
    OM::GAConfig cfg(30, 20, 0.8, 0.1);

    std::shared_ptr<CountingEvaluator> eval(new CountingEvaluator(sphere));
    OM::GAMixed ga(eval, inputs, cfg);
    ga.optimize();

    std::cout << "  evaluate calls = " << eval->batch_calls
              << ", individuals = " << eval->total_individuals << "\n";
    TEST_ASSERT(eval->batch_calls >= 1, "custom evaluator used");
    TEST_ASSERT(eval->total_individuals >= 30, "population evaluated");
    TEST_ASSERT(ga.getTraces().back() >= 0.0, "cost >= 0");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 14: 串行 / 批量评估器等价性
// ==========================================================================
static void test_evaluator_equivalence()
{
    std::cout << "[Test 14] Serial vs Parallel equivalence\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x1", -5.0, 5.0));
    inputs.push_back(makeContVar("x2", -5.0, 5.0));

    auto sphere = [](const std::vector<OM::VarValue>& v) -> double {
        return v[0].val_dou * v[0].val_dou + v[1].val_dou * v[1].val_dou;
    };
    auto sphere_batch = [](const std::vector<std::vector<OM::VarValue> >& pop) -> std::vector<double> {
        std::vector<double> out;
        out.reserve(pop.size());
        for (size_t i = 0; i < pop.size(); ++i)
            out.push_back(pop[i][0].val_dou * pop[i][0].val_dou
                        + pop[i][1].val_dou * pop[i][1].val_dou);
        return out;
    };

    // desired_err=0 且关闭停滞检测 → 两者评估序列一致
    OM::GAConfig cfg(30, 30, 0.8, 0.1, 0.01, 0.0, false);
    OM::GAMixed ga1(std::make_shared<OM::SerialEvaluator>(sphere), inputs, cfg);
    OM::GAMixed ga2(std::make_shared<OM::ParallelEvaluator>(sphere_batch), inputs, cfg);
    ga1.optimize();
    ga2.optimize();

    std::cout << "  serial traces = " << ga1.getTraces().size()
              << ", parallel traces = " << ga2.getTraces().size() << "\n";
    TEST_ASSERT(ga1.getTraces() == ga2.getTraces(), "identical traces");
    TEST_ASSERT(ga1.getStatus() == ga2.getStatus(), "identical status");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 15: step format（固定步长 → DISC_DOUBLE 档位表）
// ==========================================================================
static void test_step_format()
{
    std::cout << "[Test 15] step format (DISC_DOUBLE grid)\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeFormatVar("x", "0.5", -1.0, 1.0));   // 档位 {-1,-0.5,0,0.5,1}

    auto cost = [](const std::vector<OM::VarValue>& v) -> double {
        const double d = v[0].val_dou - 0.35;
        return d * d;                          // 最优档位 0.5 → 0.0225
    };
    OM::GAConfig cfg(50, 300, 0.8, 0.1);
    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
    GAResult r = run_and_capture(ga);

    const std::vector<OM::VarValue> best = ga.getBestPara();
    std::cout << "  best = " << best[0].val_dou << ", cost = " << r.best_cost
              << " (expect 0.5 / 0.0225)\n";
    TEST_ASSERT(best[0].type == OM::VarType::DISC_DOUBLE, "type DISC_DOUBLE");
    TEST_ASSERT(std::fabs(best[0].val_dou - 0.5) < 1e-9, "snapped to 0.5");
    TEST_ASSERT(std::fabs(r.best_cost - 0.0225) < 1e-9, "cost == 0.0225");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 16: EIA format（指数系列 → DISC_DOUBLE 档位表）
// ==========================================================================
static void test_eia_format()
{
    std::cout << "[Test 16] EIA format (E-series table)\n";

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeFormatVar("x", "E12", 0.01, 100.0));   // 49 档

    auto cost = [](const std::vector<OM::VarValue>& v) -> double {
        const double d = v[0].val_dou - 0.03;
        return d * d;
    };
    OM::GAConfig cfg(50, 300, 0.8, 0.1);
    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
    GAResult r = run_and_capture(ga);

    const std::vector<OM::VarValue> best = ga.getBestPara();
    const double expect_val  = 0.03162277660168379;   // 10^(-1.5): E12 表中最接近 0.03 的档位
    const double expect_cost = (expect_val - 0.03) * (expect_val - 0.03);

    std::cout << "  best = " << best[0].val_dou << ", cost = " << r.best_cost
              << " (expect " << expect_val << " / " << expect_cost << ")\n";
    TEST_ASSERT(best[0].type == OM::VarType::DISC_DOUBLE, "type DISC_DOUBLE");
    TEST_ASSERT(std::fabs(best[0].val_dou - expect_val) < 1e-9, "best == 0.0316227766");
    TEST_ASSERT(std::fabs(r.best_cost - expect_cost) < 1e-9, "cost == 2.6334e-6");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 17: list format（候选列表 → 离散档位）
// ==========================================================================
static void test_list_format()
{
    std::cout << "[Test 17] list format (candidate values)\n";

    std::vector<double> candidates = dblList3(0.0, 0.25, 0.75);
    candidates.push_back(1.0);   // {0, 0.25, 0.75, 1}

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeNumListVar("x", candidates, 0.0, 1.0));

    auto cost = [](const std::vector<OM::VarValue>& v) -> double {
        const double d = v[0].val_dou - 0.3;
        return d * d;                          // 表内最优 0.25 → 0.0025
    };
    OM::GAConfig cfg(50, 300, 0.8, 0.1);
    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
    GAResult r = run_and_capture(ga);

    const std::vector<OM::VarValue> best = ga.getBestPara();
    std::cout << "  best = " << best[0].val_dou << ", cost = " << r.best_cost
              << " (expect 0.25 / 0.0025)\n";
    TEST_ASSERT(std::fabs(best[0].val_dou - 0.25) < 1e-9, "best == 0.25");
    TEST_ASSERT(std::fabs(r.best_cost - 0.0025) < 1e-9, "cost == 0.0025");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 18: 整数 format（整数 step / vec_para_range → DISC_INT）
// ==========================================================================
static void test_int_format()
{
    std::cout << "[Test 18] integer format (DISC_INT)\n";

    // 18a: format="1" 整数步长
    std::cout << "  18a: format=\"1\" (integer step)\n";
    {
        std::vector<OM::VarInput> inputs;
        inputs.push_back(makeFormatVar("n", "1", 0.0, 3.0));   // 档位 {0,1,2,3}

        const std::vector<OM::VarSpec> specs = OM::detail::buildVarSpecs(inputs);
        TEST_ASSERT(specs.size() == 1, "materialized 1 spec");
        TEST_ASSERT(specs[0].type == OM::VarType::DISC_INT, "type DISC_INT");
        TEST_ASSERT(specs[0].list_int.size() == 4, "table size 4");

        auto cost = [](const std::vector<OM::VarValue>& v) -> double {
            const double d = static_cast<double>(v[0].val_int - 2);
            return d * d;
        };
        OM::GAConfig cfg(30, 50, 0.8, 0.1);
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
        GAResult r = run_and_capture(ga);
        TEST_ASSERT(ga.getBestPara()[0].val_int == 2, "best == 2");
        TEST_ASSERT(r.best_cost == 0.0, "cost == 0");
        std::cout << "      PASSED\n";
    }

    // 18b: vec_para_range 索引离散
    std::cout << "  18b: vec_para_range\n";
    {
        std::vector<std::vector<int> > para_range;
        para_range.push_back(intRange(0, 3));
        para_range.push_back(intRange(3, 6));

        const std::vector<OM::VarInput> inputs = OM::detail::inputsFromIndexRange(para_range);
        TEST_ASSERT(inputs.size() == 2, "2 variables");

        const std::vector<OM::VarSpec> specs = OM::detail::buildVarSpecs(inputs);
        TEST_ASSERT(specs.size() == 2, "materialized 2 specs");
        TEST_ASSERT(specs[0].type == OM::VarType::DISC_INT
                    && specs[1].type == OM::VarType::DISC_INT, "both DISC_INT");

        auto cost = [](const std::vector<OM::VarValue>& v) -> double {
            const double d0 = static_cast<double>(v[0].val_int - 3);
            const double d1 = static_cast<double>(v[1].val_int - 5);
            return d0 * d0 + d1 * d1;
        };
        OM::GAConfig cfg(30, 50, 0.8, 0.1);
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
        GAResult r = run_and_capture(ga);
        TEST_ASSERT(r.best_cost == 0.0, "cost == 0");

        // 回填: {变量索引, 数值}
        const std::vector<std::pair<int, double> > pairs = OM::detail::toIndexValuePairs(ga.getBestPara());
        TEST_ASSERT(pairs.size() == 2, "2 numeric pairs");
        TEST_ASSERT(pairs[0].second == 3.0 && pairs[1].second == 5.0, "values 3 and 5");
        std::cout << "      best = (" << pairs[0].second << ", " << pairs[1].second
                  << ") PASSED\n\n";
    }
}

// ==========================================================================
// Test 19: 字符串 format（string_data_range → DISC_STR）
// ==========================================================================
static void test_string_format()
{
    std::cout << "[Test 19] string format (DISC_STR)\n";

    std::vector<std::pair<std::string, std::vector<std::string> > > string_data_range;
    string_data_range.push_back(std::make_pair(std::string("L"), strList2("p1", "p2")));
    string_data_range.push_back(std::make_pair(std::string("C"), strList3("p1", "p2", "p3")));

    const std::vector<OM::VarInput> inputs = OM::detail::inputsFromStringRange(string_data_range);
    TEST_ASSERT(inputs.size() == 2, "2 string variables");

    const std::vector<OM::VarSpec> specs = OM::detail::buildVarSpecs(inputs);
    TEST_ASSERT(specs.size() == 2, "materialized 2 specs");
    TEST_ASSERT(specs[0].type == OM::VarType::DISC_STR, "type DISC_STR");
    TEST_ASSERT(specs[0].name == "L" && specs[1].name == "C", "names preserved");

    auto cost = [](const std::vector<OM::VarValue>& v) -> double {
        const bool ok = (v[0].val_str == "p2") && (v[1].val_str == "p3");
        return ok ? 0.0 : 1.0;
    };
    OM::GAConfig cfg(30, 50, 0.8, 0.1);
    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
    GAResult r = run_and_capture(ga);

    const std::vector<OM::VarValue> best = ga.getBestPara();
    TEST_ASSERT(r.best_cost == 0.0, "cost == 0");
    TEST_ASSERT(best[0].val_str == "p2" && best[1].val_str == "p3", "best string values");

    // 回填: {变量名, 取值}
    const std::vector<std::pair<std::string, std::string> > pairs = OM::detail::toStringPairs(best);
    TEST_ASSERT(pairs.size() == 2, "2 string pairs");
    TEST_ASSERT(pairs[0].first == "L" && pairs[0].second == "p2", "pair {L, p2}");
    std::cout << "  L=" << pairs[0].second << ", C=" << pairs[1].second << " PASSED\n\n";
}

// ==========================================================================
// Test 20: 三形态适配器（format / string / index + merge）
// ==========================================================================
static void test_input_adapters()
{
    std::cout << "[Test 20] input adapters (format / string / index)\n";

    // 形态 A: format 驱动
    std::vector<std::string> vec_EIAstep;
    vec_EIAstep.push_back("");      // 连续
    vec_EIAstep.push_back("E4");    // EIA
    vec_EIAstep.push_back("0.1");   // 浮点 step
    vec_EIAstep.push_back("list");  // 候选列表（全整数 → DISC_INT）

    std::vector<std::pair<double, double> > vec_range_Para;
    vec_range_Para.push_back(std::make_pair(-5.0, 5.0));
    vec_range_Para.push_back(std::make_pair(1.0, 150.0));
    vec_range_Para.push_back(std::make_pair(0.0, 1.0));
    vec_range_Para.push_back(std::make_pair(0.0, 3.0));

    std::vector<std::pair<int, std::vector<double> > > vec_list;
    vec_list.push_back(std::make_pair(3, dblList3(1.0, 2.0, 3.0)));

    const std::vector<OM::VarInput> a = OM::detail::inputsFromFormat(vec_EIAstep, vec_range_Para, vec_list);
    TEST_ASSERT(a.size() == 4, "4 inputs from format");
    TEST_ASSERT(std::fabs(a[1].range.second - 150.0) < 1e-12, "range carried");

    const std::vector<OM::VarSpec> specs_a = OM::detail::buildVarSpecs(a);
    TEST_ASSERT(specs_a.size() == 4, "materialized 4 specs");
    TEST_ASSERT(specs_a[0].type == OM::VarType::CONT_DOUBLE, "var0 CONT_DOUBLE");
    TEST_ASSERT(specs_a[1].type == OM::VarType::DISC_DOUBLE, "var1 DISC_DOUBLE (EIA)");
    TEST_ASSERT(specs_a[2].type == OM::VarType::DISC_DOUBLE, "var2 DISC_DOUBLE (step 0.1)");
    TEST_ASSERT(specs_a[3].type == OM::VarType::DISC_INT, "var3 DISC_INT (integer list)");
    TEST_ASSERT(specs_a[1].list_dou.size() == 10, "E4(1,150) table size 10");

    // 形态 B + C + 合并
    std::vector<std::pair<std::string, std::vector<std::string> > > srange;
    srange.push_back(std::make_pair(std::string("L"), strList2("p1", "p2")));
    std::vector<std::vector<int> > prange;
    prange.push_back(intRange(0, 3));

    std::vector<std::vector<OM::VarInput> > groups;
    groups.push_back(OM::detail::inputsFromStringRange(srange));
    groups.push_back(OM::detail::inputsFromIndexRange(prange));
    const std::vector<OM::VarInput> merged = OM::detail::mergeInputs(groups);
    TEST_ASSERT(merged.size() == 2, "merged 2 inputs");
    TEST_ASSERT(merged[0].name == "L", "string variable first");

    const std::vector<OM::VarSpec> specs_m = OM::detail::buildVarSpecs(merged);
    TEST_ASSERT(specs_m.size() == 2, "materialized 2 specs");
    TEST_ASSERT(specs_m[0].type == OM::VarType::DISC_STR, "merged var0 DISC_STR");
    TEST_ASSERT(specs_m[1].type == OM::VarType::DISC_INT, "merged var1 DISC_INT");
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 21: 非法输入 → ERROR
// ==========================================================================
static OM::OptimStatus run_error_case(const std::vector<OM::VarInput>& inputs)
{
    auto cost = [](const std::vector<OM::VarValue>& v) -> double {
        double s = 0.0;
        for (size_t i = 0; i < v.size(); ++i)
            s += v[i].val_dou;
        return s;
    };
    OM::GAConfig cfg(10, 10, 0.8, 0.1);
    OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
    std::stringstream buffer;
    std::streambuf* old_buf = std::cout.rdbuf(buffer.rdbuf());
    ga.optimize();
    std::cout.rdbuf(old_buf);
    return ga.getStatus();
}

static void test_format_error()
{
    std::cout << "[Test 21] invalid inputs -> ERROR\n";

    // 21a: 未知 format
    {
        std::vector<OM::VarInput> in;
        in.push_back(makeFormatVar("x", "unknown", 0.0, 1.0));
        TEST_ASSERT(run_error_case(in) == OM::OptimStatus::ERROR, "21a unknown format -> ERROR");
    }
    // 21b: step = 0
    {
        std::vector<OM::VarInput> in;
        in.push_back(makeFormatVar("x", "0", 0.0, 1.0));
        TEST_ASSERT(run_error_case(in) == OM::OptimStatus::ERROR, "21b step=0 -> ERROR");
    }
    // 21c: EIA 负区间
    {
        std::vector<OM::VarInput> in;
        in.push_back(makeFormatVar("x", "E4", -1.0, 1.0));
        TEST_ASSERT(run_error_case(in) == OM::OptimStatus::ERROR, "21c EIA negative range -> ERROR");
    }
    // 21d: list 无候选
    {
        std::vector<OM::VarInput> in;
        in.push_back(OM::VarInput("x", "list", std::make_pair(0.0, 1.0)));
        TEST_ASSERT(run_error_case(in) == OM::OptimStatus::ERROR, "21d list without candidates -> ERROR");
    }
    // 21e: list 候选全被 range 过滤
    {
        std::vector<OM::VarInput> in;
        in.push_back(makeNumListVar("x", dblList3(1.0, 2.0, 3.0), 10.0, 20.0));
        TEST_ASSERT(run_error_case(in) == OM::OptimStatus::ERROR, "21e all filtered -> ERROR");
    }
    // 21f: 数值与字符串候选同时提供（歧义）
    {
        std::vector<OM::VarInput> in;
        OM::VarInput v("x", "list", std::make_pair(0.0, 3.0));
        v.list_dou = dblList3(1.0, 2.0, 3.0);
        v.list_str = strList2("a", "b");
        in.push_back(v);
        TEST_ASSERT(run_error_case(in) == OM::OptimStatus::ERROR, "21f ambiguous -> ERROR");
    }
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 22: Debug Log（开关 + 文件, 极简实现）
// ==========================================================================
static void test_debug_log()
{
    std::cout << "[Test 22] Debug log (on/off switch)\n";

    const std::string path = "optmix_test_log.txt";
    std::remove(path.c_str());

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x", -1.0, 1.0));
    auto cost = [](const std::vector<OM::VarValue>& v) -> double { return v[0].val_dou * v[0].val_dou; };
    OM::GAConfig cfg(20, 10, 0.8, 0.1);

    // 关闭状态: 不产生日志
    {
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
        ga.optimize();
    }
    TEST_ASSERT(!OM::isLogOn(), "log off by default");
    TEST_ASSERT(std::ifstream(path.c_str()).is_open() == false, "no log file when off");

    // 打开日志（输出到文件）
    OM::setLogFile(path.c_str());
    OM::setLogOn(true);
    {
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
        ga.optimize();
    }
    OM::setLogOn(false);
    TEST_ASSERT(!OM::isLogOn(), "log switched off");

    std::string content;
    std::ifstream ifs(path.c_str());
    if (ifs.is_open()) {
        std::stringstream ss;
        ss << ifs.rdbuf();
        content = ss.str();
        ifs.close();
    }
    std::cout << "  log content size = " << content.size() << " bytes\n";
    TEST_ASSERT(!content.empty(), "log file written when on");
    TEST_ASSERT(content.find("materialize") != std::string::npos, "materialization logged");
    TEST_ASSERT(content.find("gen=") != std::string::npos, "generation statistics logged");
    TEST_ASSERT(content.find("mode dispatch") != std::string::npos, "mode dispatch logged");
    TEST_ASSERT(content.find("optimize finished") != std::string::npos, "finish line logged");

    std::remove(path.c_str());
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 23: Debug Log 追加写（多次运行累积到同一文件）
// ==========================================================================
static void test_log_file()
{
    std::cout << "[Test 23] Debug log append mode\n";

    const std::string path = "optmix_test_log2.txt";
    std::remove(path.c_str());

    std::vector<OM::VarInput> inputs;
    inputs.push_back(makeContVar("x", -1.0, 1.0));
    auto cost = [](const std::vector<OM::VarValue>& v) -> double { return v[0].val_dou * v[0].val_dou; };
    OM::GAConfig cfg(20, 10, 0.8, 0.1);

    OM::setLogFile(path.c_str());

    // 第一次运行
    OM::setLogOn(true);
    {
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
        ga.optimize();
    }
    OM::setLogOn(false);
    const long size1 = static_cast<long>(std::ifstream(path.c_str(), std::ios::binary | std::ios::ate).tellg());

    // 第二次运行（追加）
    OM::setLogOn(true);
    {
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), inputs, cfg);
        ga.optimize();
    }
    OM::setLogOn(false);
    const long size2 = static_cast<long>(std::ifstream(path.c_str(), std::ios::binary | std::ios::ate).tellg());

    std::cout << "  size after run1 = " << size1 << ", after run2 = " << size2 << "\n";
    TEST_ASSERT(size1 > 0, "first run logged");
    TEST_ASSERT(size2 > size1, "second run appended");

    //std::remove(path.c_str());
    std::cout << "  PASSED\n\n";
}

// ==========================================================================
// Test 24: pair vector 透传与回填
// ==========================================================================
static void test_pair_vector_inputs()
{
    std::cout << "[Test 24] pair vector inputs / outputs\n";

    // 24a: 位置型数量不匹配（range 少于 format）→ 装配失败
    std::cout << "  24a: mismatched range count\n";
    {
        std::vector<std::string> fmt;
        fmt.push_back("");
        fmt.push_back("E4");
        std::vector<std::pair<double, double> > rng;
        rng.push_back(std::make_pair(-5.0, 5.0));   // 少一个

        const std::vector<OM::VarInput> in = OM::detail::inputsFromFormat(
            fmt, rng, std::vector<std::pair<int, std::vector<double> > >());
        TEST_ASSERT(in.empty(), "mismatched range count -> empty");
    }
    std::cout << "      PASSED\n";

    // 24b: 键型缺索引 → 装配失败
    std::cout << "  24b: missing key in vec_list\n";
    {
        std::vector<std::string> fmt;
        fmt.push_back("list");
        std::vector<std::pair<double, double> > rng;
        rng.push_back(std::make_pair(0.0, 3.0));
        std::vector<std::pair<int, std::vector<double> > > vec_list;   // 缺索引 0

        const std::vector<OM::VarInput> in = OM::detail::inputsFromFormat(fmt, rng, vec_list);
        TEST_ASSERT(in.empty(), "missing key -> empty");
    }
    std::cout << "      PASSED\n";

    // 24c: 正常装配（数值 pair vector + 字符串 pair vector）+ 运行 + 回填
    std::cout << "  24c: assemble and back-fill\n";
    {
        std::vector<std::string> fmt;
        fmt.push_back("");       // 连续
        fmt.push_back("list");   // 候选 {10,20,30}
        std::vector<std::pair<double, double> > rng;
        rng.push_back(std::make_pair(-5.0, 5.0));
        rng.push_back(std::make_pair(0.0, 30.0));
        std::vector<std::pair<int, std::vector<double> > > vec_list;
        vec_list.push_back(std::make_pair(1, dblList3(10.0, 20.0, 30.0)));

        const std::vector<OM::VarInput> inputs = OM::detail::inputsFromFormat(fmt, rng, vec_list);
        TEST_ASSERT(inputs.size() == 2, "2 inputs from pair vectors");

        std::vector<std::pair<std::string, std::vector<std::string> > > srange;
        srange.push_back(std::make_pair(std::string("L"), strList2("p1", "p2")));

        std::vector<std::vector<OM::VarInput> > groups;
        groups.push_back(inputs);
        groups.push_back(OM::detail::inputsFromStringRange(srange));
        const std::vector<OM::VarInput> all = OM::detail::mergeInputs(groups);
        TEST_ASSERT(all.size() == 3, "3 inputs after merge");

        auto cost = [](const std::vector<OM::VarValue>& v) -> double {
            // v[1] 候选全整数 → DISC_INT（读取 val_int）; v[0] 连续, v[2] 字符串
            const double dx = v[0].val_dou;
            const double dn = static_cast<double>(v[1].val_int) - 20.0;
            const double ds = (v[2].val_str == "p2") ? 0.0 : 5.0;
            return dx * dx + dn * dn + ds;
        };
        OM::GAConfig cfg(30, 100, 0.8, 0.1);
        OM::GAMixed ga(std::make_shared<OM::SerialEvaluator>(cost), all, cfg);
        GAResult r = run_and_capture(ga);
        TEST_ASSERT(r.best_cost < 1.0, "converged near 0");

        // 回填检查（前端 pair 形态）
        const std::vector<OM::VarValue> best = ga.getBestPara();
        const std::vector<std::pair<int, double> > idx_pairs = OM::detail::toIndexValuePairs(best);
        const std::vector<std::pair<std::string, std::string> > str_pairs = OM::detail::toStringPairs(best);
        const std::vector<std::pair<double, double> > ranges = OM::detail::toRanges(ga.getSpecs());

        TEST_ASSERT(idx_pairs.size() == 2, "2 numeric pairs");
        TEST_ASSERT(idx_pairs[1].second == 20.0, "discrete value 20");
        TEST_ASSERT(str_pairs.size() == 1 && str_pairs[0].first == "L", "1 string pair");
        TEST_ASSERT(ranges.size() == 3, "3 ranges");
        TEST_ASSERT(std::fabs(ranges[0].first + 5.0) < 1e-12, "range[0].first == -5");

        std::cout << "      range = [" << ranges[0].first << ", " << ranges[0].second
                  << "], disc = " << idx_pairs[1].second
                  << ", string = " << str_pairs[0].second << "\n";
    }
    std::cout << "      PASSED\n\n";
}

// ==========================================================================
// main
// ==========================================================================
int main()
{
    std::cout << "=== OptMix v6.2 ===\n\n";

    test_continuous_only();
    test_discrete_only();
    test_mixed_variables();
    test_edge_cases();
    test_mutual_ui();
    test_early_termination();
    test_exhaustion();
    test_abort();
    test_seeding();
    test_adv_stop();
    test_determinism();
    test_boundary();
    test_custom_evaluator_injection();
    test_evaluator_equivalence();
    test_step_format();
    test_eia_format();
    test_list_format();
    test_int_format();
    test_string_format();
    test_input_adapters();
    test_format_error();
    test_debug_log();
    test_log_file();
    test_pair_vector_inputs();

    std::cout << "=== All tests passed! ===\n";
    return 0;
}
