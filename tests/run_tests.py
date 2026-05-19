# 销售话术合规检测 - 测试程序
# 使用方法：python tests/run_tests.py

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import check_message

TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cases.json")


def run_tests():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    total = len(cases)
    correct = 0
    violations_correct = 0
    violations_total = 0
    normals_correct = 0
    normals_total = 0

    print(f"共 {total} 条测试用例\n")

    for i, case in enumerate(cases, 1):
        msg = case["message"]
        expected_violation = case["is_violation"]
        expected_id = case.get("expected_rule_id")
        tags = case.get("tags", [])

        result = check_message(msg)
        actual_violation = result["is_violation"]

        # 判定是否/违规正确
        verdict_ok = actual_violation == expected_violation

        # 违规类：进一步检查是否命中正确规则
        rule_ok = True
        if expected_violation and actual_violation and expected_id:
            hit_ids = [h["id"] for h in result["hits"]]
            rule_ok = expected_id in hit_ids

        # 统计
        if verdict_ok:
            correct += 1
        if expected_violation:
            violations_total += 1
            if verdict_ok:
                violations_correct += 1
        else:
            normals_total += 1
            if verdict_ok:
                normals_correct += 1

        # 输出
        status = "✓" if verdict_ok else "✗"
        label = "违规" if expected_violation else "正常"
        detail = ""
        if expected_violation:
            if not actual_violation:
                detail = "漏检"
            elif not rule_ok:
                detail = f"命中规则不对(期望ID={expected_id}, 实际={[h['id'] for h in result['hits']]})"
            else:
                detail = f"ID={expected_id} dist={result['hits'][0]['distance']}"
        else:
            if actual_violation:
                detail = f"误判(最近ID={result['hits'][0]['id']} dist={result['hits'][0]['distance']})"

        print(f"  {status} #{i:02d} [{label}] \"{msg}\" → {detail}")

    print(f"\n{'='*50}")
    print(f"总准确率:  {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"违规召回:  {violations_correct}/{violations_total} ({violations_correct/violations_total*100:.0f}%)")
    print(f"正常排除:  {normals_correct}/{normals_total} ({normals_correct/normals_total*100:.0f}%)")


if __name__ == "__main__":
    run_tests()
