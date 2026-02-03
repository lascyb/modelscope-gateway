"""
ModelScope API 负载均衡客户端 - 使用示例。

功能说明：
- 智能路由：通过本地 AI 分析任务复杂度，自动选择合适的模型
- 自动根据优先级选择可用模型
- 从 API 响应头自动获取配额限制信息
- 记录每个模型的使用量到 JSON 文件 (usage/YYYY-MM-DD.json)
- 记录限制信息到 JSON 文件 (limits/YYYY-MM-DD.json)
- 支持流式和非流式响应

使用前请复制 env.example 为 .env 并填入 MODELSCOPE_API_KEY
"""

from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(Path(__file__).parent / ".env")

from core.api_client import (
    ModelScopeClient,
    NoAvailableModelError,
    RateLimitError,
    AuthenticationError,
)


def print_status(client: ModelScopeClient) -> None:
    """print_status 打印当前使用状态。"""
    status = client.get_status()
    print("\n" + "=" * 80)

    global_limit = status["global_daily_limit"]
    global_remaining = status["remaining"]
    global_limit_str = str(global_limit) if global_limit is not None else "未知"
    global_remaining_str = str(global_remaining) if global_remaining is not None else "未知"
    total_usage = status["total_usage"]
    smart_routing = "✓" if status.get("smart_routing_available") else "✗"

    print(f"全局限制: {global_limit_str} | 本地记录使用: {total_usage} | API剩余: {global_remaining_str} | 智能路由: {smart_routing}")
    print("-" * 80)
    print(f"{'模型名称':<30} {'层级':>5} {'本地用':>5} {'限制':>7} {'剩余':>7} {'状态':>10}")
    print("-" * 80)

    for m in status["models"]:
        limit_str = str(m["daily_limit"]) if m["daily_limit"] is not None else "-"
        remaining_str = str(m["remaining"]) if m["remaining"] is not None else "-"
        state = "✓ 可用" if m["available"] else "✗ 不可用"
        print(
            f"{m['name']:<35} {m['tier']:>4} {m['usage']:>6} "
            f"{limit_str:>8} {remaining_str:>8} {state:>10}"
        )

    print("=" * 80 + "\n")


def example_simple_task(client: ModelScopeClient) -> None:
    """example_simple_task 简单任务示例（智能路由应选择小模型）。"""
    print("\n--- 简单任务示例 ---")
    try:
        response = client.chat(
            messages=[
                {"role": "user", "content": "你好，今天天气怎么样？"},
            ],
            max_tokens=100,
        )
        print(f"使用模型: {response['model']}")
        print(f"回复内容: {response['content'][:100]}...")
        if "routing" in response:
            r = response["routing"]
            print(f"路由信息: 复杂度={r['complexity']}, 分数={r['score']}/10, 层级={r['actual_tier']}")
        if "retried_models" in response:
            print(f"跳过的模型: {response['retried_models']}")
    except AuthenticationError as e:
        print(f"认证错误: {e}")
    except RateLimitError as e:
        print(f"配额用尽: {e}")
    except NoAvailableModelError as e:
        print(f"无可用模型: {e}")


def example_complex_task(client: ModelScopeClient) -> None:
    """example_complex_task 复杂任务示例（智能路由应选择大模型）。"""
    print("\n--- 复杂任务示例 ---")
    try:
        response = client.chat(
            messages=[
                {
                    "role": "user",
                    "content": "用标点符号显示出来一个小女孩的脸，最少使用200个标点符号",
                },
            ],
            max_tokens=500,
        )
        print(f"使用模型: {response['model']}")
        print(f"回复内容: {response['content']}")
        if "routing" in response:
            r = response["routing"]
            print(f"路由信息: 复杂度={r['complexity']}, 分数={r['score']}/10, 层级={r['actual_tier']}")
        if "retried_models" in response:
            print(f"跳过的模型: {response['retried_models']}")
    except AuthenticationError as e:
        print(f"认证错误: {e}")
    except RateLimitError as e:
        print(f"配额用尽: {e}")
    except NoAvailableModelError as e:
        print(f"无可用模型: {e}")


def example_medium_task(client: ModelScopeClient) -> None:
    """example_medium_task 中等任务示例。"""
    print("\n--- 中等任务示例 ---")
    try:
        response = client.chat(
            messages=[
                {
                    "role": "user",
                    "content": "解释一下什么是依赖注入，以及它在软件开发中的作用。",
                },
            ],
            max_tokens=300,
        )
        print(f"使用模型: {response['model']}")
        print(f"回复内容: {response['content'][:150]}...")
        if "routing" in response:
            r = response["routing"]
            print(f"路由信息: 复杂度={r['complexity']}, 分数={r['score']}/10, 层级={r['actual_tier']}")
        if "retried_models" in response:
            print(f"跳过的模型: {response['retried_models']}")
    except AuthenticationError as e:
        print(f"认证错误: {e}")
    except RateLimitError as e:
        print(f"配额用尽: {e}")
    except NoAvailableModelError as e:
        print(f"无可用模型: {e}")


def example_analyze_only(client: ModelScopeClient) -> None:
    """example_analyze_only 仅分析任务复杂度（不调用远程 API）。"""
    print("\n--- 仅分析任务复杂度 ---")

    tasks = [
        "你好",
        "帮我把这段话翻译成英文：今天天气很好",
        "解释一下 TCP 三次握手的过程",
        "用动态规划解决背包问题，并分析时间和空间复杂度",
    ]

    for task in tasks:
        result = client.analyze_task([{"role": "user", "content": task}])
        if result:
            print(f"任务: {task[:30]}...")
            print(f"  复杂度: {result.complexity.name}, 分数: {result.score}/10")
            print(f"  原因: {result.reason}")
            print(f"  建议层级: {result.suggested_model_tier}")
            print()
        else:
            print("智能路由未启用")
            break


def example_specify_model(client: ModelScopeClient) -> None:
    """example_specify_model 指定模型示例（跳过智能路由）。"""
    print("\n--- 指定模型示例 ---")
    try:
        response = client.chat(
            messages=[
                {"role": "user", "content": "你好！"},
            ],
            model_id="qwen/Qwen2.5-7B-Instruct",
            max_tokens=50,
        )
        print(f"使用模型: {response['model']}")
        print(f"回复内容: {response['content']}")
    except NoAvailableModelError as e:
        print(f"错误: {e}")
    except ValueError as e:
        print(f"错误: {e}")


def main():
    """main 主函数，运行示例。"""
    # 初始化客户端
    # 需要设置环境变量 MODELSCOPE_API_KEY 或传入 api_key 参数
    client = ModelScopeClient()

    # 打印当前状态
    print("当前状态（限制信息将在首次 API 调用后自动获取）：")
    print_status(client)

    # 如果智能路由可用，先展示分析功能
    if client.smart_routing_enabled:
        example_analyze_only(client)

    # 运行不同复杂度的任务示例
    example_simple_task(client)
    print_status(client)
    
    example_medium_task(client)
    print_status(client)

    example_complex_task(client)
    print_status(client)


if __name__ == "__main__":
    main()
