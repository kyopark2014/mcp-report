# MCP를 이용한 보고서 작성

여기에서는 MCP로 agent를 생성하여 다양한 보고서를 생성하는 방법에 대해 설명합니다.

## Workflow를 이용한 비용 분석

비용분석을 위한 Workflow는 아래와 같이 정의합니다. [AWS Cost Explorer]를 이용해, 서비스별, 리전별, 기간별 사용량 데이터를 가져옵니다. 이후 정해진 서식에 맞추어 보고서를 생성합니다. 생성된 보고서에 부족한 부분은 MCP를 이용해 얻어진 aws document, aws cli를 이용해 결과를 업데이트합니다.

<img src="https://github.com/user-attachments/assets/23c05fec-cf67-48ef-8bdd-c1f13539174e" width="400">

비용 분석에 대한 리포트는 아래와 같이 주어진 [cost_insight.md](./application/aws_cost/cost_insight.md)는 아래와 같습니다.

```text
당신의 AWS solutions architect입니다.
다음의 Cost Data을 이용하여 user의 질문에 답변합니다.
모르는 질문을 받으면 솔직히 모른다고 말합니다.
답변의 이유를 풀어서 명확하게 설명합니다.

다음 항목들에 대해 분석해주세요:
1. 주요 비용 발생 요인
2. 비정상적인 패턴이나 급격한 비용 증가
3. 비용 최적화가 가능한 영역
4. 전반적인 비용 추세와 향후 예측

분석 결과를 다음과 같은 형식으로 제공해주세요:
## 주요 비용 발생 요인
- [구체적인 분석 내용]

## 이상 패턴 분석
- [비정상적인 비용 패턴 설명]

## 최적화 기회
- [구체적인 최적화 방안]

## 비용 추세
- [추세 분석 및 예측]
```

## 상세 구현

LangGraph Builder로 구현된 graph는 [stub.py](./application/aws_cost/stub.py)와 같이 구현됩니다. 이 코드는 LangGraph에 자동 생성된 코드에서 Agent이름만을 수정하였습니다.

```python
def CostAgent(
    *,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
    input: Optional[Type[Any]] = None,
    output: Optional[Type[Any]] = None,
    impl: list[tuple[str, Callable]],
) -> StateGraph:
    """Create the state graph for CostAgent."""
    # Declare the state graph
    builder = StateGraph(
        state_schema, config_schema=config_schema, input=input, output=output
    )

    nodes_by_name = {name: imp for name, imp in impl}

    all_names = set(nodes_by_name)

    expected_implementations = {
        "service_cost",
        "region_cost",
        "daily_cost",
        "generate_insight",
        "reflect_context",
        "should_end",
        "mcp_tools",
    }

    missing_nodes = expected_implementations - all_names
    if missing_nodes:
        raise ValueError(f"Missing implementations for: {missing_nodes}")

    extra_nodes = all_names - expected_implementations

    if extra_nodes:
        raise ValueError(
            f"Extra implementations for: {extra_nodes}. Please regenerate the stub."
        )

    # Add nodes
    builder.add_node("service_cost", nodes_by_name["service_cost"])
    builder.add_node("region_cost", nodes_by_name["region_cost"])
    builder.add_node("daily_cost", nodes_by_name["daily_cost"])
    builder.add_node("generate_insight", nodes_by_name["generate_insight"])
    builder.add_node("reflect_context", nodes_by_name["reflect_context"])
    builder.add_node("mcp_tools", nodes_by_name["mcp_tools"])

    # Add edges
    builder.add_edge(START, "service_cost")
    builder.add_edge("service_cost", "region_cost")
    builder.add_edge("region_cost", "daily_cost")
    builder.add_edge("daily_cost", "generate_insight")
    builder.add_conditional_edges(
        "generate_insight",
        nodes_by_name["should_end"],
        [
            END,
            "reflect_context",
        ],
    )
    builder.add_edge("reflect_context", "mcp_tools")
    builder.add_edge("mcp_tools", "generate_insight")
    
    return builder
```


## 실행 결과

아래와 같은 Plan으로 실행된 것울 알 수 있습니다.

<img src="https://github.com/user-attachments/assets/5ac86ba4-7aec-44ac-b86a-7796f59a630f" width="250">

순차적으로 실행된 결과를 확인합니다.

<img src="https://github.com/user-attachments/assets/420c185b-9ad6-4b28-af5d-bcb2ac98ece0" width="750">

최종 결과는 아래와 같이 보여집니다.

<img src="https://github.com/user-attachments/assets/88ef1051-8663-43bb-ac79-84d0d335d7d5" width="750">

