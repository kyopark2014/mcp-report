# MCP를 이용한 보고서 작성

여기에서는 MCP로 agent를 생성하여 다양한 보고서를 생성하는 방법에 대해 설명합니다.

## Workflow를 이용한 비용 분석

비용분석을 위한 Workflow는 아래와 같이 정의합니다. [AWS Cost Explorer]를 이용해, 서비스별, 리전별, 기간별 사용량 데이터를 가져옵니다. 이후 정해진 서식에 맞추어 보고서를 생성합니다. 생성된 보고서에 부족한 부분은 MCP를 이용해 얻어진 aws document, aws cli를 이용해 결과를 업데이트합니다.

<img src="https://github.com/user-attachments/assets/23c05fec-cf67-48ef-8bdd-c1f13539174e" width="400">

비용 분석에 대한 리포트는 아래와 같이 [cost_insight.md](./application/aws_cost/cost_insight.md)을 참조합니다.

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

### Agent의 구현

[LangGraph Builder](https://build.langchain.com/)로 구현된 graph는 [stub.py](./application/aws_cost/stub.py)와 같이 구현됩니다. 이 코드는 LangGraph에 자동 생성된 코드에서 Agent이름만을 수정하였습니다.

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

이때 Node간 데이터 교환을 위해 사용하는 state는 아래와 같습니다. 상세코드는 [implementation.py](./application/aws_cost/implementation.py)를 참조합니다. 여기서 service_costs, region_costs, daily_cost는 Boto3 API를 이용해 획득안 json 형태의 데이터이고, appendix는 그림파일과 설명을 가진 마찬가지로 json 파입니다. additonal_context는 reflection과 MCP tool 활용으로 얻어진 contents가 저장됩니다. iteration은 reflection의 반복 횟수를 의미하고 reflection은 draft의 수정사항과 추가 검색어를 가지고 있습니다. final_response는 최종 답변이 저장됩니다.

```python
class CostState(TypedDict):
    service_costs: dict
    region_costs: dict
    daily_costs: dict
    additional_context: list[str]
    appendix: list[str]
    iteration: int
    reflection: list[str]
    final_response: str
```

### 각종 데이터의 획득

모든 API를 MCP로 하지 않고, report의 기본 데이터는 직접 boto3를 이용해 획득합니다. 아래에서는 service 사용에 대한 정보를 cost explorer를 이용해 수집합니다. 기간은 30일이지만 사용자의 목적에 따라 변경 가능합니다. Service 사용 정보롤 data frame으로 변환후 pie 그래프를 그리고 결과를 appendex로 저장합니다. 언어모델인 LLM은 figure와 같은 URL 정보를 최종 결과에 누락할 수 있으므로 별도로 관리합니다. 

```python
def service_cost(state: CostState, config) -> dict:
    logger.info(f"###### service_cost ######")

    logger.info(f"Getting cost analysis...")
    days = 30

    request_id = config.get("configurable", {}).get("request_id", "")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # cost explorer
        ce = boto3.client('ce')

        # service cost
        service_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        logger.info(f"service_response: {service_response}")

    except Exception as e:
        logger.info(f"Error in cost analysis: {str(e)}")
        return None
    
    service_costs = pd.DataFrame([
        {
            'SERVICE': group['Keys'][0],
            'cost': float(group['Metrics']['UnblendedCost']['Amount'])
        }
        for group in service_response['ResultsByTime'][0]['Groups']
    ])
    logger.info(f"Service Costs: {service_costs}")
    
    # service cost (pie chart)
    fig_pie = px.pie(
        service_costs,
        values='cost',
        names='SERVICE',
        color='SERVICE',
        title='Service Cost',
        template='plotly_white',  # Clean background
        color_discrete_sequence=px.colors.qualitative.Set3  # Color palette
    )    

    url = get_url(fig_pie, "service_cost")

    task = "AWS 서비스 사용량"
    output_images = f"![{task} 그래프]({url})\n\n"

    key = f"artifacts/{request_id}_steps.md"
    time = f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    instruction = f"이 이미지는 {task}에 대한 그래프입니다. 하나의 문장으로 이 그림에 대해 500자로 설명하세요."
    summary = get_summary(fig_pie, instruction)

    body = f"## {task}\n\n{output_images}\n\n{summary}\n\n"
    chat.updata_object(key, time + body, 'append')

    appendix = state["appendix"] if "appendix" in state else []
    appendix.append(body)

    return {
        "appendix": appendix,
        "service_costs": service_response,
    }
```

마찬가지로 region과 daily cost를 추출할 수 있습니다. 상세코드는 [implementation.py](./application/aws_cost/implementation.py)를 참조합니다. 

### 초안의 생성

주어진 데이터로 아래와 같이 draft를 생성합니다. Reflection과 MCP tool을 이용해 additional_context을 이용하면 draft를 개선하는 효과가 있습니다. draft 생성시 아래와 같이 [cost_insight.md](./application/aws_cost/cost_insight.md)를 이용하여 기본 포맷을 설정합니다. 따라서 insight 생성시 기본 포맷을 유지하기 위하여 additional_context를 generate_insight, reflection, mcp-tools를 이용해 업데이트를 수행합니다.

```python
def generate_insight(state: CostState, config) -> dict:
    logger.info(f"###### generate_insight ######")

    prompt_name = "cost_insight"
    request_id = config.get("configurable", {}).get("request_id", "")    
    additional_context = state["additional_context"] if "additional_context" in state else []

    system_prompt=get_prompt_template(prompt_name)
    logger.info(f"system_prompt: {system_prompt}")

    human = (
        "다음 AWS 비용 데이터를 분석하여 상세한 인사이트를 제공해주세요:"
        "Cost Data:"
        "<service_costs>{service_costs}</service_costs>"
        "<region_costs>{region_costs}</region_costs>"
        "<daily_costs>{daily_costs}</daily_costs>"

        "다음의 additional_context는 관련된 다른 보고서입니다. 이 보고서를 현재 작성하는 보고서에 추가해주세요. 단, 전체적인 문맥에 영향을 주면 안됩니다."
        "<additional_context>{additional_context}</additional_context>"
    )

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human)])
    logger.info(f'prompt: {prompt}')    

    llm = chat.get_chat(extended_thinking="Disable")
    chain = prompt | llm

    service_costs = json.dumps(state["service_costs"])
    region_costs = json.dumps(state["region_costs"])
    daily_costs = json.dumps(state["daily_costs"])

    try:
        response = chain.invoke(
            {
                "service_costs": service_costs,
                "region_costs": region_costs,
                "daily_costs": daily_costs,
                "additional_context": additional_context
            }
        )
        logger.info(f"response: {response.content}")

    except Exception:
        err_msg = traceback.format_exc()
        logger.debug(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")
    
    # logging in step.md
    key = f"artifacts/{request_id}_steps.md"
    time = f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    chat.updata_object(key, time + response.content, 'append')
    
    # report.md
    key = f"artifacts/{request_id}_report.md"
    body = "# AWS 사용량 분석\n\n" + response.content + "\n\n"  

    appendix = state["appendix"] if "appendix" in state else []
    values = '\n\n'.join(appendix)

    logger.info(f"body: {body}")
    chat.updata_object(key, time+body+values, 'prepend')

    iteration = state["iteration"] if "iteration" in state else 0

    return {
        "final_response": body+values,
        "iteration": iteration+1
    }
```

### 초안의 개선점 파악

초안(draft)의 개선은 아래와 같이 수행합니다. 진행 결과를 확인하기 위해 중간 결과를 steps.md에 저장합니다.

```python
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    advisable: str = Field(description="Critique of what is helpful for better answer")
    superfluous: str = Field(description="Critique of what is superfluous")

class Research(BaseModel):
    """Provide reflection and then follow up with search queries to improve the answer."""

    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )
    
def reflect(draft):
    logger.info(f"###### reflect ######")

    reflection = []
    search_queries = []
    for attempt in range(5):
        llm = chat.get_chat(extended_thinking="Disable")
        structured_llm = llm.with_structured_output(Research, include_raw=True)
        
        info = structured_llm.invoke(draft)
        logger.info(f'attempt: {attempt}, info: {info}')
            
        if not info['parsed'] == None:
            parsed_info = info['parsed']
            reflection = [parsed_info.reflection.missing, parsed_info.reflection.advisable]
            logger.info(f"reflection: {reflection}")
            search_queries = parsed_info.search_queries
            logger.info(f"search_queries: {search_queries}")            
            break
    
    return {
        "reflection": reflection,
        "search_queries": search_queries
    }

def reflect_context(state: CostState, config) -> dict:
    # earn reflection from the previous final response    
    result = reflect(state["final_response"])

    # logging in step.md
    request_id = config.get("configurable", {}).get("request_id", "")
    key = f"artifacts/{request_id}_steps.md"
    time = f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"    
    body = f"Reflection: {result['reflection']}\n\nSearch Queries: {result['search_queries']}\n\n"
    chat.updata_object(key, time + body, 'append')

    return {
        "reflection": result
    }
```

### MCP로 개선(reflection)에 필요한 데이터 획득

아래와 같이 MCP Tools 노드에서 reflection에 필요한 데이터를 가져옵니다. 

```python
def mcp_tools(state: CostState, config) -> dict:
    logger.info(f"###### mcp_tools ######")
    draft = state['final_response']

    reflection_result = chat.run_reflection_agent(draft, state["reflection"])
    logger.info(f"reflection result: {reflection_result}")

    # logging in step.md
    request_id = config.get("configurable", {}).get("request_id", "")
    key = f"artifacts/{request_id}_steps.md"
    time = f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"    
    body = f"{reflection_result}\n\n"
    chat.updata_object(key, time + body, 'append')

    additional_context = state["additional_context"] if "additional_context" in state else []
    additional_context.append(reflection_result)

    return {
        "additional_context": additional_context
    }
```

Reflection agent는 아래와 같이 구성합니다. Reflection agent에 대한 상세 코드는 [chat.py](./application/chat.py)를 참조합니다.

```python
async def reflection_mcp_agent(draft, reflection):
    global mcp_json
    logger.info(f"mcp_json: {mcp_json}")
    
    server_params = load_multiple_mcp_server_parameters(mcp_json)
    logger.info(f"server_params: {server_params}")

    async with MultiServerMCPClient(server_params) as client:
        ref = ""
        tools = client.get_tools()
        if debug_mode == "Enable":
            logger.info(f"tools: {tools}")

        agent, config = create_reflection_agent(tools, draft)

        instraction = reflection['reflection']
        search_queries = reflection['search_queries']
        logger.info(f"reflection: {instraction}, search_queries: {search_queries}")

        try:
            response = await agent.ainvoke({"messages": instraction[0]}, config)
            result = response["messages"][-1].content

            debug_msgs = get_debug_messages()
            for msg in debug_msgs:
                # logger.info(f"msg: {msg}")
                if "image" in msg:
                    logger.info(f"image: {msg['image']}")
                elif "text" in msg:
                    logger.info(f"text: {msg['text']}")

            image_url = response["image_url"] if "image_url" in response else []
            logger.info(f"image_url: {image_url}")

            for image in image_url:
                logger.info(f"image: {image}")

            return result
        except Exception as e:
            logger.error(f"Error during agent invocation: {str(e)}")
            raise Exception(f"Agent invocation failed: {str(e)}")
```


### 개선이 반영된 문서 생성

MCP로 얻어진 데이터를 context에 저장한 후에 아래와 같이 초안의 개선을 수행합니다.

```python
def revise_draft(draft, context):   
    logger.info(f"###### revise_draft ######")
        
    system = (
        "당신은 보고서를 잘 작성하는 논리적이고 똑똑한 AI입니다."
        "당신이 작성하여야 할 보고서 <draft>의 소제목과 기본 포맷을 유지한 상태에서, 다음의 <context>의 내용을 추가합니다."
        "초등학생도 쉽게 이해하도록 풀어서 씁니다."
    )
    human = (
        "<draft>{draft}</draft>"
        "<context>{context}</context>"
    )
                
    reflection_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human)
        ]
    )        
    reflect = reflection_prompt | chat.get_chat(extended_thinking="Disable")
        
    result = reflect.invoke({
        "draft": draft,
        "context": context
    })   
                            
    return result.content
```

### Local에서 실행하기

1) 필수는 아니지만 정상적인 진행을 위해서는 AWS CLI가 필요합니다. [최신 버전의 AWS CLI 설치 또는 업데이트](https://docs.aws.amazon.com/ko_kr/cli/latest/userguide/getting-started-install.html)에 따라 설치후 "aws configure" 명령으로 AWS credential을 등록합니다.

2) venv로 환경을 구성하면 편리합니다. 적당한 폴더를 만들고 아래와 같이 환경을 설정합니다.

```text
python -m venv venv
source venv/bin/activate
```

3) 소스를 다운로드 합니다.

```python
git clone https://github.com/kyopark2014/mcp-report
```

4) 이후 다운로드 받은 github 폴더로 이동한 후에 아래와 같이 필요한 패키지를 추가로 설치 합니다.

```text
cd mcp-report && python -m pip install -r requirements.txt
```

5) [실습에 필요한 Key 설정](https://github.com/kyopark2014/mcp-agent/blob/main/mcp.md#%EC%8B%A4%EC%8A%B5%EC%97%90-%ED%95%84%EC%9A%94%ED%95%9C-key-%EC%84%A4%EC%A0%95)에 따라서, 인터넷과 날씨조회 API에 대한 key를 설정합니다. 설정이 되면, application/config.json을 아래와 같은 json 파일이 생성됩니다.

```java
{
    "WEATHER_API_KEY": "fbd00245cabcedefghijkd3e94905f7049",
    "TAVILY_API_KEY": "tvly-1234567890U3imZFs4LNO2g0Qv1LoE"
}
```

6) 이제 준비가 되었으므로, 아래와 같은 명령어로 streamlit을 실행합니다. [MCP Tool 사용 방법](https://github.com/kyopark2014/mcp-agent/blob/main/mcp.md#mcp-tool-%EC%82%AC%EC%9A%A9-%EB%B0%A9%EB%B2%95)을 참조하여 동작 테스트를 수행합니다.

```text
streamlit run application/app.py
```

### Local에서 Docker로 실행하기

아래와 같이 docker를 설치하고 실행합니다. 아래는 Mac의 예제입니다.

```text
brew install --cask docker
```

빌드시 사용하는 Dockerfile은 Mac(ARM)입니다. Windows라면 [Dockerfile_x86](./Dockerfile_x86)을 아래와 같이 사용합니다.

```text
cp Dockerfile_x86 Dockerfile
```

이제 아래와 같이 스크립트를 이용해 빌드합니다. [build.sh](./build.sh)는 aws credential을 조회해서 build 할때에 포함합니다.

```text
./build.sh
```

이제 아래와 같이 실행합니다. 아래에서는 편의상 docker의 포트를 8502로 설정하였는데 자신의 환경에 따라 설정할 수 있습니다.

```text
docker run -p 8502:8501 mcp-report
```

브라우저에서 아래 URL로 접속합니다. 

```text
http://0.0.0.0:8502
```

### EC2에서 업데이트 후 실행하기

EC2가 private subnet에 있으므로 Session Manger로 접속합니다. 이후 아래와 같이 EC2를 위한 Dockerfile로 바꿉니다.

```text
sudo runuser -l ec2-user -c "cd mcp-report&&cp Dockerfile_x86 Dockerfile"
```

ec2-user로 설치가 진행되었으므로 아래와 같이 code를 업데이트합니다.

```text
sudo runuser -l ec2-user -c 'cd /home/ec2-user/mcp-report && git pull'
```

이제 아래와 같이 docker를 빌드합니다.

```text
sudo runuser -l ec2-user -c "cd mcp-report && docker build -t streamlit-app ."
```

빌드가 완료되면 "sudo docker ps"로 docker id를 확인후에 "sudo docker kill" 명령어로 종료합니다.

![noname](https://github.com/user-attachments/assets/4afb2af8-d092-4aaa-813a-65975375f7d4)

이후 아래와 같이 다시 실행합니다.

```text
sudo runuser -l ec2-user -c 'docker run -d -p 8501:8501 streamlit-app'
```

만약 console에서 debugging할 경우에는 -d 옵션없이 아래와 같이 실행합니다.

```text
sudo runuser -l ec2-user -c 'docker run -p 8501:8501 streamlit-app'
```




## 실행 결과

아래와 같은 Plan으로 실행된 것울 알 수 있습니다.

<img src="https://github.com/user-attachments/assets/5ac86ba4-7aec-44ac-b86a-7796f59a630f" width="250">

순차적으로 실행된 결과를 확인합니다.

<img src="https://github.com/user-attachments/assets/420c185b-9ad6-4b28-af5d-bcb2ac98ece0" width="750">

최종 결과는 아래와 같이 보여집니다.

<img src="https://github.com/user-attachments/assets/88ef1051-8663-43bb-ac79-84d0d335d7d5" width="750">

