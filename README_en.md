# MCP Report Agent

This repository explains how to generate various reports using the MCP agent. The agent can collect necessary data using various APIs and generate appropriate reports using LLM prompts. The generated reports can be improved using the agent's reflection pattern, which extracts improvement points and additional search keywords, and generates new reports based on them. With MCP, you can dynamically use data from various sources according to the improvement points of the report, and even if new tools are added or data formats change due to API changes, the application is not affected.

The overall architecture is as follows. For debugging convenience, the system is configured with Streamlit running on EC2. Data is collected using AWS Cost Explorer, AWS CLI, and AWS Document. If necessary, external data sources such as Tavily can be used for internet searches. Users can safely use the application built with Streamlit via Amazon CloudFront and ALB, and download reports through CloudFront - S3.

![image](https://github.com/user-attachments/assets/3d2bf057-50e5-4899-a1c7-c71233f6e90b)

## Cost Analysis Using Workflow

The workflow for cost analysis is defined as follows. Using [AWS Cost Explorer], data on usage by service, region, and period is obtained. Then, a report is generated in a specified format. Any deficiencies in the generated report are updated using results obtained from aws document and aws cli via MCP.

<img src="https://github.com/user-attachments/assets/2ce8952c-fd8b-441e-94f8-77d9da98708b" width="400">

For the cost analysis report, refer to [cost_insight.md](./application/aws_cost/cost_insight.md).

```text
You are an AWS solutions architect.
Answer the user's questions using the following Cost Data.
If you receive a question you don't know, honestly say you don't know.
Explain the reason for your answer in detail and clearly.

Please analyze the following items:
1. Major cost drivers
2. Abnormal patterns or sudden cost increases
3. Areas for cost optimization
4. Overall cost trends and future predictions

Provide the analysis results in the following format:
## Major Cost Drivers
- [Detailed analysis]

## Abnormal Pattern Analysis
- [Description of abnormal cost patterns]

## Optimization Opportunities
- [Specific optimization measures]

## Cost Trends
- [Trend analysis and predictions]
```

## Detailed Implementation

### Implementation of the Agent

The graph implemented with [LangGraph Builder](https://build.langchain.com/) is implemented as shown in [stub.py](./application/aws_cost/stub.py). This code only modifies the Agent name from the automatically generated code in LangGraph.

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

The state used for data exchange between nodes is as follows. For detailed code, refer to [implementation.py](./application/aws_cost/implementation.py). Here, service_costs, region_costs, and daily_costs are JSON data obtained using the Boto3 API, and appendix is also a JSON file containing image files and descriptions. additonal_context stores contents obtained through reflection and MCP tool utilization. iteration means the number of reflection repetitions, and reflection contains draft revisions and additional search terms. final_response stores the final answer.

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

### Acquiring Various Data

Not all APIs are done with MCP; the basic data for the report is obtained directly using boto3. Below, information on service usage is collected using cost explorer. The period is 30 days but can be changed according to the user's purpose. After converting the service usage information to a data frame, a pie chart is drawn and the result is saved as an appendix. Since the LLM may omit URL information such as figures in the final result, it is managed separately.

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

    task = "AWS Service Usage"
    output_images = f"![{task} graph]({url})\n\n"

    key = f"artifacts/{request_id}_steps.md"
    time = f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    instruction = f"This image is a graph for {task}. Please describe this figure in one sentence within 500 characters."
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

Similarly, region and daily costs can be extracted. For detailed code, refer to [implementation.py](./application/aws_cost/implementation.py).

### Draft Generation

A draft is generated with the given data as shown below. Using additional_context with Reflection and MCP tool is effective in improving the draft. When generating a draft, the basic format is set using [cost_insight.md](./application/aws_cost/cost_insight.md) as shown below. Therefore, to maintain the basic format when generating insight, additional_context is updated using generate_insight, reflection, and mcp-tools.

```python
def generate_insight(state: CostState, config) -> dict:
    logger.info(f"###### generate_insight ######")

    prompt_name = "cost_insight"
    request_id = config.get("configurable", {}).get("request_id", "")    
    additional_context = state["additional_context"] if "additional_context" in state else []

    system_prompt=get_prompt_template(prompt_name)
    logger.info(f"system_prompt: {system_prompt}")

    human = (
        "Please analyze the following AWS cost data and provide detailed insights:"
        "Cost Data:"
        "<service_costs>{service_costs}</service_costs>"
        "<region_costs>{region_costs}</region_costs>"
        "<daily_costs>{daily_costs}</daily_costs>"

        "The following additional_context contains other related reports. Please add these reports to the current report you are writing. However, do not let them affect the overall context."
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
    body = "# AWS Usage Analysis\n\n" + response.content + "\n\n"  

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

### Identifying Improvements to the Draft

Draft improvements are performed as shown below. Intermediate results are saved in steps.md to check the progress.

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

### Acquiring Data Needed for Improvement (Reflection) with MCP

As shown below, the MCP Tools node obtains the data needed for reflection. The required data source may vary depending on the result of the reflection. By using MCP, you can select and utilize appropriate tools from various data sources.

```python
def mcp_tools(state: CostState, config) -> dict:
    draft = state['final_response']

    appendix = state["appendix"] if "appendix" in state else []

    reflection_result, image_url= asyncio.run(reflection_agent.run(draft, state["reflection"]))

    value = ""
    if image_url:
        for url in image_url:
            value += f"![image]({url})\n\n"
    if value:
        appendix.append(value)
    
    # logging in step.md
    request_id = config.get("configurable", {}).get("request_id", "")
    key = f"artifacts/{request_id}_steps.md"
    time = f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"    
    body = f"{reflection_result}\n\n"
    value = '\n\n'.join(appendix)
    chat.updata_object(key, time + body + value, 'append')

    if response_container:
        value = body
        response_container.info('[response]\n' + value[:500])
        response_msg.append(value[:500])

    additional_context = state["additional_context"] if "additional_context" in state else []
    additional_context.append(reflection_result)

    return {
        "additional_context": additional_context
    }
```

The Reflection agent is configured as follows. For detailed code on the Reflection agent, refer to [reflection_agent.py](./application/aws_cost/reflection_agent.py).

```python
async def run(draft, reflection):
    server_params = chat.load_multiple_mcp_server_parameters()

    async with MultiServerMCPClient(server_params) as client:
        tools = client.get_tools()

        instruction = (
            f"<reflection>{reflection}</reflection>\n\n"
            f"<draft>{draft}</draft>"
        )

        app = buildChatAgent(tools)
        config = {
            "recursion_limit": 50,
            "tools": tools            
        }

        value = None
        inputs = {
            "messages": [HumanMessage(content=instruction)]
        }

        references = []
        final_output = None
        async for output in app.astream(inputs, config):
            for key, value in output.items():
                logger.info(f"--> key: {key}, value: {value}")
                final_output = output
        
        result = final_output["messages"][-1].content
        logger.info(f"result: {result}")
        image_url = final_output["image_url"] if "image_url" in final_output else []

        return result, image_url
```

### Generating Documents with Reflected Improvements

After storing the data obtained with MCP in the context, the draft is improved as follows.

```python
def revise_draft(draft, context):   
    logger.info(f"###### revise_draft ######")
        
    system = (
        "You are a logical and intelligent AI that writes reports well."
        "While maintaining the subtitles and basic format of the <draft> report you need to write, add the contents of the following <context>."
        "Write in a way that even elementary school students can easily understand."
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

### Running Locally

1) It is not mandatory, but for normal operation, AWS CLI is required. Install the [latest version of AWS CLI](https://docs.aws.amazon.com/ko_kr/cli/latest/userguide/getting-started-install.html) and register your AWS credentials with the "aws configure" command.

2) It is convenient to set up the environment with venv. Create an appropriate folder and set up the environment as follows.

```text
python -m venv venv
source venv/bin/activate
```

3) Download the source code.

```python
git clone https://github.com/kyopark2014/mcp-report
```

4) After moving to the downloaded github folder, install the required packages as follows.

```text
cd mcp-report && python -m pip install -r requirements.txt
```

5) Set the keys required for the practice according to [Key Settings for Practice](https://github.com/kyopark2014/mcp-agent/blob/main/mcp.md#%EC%8B%A4%EC%8A%B5%EC%97%90-%ED%95%84%EC%9A%94%ED%95%9C-key-%EC%84%A4%EC%A0%95). Once set, a json file like below will be created as application/config.json.

```java
{
    "WEATHER_API_KEY": "fbd00245cabcedefghijkd3e94905f7049",
    "TAVILY_API_KEY": "tvly-1234567890U3imZFs4LNO2g0Qv1LoE"
}
```

6) Now you are ready, so run streamlit with the following command. Refer to [How to Use MCP Tool](https://github.com/kyopark2014/mcp-agent/blob/main/mcp.md#mcp-tool-%EC%82%AC%EC%9A%A9-%EB%B0%A9%EB%B2%95) for operation tests.

```text
streamlit run application/app.py
```

### Running Locally with Docker

Install and run docker as follows. The example below is for Mac.

```text
brew install --cask docker
```

The Dockerfile used for building is for Mac (ARM). If you are on Windows, use [Dockerfile_x86](./Dockerfile_x86) as shown below.

```text
cp Dockerfile_x86 Dockerfile
```

Now build using the script as follows. [build.sh](./build.sh) includes aws credentials during the build.

```text
./build.sh
```

Now run as follows. In the example below, the docker port is set to 8502 for convenience, but you can set it according to your environment.

```text
docker run -p 8502:8501 mcp-report
```

Access the following URL in your browser.

```text
http://0.0.0.0:8502
```

### Updating and Running on EC2

Since EC2 is in a private subnet, connect via Session Manager. Then change to the Dockerfile for EC2 as follows.

```text
sudo runuser -l ec2-user -c "cd mcp-report&&cp Dockerfile_ec2 Dockerfile" 
```

Since the installation was done as ec2-user, update the code as follows.

```text
sudo runuser -l ec2-user -c 'cd /home/ec2-user/mcp-report && git pull'
```

Now build the docker as follows.

```text
sudo runuser -l ec2-user -c "cd mcp-report && docker build -t streamlit-app ."
```

After the build is complete, check the docker id with "sudo docker ps" and terminate it with the "sudo docker kill" command.

![noname](https://github.com/user-attachments/assets/4afb2af8-d092-4aaa-813a-65975375f7d4)

Then run again as follows.

```text
sudo runuser -l ec2-user -c 'docker run -d -p 8501:8501 streamlit-app'
```

If you want to debug in the console, run without the -d option as follows.

```text
sudo runuser -l ec2-user -c 'docker run -p 8501:8501 streamlit-app'
```

## Execution Results

After collecting costs for service, region, and daily, insight is extracted and analyzed with reflection, and additional information is collected using MCP with search_documentation and get_service_cost. Then, insight is extracted again and the report is written.

![image](https://github.com/user-attachments/assets/b97dfc98-a672-4732-bbfb-ae0367daabe5)

If you draw this as a graph diagram, you can see that it was executed as the following plan.

<img src="https://github.com/user-attachments/assets/5ac86ba4-7aec-44ac-b86a-7796f59a630f" width="250">

The execution result provides an integrated analysis of major cost drivers, abnormal patterns, optimization, and cost trends according to the given format.

![image](https://github.com/user-attachments/assets/5bfa8865-efc9-4dc7-9ff5-a0d3f20fb796)

The analysis result for the obtained service cost is as follows.

![image](https://github.com/user-attachments/assets/b1ccbc5e-566f-49f2-a539-9463dbdea6f0)

This pie chart shows the distribution of AWS service costs. Amazon OpenSearch Service accounts for 41.9%, followed by Amazon Elastic Compute Cloud - Compute (21.1%), Amazon Bedrock (13%), and EC2 - Other (10.6%). The remaining services (Amazon Virtual Private Cloud, Elastic Load Balancing, CloudWatch, Cost Explorer, etc.) each account for less than 5%.

The cost by region is as follows.

![image](https://github.com/user-attachments/assets/dd996bb3-3897-4a83-82cf-9d038cc769eb)

In the graph showing AWS costs by region, the us-west-2 (Oregon, USA) region shows the highest cost at about $1,900, while other regions (NoRegion, ap-northeast-1, ap-northeast-2, ap-southeast-1, eu-west-1, global, us-east-1, us-east-2) show much lower costs.

The daily cost is as follows.

![image](https://github.com/user-attachments/assets/35736080-3dd3-46ca-a1ef-0a577ffce27d)

Among various AWS services (Amplify, Cost Explorer, ECR, EC2, CloudFront, etc.), AmazonCloudWatch shows a unique pattern, rising sharply to about $95 around May 11 and then decreasing again. Most other services maintain a stable cost trend between $0-30 from May 4, 2025, to June 1, 2025. 
