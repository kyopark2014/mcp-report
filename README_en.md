# MCP Report Agent

This repository explains how to generate various reports using MCP by creating agents. You can collect necessary data using various APIs and generate appropriate reports using LLM prompts. The generated reports can be improved using the agent's reflection pattern, which extracts improvement points and additional search keywords, and generates new reports based on them. With MCP, you can dynamically use data from various sources according to the improvement points of the report, and even if new tools are added or data formats change due to API changes, the application is not affected.

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
// ... existing code ...
```

The state used for data exchange between nodes is as follows. For detailed code, refer to [implementation.py](./application/aws_cost/implementation.py). Here, service_costs, region_costs, and daily_costs are JSON data obtained using the Boto3 API, and appendix is also a JSON file containing image files and descriptions. additonal_context stores contents obtained through reflection and MCP tool utilization. iteration means the number of reflection repetitions, and reflection contains draft revisions and additional search terms. final_response stores the final answer.

```python
// ... existing code ...
```

### Acquiring Various Data

Not all APIs are done with MCP; the basic data for the report is obtained directly using boto3. Below, information on service usage is collected using cost explorer. The period is 30 days but can be changed according to the user's purpose. After converting the service usage information to a data frame, a pie chart is drawn and the result is saved as an appendix. Since the LLM may omit URL information such as figures in the final result, it is managed separately.

```python
// ... existing code ...
```

Similarly, region and daily costs can be extracted. For detailed code, refer to [implementation.py](./application/aws_cost/implementation.py).

### Draft Generation

A draft is generated with the given data as shown below. Using additional_context with Reflection and MCP tool is effective in improving the draft. When generating a draft, the basic format is set using [cost_insight.md](./application/aws_cost/cost_insight.md) as shown below. Therefore, to maintain the basic format when generating insight, additional_context is updated using generate_insight, reflection, and mcp-tools.

```python
// ... existing code ...
```

### Identifying Improvements to the Draft

Draft improvements are performed as shown below. Intermediate results are saved in steps.md to check the progress.

```python
// ... existing code ...
```

### Acquiring Data Needed for Improvement (Reflection) with MCP

As shown below, the MCP Tools node obtains the data needed for reflection. The required data source may vary depending on the result of the reflection. By using MCP, you can select and utilize appropriate tools from various data sources.

```python
// ... existing code ...
```

The Reflection agent is configured as follows. For detailed code on the Reflection agent, refer to [reflection_agent.py](./application/aws_cost/reflection_agent.py).

```python
// ... existing code ...
```

### Generating Documents with Reflected Improvements

After storing the data obtained with MCP in the context, the draft is improved as follows.

```python
// ... existing code ...
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

<img width="692" alt="image" src="https://github.com/user-attachments/assets/567daa25-670d-49c9-b6f7-c34969d8d146" />

If you draw this as a graph diagram, you can see that it was executed as the following plan.

<img src="https://github.com/user-attachments/assets/5ac86ba4-7aec-44ac-b86a-7796f59a630f" width="250">

The execution result provides an integrated analysis of major cost drivers, abnormal patterns, optimization, and cost trends according to the given format.

![image](https://github.com/user-attachments/assets/5c8ead4f-cdf5-4caa-8f8d-daa6ef130c73)

The analysis result for the obtained service cost is as follows.

![image](https://github.com/user-attachments/assets/b1ccbc5e-566f-49f2-a539-9463dbdea6f0)

This pie chart shows the distribution of AWS service costs. Amazon OpenSearch Service accounts for 41.9%, followed by Amazon Elastic Compute Cloud - Compute (21.1%), Amazon Bedrock (13%), and EC2 - Other (10.6%). The remaining services (Amazon Virtual Private Cloud, Elastic Load Balancing, CloudWatch, Cost Explorer, etc.) each account for less than 5%.

The cost by region is as follows.

![image](https://github.com/user-attachments/assets/dd996bb3-3897-4a83-82cf-9d038cc769eb)

In the graph showing AWS costs by region, the us-west-2 (Oregon, USA) region shows the highest cost at about $1,900, while other regions (NoRegion, ap-northeast-1, ap-northeast-2, ap-southeast-1, eu-west-1, global, us-east-1, us-east-2) show much lower costs.

The daily cost is as follows.

![image](https://github.com/user-attachments/assets/35736080-3dd3-46ca-a1ef-0a577ffce27d)

Among various AWS services (Amplify, Cost Explorer, ECR, EC2, CloudFront, etc.), AmazonCloudWatch shows a unique pattern, rising sharply to about $95 around May 11 and then decreasing again. Most other services maintain a stable cost trend between $0-30 from May 4, 2025, to June 1, 2025. 