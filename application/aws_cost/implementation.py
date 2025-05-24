"""This file was generated using `langgraph-gen` version 0.0.3.

This file provides a placeholder implementation for the corresponding stub.

Replace the placeholder implementation with your own logic.
"""

from typing_extensions import TypedDict

from aws_cost.stub import CostAgent

from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END

import pandas as pd
import plotly.express as px
import plotly.io as pio
import boto3
import logging
import sys
import base64
import random
import chat
import os
import json
import traceback
import utils

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("cost_analysis")

def get_url(figure, prefix):
    # Convert fig_pie to base64 image
    img_bytes = pio.to_image(figure, format="png")
    base64_image = base64.b64encode(img_bytes).decode('utf-8')

    random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    image_filename = f'{prefix}_{random_id}.png'
    
    # Convert base64 string back to bytes for S3 upload
    image_bytes = base64.b64decode(base64_image)
    url = chat.upload_to_s3(image_bytes, image_filename)
    logger.info(f"Uploaded image to S3: {url}")
    
    return url

def get_prompt_template(prompt_name: str) -> str:
    template = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
    # logger.info(f"template: {template}")
    return template

class CostSate(TypedDict):
    messages: Annotated[list, add_messages]
    service_costs: dict
    region_costs: dict
    daily_costs: dict
    final_response: str

# Define stand-alone functions
def service_cost(state: CostSate, config) -> dict:
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
    time = f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    body = f"## {task}\n\n{output_images}"
    chat.updata_object(key, time + body, 'append')

    return {
        "messages": [AIMessage(content=body)],
        "service_costs": service_response
    }

def region_cost(state: CostSate, config) -> dict:
    logger.info(f"###### region_cost ######")

    logger.info(f"Getting cost analysis...")
    days = 30

    request_id = config.get("configurable", {}).get("request_id", "")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # cost explorer
        ce = boto3.client('ce')

        # region cost
        region_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'REGION'}]
        )
        logger.info(f"region_response: {region_response}")
    
    except Exception as e:
        logger.info(f"Error in cost analysis: {str(e)}")
        return None
    
    region_costs = pd.DataFrame([
        {
            'REGION': group['Keys'][0],
            'cost': float(group['Metrics']['UnblendedCost']['Amount'])
        }
        for group in region_response['ResultsByTime'][0]['Groups']
    ])
    logger.info(f"Region Costs: {region_costs}")

    # region cost (bar chart)
    fig_bar = px.bar(
        region_costs,
        x='REGION',
        y='cost',
        color='REGION',
        title='Region Cost',
        template='plotly_white',  # Clean background
        color_discrete_sequence=px.colors.qualitative.Set3  # Color palette
    )
    url = get_url(fig_bar, "region_costs")
    task = "AWS 리전별 사용량"
    output_images = f"![{task} 그래프]({url})\n\n"

    key = f"artifacts/{request_id}_steps.md"
    time = f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    body = f"## {task}\n\n{output_images}"
    chat.updata_object(key, time + body, 'append')

    return {
        "messages": [AIMessage(content=body)],
        "region_costs": region_response
    }

def daily_cost(state: CostSate, config) -> dict:
    logger.info(f"###### daily_cost ######")
    logger.info(f"Getting cost analysis...")
    days = 30

    request_id = config.get("configurable", {}).get("request_id", "")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # cost explorer
        ce = boto3.client('ce')

       # Daily Cost
        daily_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        logger.info(f"Daily Cost: {daily_response}")
    
    except Exception as e:
        logger.info(f"Error in cost analysis: {str(e)}")
        return None
    
    daily_costs = []
    for time_period in daily_response['ResultsByTime']:
        date = time_period['TimePeriod']['Start']
        for group in time_period['Groups']:
            daily_costs.append({
                'date': date,
                'SERVICE': group['Keys'][0],
                'cost': float(group['Metrics']['UnblendedCost']['Amount'])
            })
    
    daily_costs_df = pd.DataFrame(daily_costs)
    logger.info(f"Daily Costs: {daily_costs_df}")

    # daily trend cost (line chart)
    fig_line = px.line(
        daily_costs_df,
        x='date',
        y='cost',
        color='SERVICE',
        title='Daily Cost Trend',
        template='plotly_white',  # Clean background
        markers=True,  # Add markers to data points
        line_shape='spline'  # Smooth curve display
    )
    url = get_url(fig_line, "daily_costs")
    
    task = "AWS 일자별 사용량"
    output_images = f"![{task} 그래프]({url})\n\n"

    key = f"artifacts/{request_id}_steps.md"
    time = f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    body = f"## {task}\n\n{output_images}"
    chat.updata_object(key, time + body, 'append')

    return {
        "messages": [AIMessage(content=body)],
        "daily_costs": daily_response
    }

def generate_insight(state: CostSate, config) -> dict:
    logger.info(f"###### generate_insight ######")

    prompt_name = "cost_insight"

    request_id = config.get("configurable", {}).get("request_id", "")    

    system_prompt=get_prompt_template(prompt_name)
    logger.info(f"system_prompt: {system_prompt}")

    human = (
        "다음 AWS 비용 데이터를 분석하여 상세한 인사이트를 제공해주세요:"
        "Cost Data:"
        "<service_costs>{service_costs}</service_costs>"
        "<region_costs>{region_costs}</region_costs>"
        "<daily_costs>{daily_costs}</daily_costs>"
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
                "daily_costs": daily_costs
            }
        )
        logger.info(f"response: {response.content}")

    except Exception:
        err_msg = traceback.format_exc()
        logger.debug(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")

    return {
        "final_response": response.content
    }

def reflect_context(state: CostSate) -> dict:
    logger.info(f"###### reflect_context ######")
    return {
        # Add your state update logic here
    }

def should_end(state: CostSate) -> str:
    logger.info(f"###### should_end ######")
    if "final_response" in state and state["final_response"] != "":
        next = END
    else:
        logger.info(f"final_response is empty")
        next = END

    return next

agent = CostAgent(
    state_schema=CostSate,
    impl=[
        ("service_cost", service_cost),
        ("region_cost", region_cost),
        ("daily_cost", daily_cost),
        ("generate_insight", generate_insight),
        ("reflect_context", reflect_context),
        ("should_end", should_end),
    ],
)

cost_agent = agent.compile()

def run(request_id: str):
    logger.info(f"request_id: {request_id}")

    question = "AWS 사용량을 분석하세요."
        
    inputs = {
        "messages": [HumanMessage(content=question)],
        "final_response": ""
    }
    config = {
        "request_id": request_id,
        "recursion_limit": 50
    }

    value = None
    for output in cost_agent.stream(inputs, config):
        for key, value in output.items():
            logger.info(f"Finished running: {key}")
    
    logger.info(f"value: {value}")

    return value["final_response"]