import traceback
import datetime
import utils
import chat
import os
import traceback
import datetime
import biology_agent.biology as biology
import agent
import json
import re
import random
import string
import trans

from datetime import datetime
from langchain_core.tools import tool
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.graph import START, END, StateGraph

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from langchain_core.tools import BaseTool
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_mcp_adapters.client import MultiServerMCPClient

import logging
import sys
from reportlab.lib import colors

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("biology_agent")

status_msg = []
def get_status_msg(status):
    global status_msg
    status_msg.append(status)

    if status != "end":
        status = " -> ".join(status_msg)
        return "[status]\n" + status + "..."
    else: 
        status = " -> ".join(status_msg)
        return "[status]\n" + status

response_msg = []

index = 0
def add_notification(container, message):
    global index
    container['notification'][index].info(message)
    index += 1

####################### Agent #######################
# Biology Agent
#########################################################
def get_prompt_template(prompt_name: str) -> str:
    template = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
    return template

def get_mcp_tools(tools):
    mcp_tools = []
    for tool in tools:
        name = tool.name
        description = tool.description
        description = description.replace("\n", "")
        mcp_tools.append(f"{name}: {description}")
        # logger.info(f"mcp_tools: {mcp_tools}")

    return mcp_tools

class State(TypedDict):
    full_plan: str
    messages: Annotated[list, add_messages]
    appendix: list[str]
    final_response: str
    report: str
    urls: list[str]

async def Planner(state: State, config: dict) -> dict:
    logger.info(f"###### Planner ######")
    # logger.info(f"state: {state}")

    request_id = config.get("configurable", {}).get("request_id", "")
    logger.info(f"request_id: {request_id}")

    containers = config.get("configurable", {}).get("containers", None)
    tools = config.get("configurable", {}).get("tools", None)

    mcp_tools = get_mcp_tools(tools)
    
    prompt_name = "planner"

    if chat.debug_mode == "Enable":
        containers['status'].info(get_status_msg(f"{prompt_name}"))

    system = get_prompt_template(prompt_name)
    # logger.info(f"system_prompt of planner: {system}")

    human = "{input}" 

    llm = chat.get_chat(extended_thinking="Disable")
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    chain = planner_prompt | llm 
    result = chain.invoke({
        "mcp_tools": mcp_tools,
        "input": state
    })
    logger.info(f"Planner: {result.content}")

    if chat.debug_mode == "Enable":
        add_notification(containers, result.content)

    # Update the plan into s3
    key = f"artifacts/{request_id}_plan.md"
    time = f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    chat.updata_object(key, time + result.content, 'prepend')

    output = result.content
    if output.find("<status>") != -1:
        status = output.split("<status>")[1].split("</status>")[0]
        logger.info(f"status: {status}")

        if status == "Completed":
            final_response = state["messages"][-1].content
            logger.info(f"final_response: {final_response}")

            return {
                "full_plan": result.content,
                "final_response": final_response                
            }

    return {
        "full_plan": result.content,
    }

async def to_operator(state: State, config: dict) -> str:
    logger.info(f"###### to_operator ######")
    # logger.info(f"state: {state}")

    request_id = config.get("configurable", {}).get("request_id", "")
    logger.info(f"request_id: {request_id}")

    if "final_response" in state and state["final_response"] != "":
        logger.info(f"Finished!!!")
        next = "Reporter"

        key = f"artifacts/{request_id}.md"
        body = f"# Final Response\n\n{state["final_response"]}\n\n"
        chat.updata_object(key, body, 'append')

    else:
        logger.info(f"go to Operator...")
        next = "Operator"

    return next

async def Operator(state: State, config: dict) -> dict:
    logger.info(f"###### Operator ######")
    # logger.info(f"state: {state}")
    appendix = state["appendix"] if "appendix" in state else []

    containers = config.get("configurable", {}).get("containers", None)
    tools = config.get("configurable", {}).get("tools", None)

    mcp_tools = get_mcp_tools(tools)
    
    last_state = state["messages"][-1].content
    logger.info(f"last_state: {last_state}")

    full_plan = state["full_plan"]
    logger.info(f"full_plan: {full_plan}")

    request_id = config.get("configurable", {}).get("request_id", "")
    prompt_name = "operator"

    if chat.debug_mode == "Enable":
        containers['status'].info(get_status_msg(f"{prompt_name}"))

    system = get_prompt_template(prompt_name)
    # logger.info(f"system_prompt: {system}")

    human = (
        "<full_plan>{full_plan}</full_plan>\n"
        "<tools>{mcp_tools}</tools>\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    logger.info(f"mcp_tools: {mcp_tools}")

    llm = chat.get_chat(extended_thinking="Disable")
    chain = prompt | llm 
    result = chain.invoke({
        "full_plan": full_plan,
        "mcp_tools": mcp_tools
    })
    logger.info(f"result: {result}")
    
    content = result.content
    # Remove control characters
    content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)
    # Try to extract JSON string
    try:
        # Regular expression to find JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        result_dict = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Problematic content: {content}")
        return {
            "messages": [
                HumanMessage(content="JSON parsing error occurred. Please try again.")
            ]
        }

    next = result_dict["next"]
    logger.info(f"next: {next}")

    task = result_dict["task"]
    logger.info(f"task: {task}")

    if chat.debug_mode == "Enable":
        add_notification(containers, f"{next}: {task}")

    if next == "FINISHED":
        return
    else:
        tool_info = []
        for tool in tools:
            if tool.name == next:
                tool_info.append(tool)
                logger.info(f"tool_info: {tool_info}")
                
        global status_msg, response_msg

        result, image_url, status_msg, response_msg = await agent.run_task(
            question = task, 
            tools = tool_info, 
            system_prompt = None, 
            containers = containers, 
            historyMode = "Disable", 
            previous_status_msg = status_msg, 
            previous_response_msg = response_msg)

        logger.info(f"response of Operator: {result}, {image_url}")

        if image_url:
            output_images = ""
            for url in image_url:
                output_images += f"![{task}]({url})\n\n"
            body = f"# {task}\n\n{result}\n\n{output_images}"
            
            logger.info(f"output_images: {output_images}")
            appendix.append(f"{output_images}")

            if chat.debug_mode == "Enable":
                add_notification(containers, f"{task}\n\n{body}")
        
        else:
            body = f"# {task}\n\n{result}\n\n"

            if chat.debug_mode == "Enable":
                add_notification(containers, body)

        key = f"artifacts/{request_id}_steps.md"
        time = f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        chat.updata_object(key, time + body, 'append')
        
        return {
            "messages": [
                HumanMessage(content=json.dumps(task)),
                AIMessage(content=body)
            ],
            "appendix": appendix
        }

async def create_final_report(request_id, question, body, urls):
    # report.html
    output_html = trans.trans_md_to_html(body, question)
    chat.create_object(f"artifacts/{request_id}_report.html", output_html)

    logger.info(f"url of html: {chat.path}/artifacts/{request_id}_report.html")
    urls.append(f"{chat.path}/artifacts/{request_id}_report.html")

    output = await utils.generate_pdf_report(body, request_id)
    logger.info(f"result of generate_pdf_report: {output}")
    if output: # reports/request_id.pdf         
        pdf_filename = f"artifacts/{request_id}.pdf"
        with open(pdf_filename, 'rb') as f:
            pdf_bytes = f.read()
            chat.upload_to_s3_artifacts(pdf_bytes, f"{request_id}.pdf")
        logger.info(f"url of pdf: {chat.path}/artifacts/{request_id}.pdf")
    
    urls.append(f"{chat.path}/artifacts/{request_id}.pdf")

    # report.md
    key = f"artifacts/{request_id}_report.md"
    time = f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"    
    final_result = body + "\n\n" + f"## 최종 결과\n\n"+'\n\n'.join(urls)
    
    chat.create_object(key, time + final_result)
    return urls

async def Reporter(state: State, config: dict) -> dict:
    logger.info(f"###### Reporter ######")

    prompt_name = "reporter"

    containers = config.get("configurable", {}).get("containers", None)

    if chat.debug_mode == "Enable":
        containers['status'].info(get_status_msg(f"{prompt_name}"))

    request_id = config.get("configurable", {}).get("request_id", "")    
    
    key = f"artifacts/{request_id}_steps.md"
    context = chat.get_object(key)

    logger.info(f"context: {context}")

    system_prompt=get_prompt_template(prompt_name)
    # logger.info(f"system_prompt: {system_prompt}")
    
    llm = chat.get_chat(extended_thinking="Disable")

    human = (
        "다음의 context를 바탕으로 사용자의 질문에 대한 답변을 작성합니다.\n"
        "<question>{question}</question>\n"
        "<context>{context}</context>"
    )
    reporter_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human)
        ]
    )

    question = state["messages"][0].content
    logger.info(f"question: {question}")

    prompt = reporter_prompt | llm 
    result = prompt.invoke({
        "context": context,
        "question": question
    })
    logger.info(f"result of Reporter: {result}")

    if chat.debug_mode == "Enable":
        add_notification(containers, result.content)

    if chat.debug_mode == "Enable":
        containers['status'].info(get_status_msg("end)"))

    appendix = state["appendix"] if "appendix" in state else []
    values = '\n\n'.join(appendix)
    logger.info(f"values: {values}")

    question = state["messages"][0].content
    urls = state["urls"] if "urls" in state else []
    urls = await create_final_report(request_id, question, result.content+values, urls)
    logger.info(f"urls: {urls}")
        
    return {
        "report": result.content,
        "urls": urls
    }
                
def buildBioAgent():
    workflow = StateGraph(State)

    workflow.add_node("Planner", Planner)
    workflow.add_node("Operator", Operator)
    workflow.add_node("Reporter", Reporter)

    workflow.add_edge(START, "Planner")
    workflow.add_conditional_edges(
        "Planner",
        to_operator,
        {
            "Reporter": "Reporter",
            "Operator": "Operator",
        },
    )
    workflow.add_edge("Operator", "Planner")
    workflow.add_edge("Reporter", END)

    return workflow.compile()

bio_agent = buildBioAgent()

def get_tool_info(tools, st):    
    toolList = []
    for tool in tools:
        name = tool.name
        toolList.append(name)
    
    toolmsg = ', '.join(toolList)
    st.info(f"Tools: {toolmsg}")

def initiate_report(agent,st):
    # request id
    request_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    template = open(os.path.join(os.path.dirname(__file__), f"report.html")).read()
    template = template.replace("{request_id}", request_id)
    template = template.replace("{sharing_url}", chat.path)
    key = f"artifacts/{request_id}.html"
    chat.create_object(key, template)

    report_url = chat.path + "/artifacts/" + request_id + ".html"
    logger.info(f"report_url: {report_url}")
    st.info(f"report_url: {report_url}")

    # draw a graph
    graph_diagram = agent.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
        curve_style=CurveStyle.LINEAR
    )    
    random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    image_filename = f'workflow_{random_id}.png'
    url = chat.upload_to_s3(graph_diagram, image_filename)
    logger.info(f"url: {url}")

    # add plan to report
    key = f"artifacts/{request_id}_plan.md"
    task = "실행 계획"
    output_images = f"![{task}]({url})\n\n"
    body = f"## {task}\n\n{output_images}"
    chat.updata_object(key, body, 'prepend')

    return request_id, report_url

async def run_biology_agent(query, st):
    logger.info(f"###### run_biology_agent ######")
    logger.info(f"query: {query}")

    server_params = chat.load_multiple_mcp_server_parameters()
    logger.info(f"server_params: {server_params}")

    global status_msg, response_msg
    status_msg = []
    response_msg = []
    
    async with MultiServerMCPClient(server_params) as client:
        response = ""
        with st.status("thinking...", expanded=True, state="running") as status:            
            tools = client.get_tools()

            if chat.debug_mode == "Enable":
                get_tool_info(tools, st)
                logger.info(f"tools: {tools}")
        
            request_id, report_url = initiate_report(bio_agent, st)

            containers = {
                "status": st.empty(),
                "notification": [st.empty() for _ in range(100)]
            }

            global index
            index = 0

            if chat.debug_mode == "Enable":
                containers['status'].info(get_status_msg("start"))
                
            inputs = {
                "messages": [HumanMessage(content=query)],
                "final_response": "",
                "urls": [report_url]
            }
            config = {
                "request_id": request_id,
                "recursion_limit": 50,
                "containers": containers,
                "tools": tools
            }

            value = None
            async for output in bio_agent.astream(inputs, config):
                for key, value in output.items():
                    logger.info(f"Finished running: {key}")            
            logger.info(f"value: {value}")

            if "report" in value:
                response = value["report"]
                urls = value["urls"]
            else:
                response = value["final_response"], 
                urls = value["urls"]

            logger.info(f"response: {response}")
            logger.info(f"urls: {urls}")

        if response_msg:
            with st.expander(f"수행 결과"):
                response_msgs = '\n\n'.join(response_msg)
                st.markdown(response_msgs)

        image_url = []  # todo
    
    return response, image_url, urls

