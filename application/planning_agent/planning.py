import knowledge_base as kb
import utils
import operator
import chat
import traceback
import logging
import sys
import utils
import os
import random
import string
import langgraph_agent
import strands_agent
import trans

from typing_extensions import Annotated, TypedDict
from typing import List, Tuple,Literal
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langchain_core.tools import BaseTool
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_mcp_adapters.client import MultiServerMCPClient
from datetime import datetime

import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("planning-agent")

index = 0
def add_notification(containers, message):
    global index
    containers['notification'][index].info(message)
    index += 1

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
mcp_server_info = {}

####################### LangGraph #######################
# Planning Agent
#########################################################
class State(TypedDict):
    input: str
    plan: list[str]
    past_steps: Annotated[List[Tuple], operator.add]
    info: Annotated[List[Tuple], operator.add]
    response: list[str]
    answer: str
    urls: list[str]

async def plan_node(state: State, config):
    logger.info(f"###### plan ######")
    logger.info(f"input: {state['input']}")

    containers = config.get("configurable", {}).get("containers", None)
    request_id = config.get("configurable", {}).get("request_id", "")

    if chat.debug_mode == "Enable":
        containers['status'].info(get_status_msg(f"plan"))
        add_notification(containers, f"계획을 생성합니다.")
    
    system = (
        "당신은 user의 question을 해결하기 위해 step by step plan을 생성하는 AI agent입니다."  
        "생성된 계획은 <plan> tag를 붙여주세요."              
        
        "문제를 충분히 이해하고, 문제 해결을 위한 계획을 다음 형식으로 4단계 이하의 계획을 세웁니다."                
        "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
        "1. [질문을 해결하기 위한 단계]"
        "2. [질문을 해결하기 위한 단계]"
        "..."                
    )
    
    human = (
        "{question}"
    )
                        
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )
    llm = chat.get_chat(extended_thinking="Disable")
    planner = planner_prompt | llm
    response = planner.invoke({
        "question": state["input"]
    })
    logger.info(f"response: {response.content}")
    result = response.content
    
    output = result[result.find('<plan>')+6:result.find('</plan>')]
    logger.info(f"plan: {output}")
    
    plan = output.strip().replace('\n\n', '\n')
    planning_steps = plan.split('\n')
    logger.info(f"planning_steps: {planning_steps}")

    # Update the plan into s3
    key = f"artifacts/{request_id}_plan.md"
    time = f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    plan = f"{time}\n{planning_steps}"
    chat.updata_object(key, f"{time}\n{plan}", 'prepend')

    if chat.debug_mode=="Enable":
        add_notification(containers, f"Plan: {planning_steps}")
    
    return {
        "input": state["input"],
        "plan": planning_steps
    }

async def execute_node(state: State, config):
    logger.info(f"###### execute ######")
    plan = state["plan"]
    logger.info(f"plan: {plan}")
    
    containers = config.get("configurable", {}).get("containers", None)
    request_id = config.get("configurable", {}).get("request_id", "")
    tools = config.get("configurable", {}).get("tools", None)
    agent_type = config.get("configurable", {}).get("agent_type", "LangGraph")

    task = plan[0]
    logger.info(f"task: {task}")
    if chat.debug_mode == "Enable":
        containers['status'].info(get_status_msg(f"execute"))    
        add_notification(containers, f"현재 계획: {task}")

    if agent_type == "LangGraph":
        global status_msg, response_msg
        result, image_url, status_msg, response_msg = await langgraph_agent.run_task(
                question = task, 
                tools = tools, 
                system_prompt = None, 
                containers = containers, 
                historyMode = "Disable", 
                previous_status_msg = status_msg, 
                previous_response_msg = response_msg
        )
    else:
        mcp_servers = get_mcp_server_list()
        logger.info(f"mcp_servers: {mcp_servers}")
        result, image_url, status_msg, response_msg = await strands_agent.run_task(
            question = task, 
            mcp_servers = mcp_servers, 
            system_prompt = None, 
            containers = containers, 
            historyMode = "Disable", 
            previous_status_msg = status_msg, 
            previous_response_msg = response_msg)
    
    subresult = f"{task}:\n\n{result}"
    logger.info(f"subresult: {subresult}, image_url: {image_url}")

    key = f"artifacts/{request_id}_steps.md"
    time = f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"        
    chat.updata_object(key, time + subresult, 'append')

    if chat.debug_mode=="Enable":
        add_notification(containers, subresult)
    
    return {
        "input": state["input"],
        "plan": state["plan"],
        "info": [subresult],
        "past_steps": [plan[0]],
    }
        
async def replan_node(state: State, config):
    logger.info(f"#### replan ####")
    logger.info(f"state of replan node: {state}")

    if len(state["plan"]) == 1:
        logger.info(f"last plan: {state['plan']}")
        logger.info(f"final info: {state['info'][-1]}")
        return {"response":state['info'][-1]}    
    
    containers = config.get("configurable", {}).get("containers", None)
    request_id = config.get("configurable", {}).get("request_id", "")
    
    if chat.debug_mode=="Enable":
        containers['status'].info(get_status_msg(f"replan"))
        add_notification(containers, f"새로운 계획을 생성합니다.")
    
    system = (
        "당신은 복잡한 문제를 해결하기 위해 step by step plan을 생성하는 AI agent입니다."
        "당신은 다음의 Question에 대한 적절한 답변을 얻고자합니다."
    )        
    human = (
        "Question: {input}"
                    
        "당신의 원래 계획은 아래와 같습니다." 
        "Original Plan:"
        "{plan}"

        "완료한 계획는 아래와 같습니다."
        "Past steps:"
        "{past_steps}"
        
        "당신은 Original Plan의 원래 계획을 상황에 맞게 수정하세요."
        "계획에 아직 해야 할 단계만 추가하세요. 이전에 완료한 계획는 계획에 포함하지 마세요."                
        "수정된 계획에는 <plan> tag를 붙여주세요."
        "만약 더 이상 계획을 세우지 않아도 Question에 답변할 수 있다면, 최종 결과로 Question에 대한 답변을 <result> tag를 붙여 전달합니다."
        
        "수정된 계획의 형식은 아래와 같습니다."
        "각 단계는 반드시 한줄의 문장으로 AI agent가 수행할 내용을 명확히 나타냅니다."
        "1. [질문을 해결하기 위한 단계]"
        "2. [질문을 해결하기 위한 단계]"
        "..."         
    )                   
    
    replanner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )     
    
    llm = chat.get_chat(extended_thinking="Disable")
    replanner = replanner_prompt | llm
    
    response = replanner.invoke({
        "input": state["input"],
        "plan": state["plan"],
        "past_steps": state["past_steps"]
    })
    logger.info(f"replanner output:: {response.content}")
    result = response.content

    if result.find('<plan>') == -1:
        return {"response":response.content}
    else:
        output = result[result.find('<plan>')+6:result.find('</plan>')]
        logger.info(f"plan output: {output}")

        plans = output.strip().replace('\n\n', '\n')
        planning_steps = plans.split('\n')
        logger.info(f"planning_steps: {planning_steps}")

        key = f"artifacts/{request_id}_plan.md"
        time = f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"        
        chat.updata_object(key, f"{time}\n{planning_steps}", 'append')

        if chat.debug_mode=="Enable":
            add_notification(containers, f"plan: {planning_steps}")

        return {"plan": planning_steps}
    
async def should_end(state: State) -> Literal["continue", "end"]:
    logger.info(f"#### should_end ####")
    logger.info(f"state: {state}")
    
    if "response" in state and state["response"]:
        logger.info(f"response: {state['response']}")
        next = "end"
    else:
        logger.info(f"plan: {state['plan']}")
        next = "continue"
    logger.info(f"should_end response: {next}")
    
    return next

async def create_final_report(request_id, question, body, urls):
    logger.info(f"#### create_final_report ####")
    logger.info(f"request_id: {request_id}")
    logger.info(f"question: {question}")
    logger.info(f"body: {body}")
    logger.info(f"urls: {urls}")

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
    
async def final_answer(state: State, config) -> str:
    logger.info(f"#### final_answer ####")
    
    # get final answer
    context = "".join(f"{info}\n" for info in state['info'])
    logger.info(f"context: {context}")
    
    query = state['input']
    logger.info(f"query: {query}")

    containers = config.get("configurable", {}).get("containers", None)
    request_id = config.get("configurable", {}).get("request_id", "")

    if chat.debug_mode=="Enable":
        containers['status'].info(get_status_msg(f"final_answer"))
        add_notification(containers, f"최종 답변을 생성합니다.")
    
    if chat.isKorean(query)==True:
        system = (
            "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
            "답변의 이유를 풀어서 명확하게 설명합니다."
            #"결과는 <result> tag를 붙여주세요."
        )
    else: 
        system = (
            "Here is pieces of context, contained in <context> tags."
            "Provide a concise answer to the question at the end."
            "Explains clearly the reason for the answer."
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            #"Put it in <result> tags."
        )

    human = (
        "<context>"
        "{context}"
        "</context>"

        "Question: {input}"
    )
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
                
    llm = chat.get_chat(extended_thinking="Disable")
    chain = prompt | llm
    
    try: 
        response = chain.invoke(
            {
                "context": context,
                "input": query,
            }
        )
        result = response.content

        if result.find('<result>')==-1:
            output = result
        else:
            output = result[result.find('<result>')+8:result.find('</result>')]
            
        logger.info(f"output: {output}")

        if chat.debug_mode=="Enable":
            add_notification(containers, f"최종결과: {output}")

        question = state["input"]
        urls = state["urls"] if "urls" in state else []
        urls = await create_final_report(request_id, question, output, urls)
        logger.info(f"urls: {urls}")

        return {"answer": output, "urls": urls}
        
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")      
        
        return {"answer": err_msg}
      
def buildPlanAndExecute():
    workflow = StateGraph(State)
    workflow.add_node("planner", plan_node)
    workflow.add_node("executor", execute_node)
    workflow.add_node("replaner", replan_node)
    workflow.add_node("final_answer", final_answer)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "replaner")
    workflow.add_conditional_edges(
        "replaner",
        should_end,
        {
            "continue": "executor",
            "end": "final_answer",
        },
    )
    workflow.add_edge("final_answer", END)

    return workflow.compile()

# workflow
planning_app = buildPlanAndExecute()    

def get_tool_info(tools, st):    
    toolList = []
    for tool in tools:
        name = tool.name
        toolList.append(name)
    
    toolmsg = ', '.join(toolList)
    st.info(f"Tools: {toolmsg}")

def initiate_report(agent, st):
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

def get_mcp_server_name(too_name):
    mcp_server_name = {}
    for server_name, tools in mcp_server_info:
        tool_names = [tool.name for tool in tools]
        logger.info(f"{server_name}: {tool_names}")
        for name in tool_names:
            mcp_server_name[name] = server_name
    return mcp_server_name[too_name]

def get_mcp_server_list():
    server_lists = []
    for server_name, tools in mcp_server_info:
        server_lists.append(server_name)
    return server_lists
    
async def run_planning_agent(query, agent_type, st):
    logger.info(f"###### run_planning_agent ######")
    logger.info(f"query: {query}")

    server_params = langgraph_agent.load_multiple_mcp_server_parameters()
    logger.info(f"server_params: {server_params}")

    global status_msg, response_msg, mcp_server_info
    status_msg = []
    response_msg = []
    
    async with MultiServerMCPClient(server_params) as client:
        response = ""
        with st.status("thinking...", expanded=True, state="running") as status:       
            mcp_server_info = client.server_name_to_tools.items()

            tools = client.get_tools()

            if chat.debug_mode == "Enable":
                get_tool_info(tools, st)

            request_id, report_url = initiate_report(planning_app, st)
            
            containers = {
                "tools": st.empty(),
                "status": st.empty(),
                "notification": [st.empty() for _ in range(100)]
            }            
            if chat.debug_mode == "Enable":
                containers['status'].info(get_status_msg("start"))
                                                        
            inputs = {
                "input": query
            }
            config = {
                "request_id": request_id,
                "recursion_limit": 50,
                "containers": containers,
                "tools": tools,
                "agent_type": agent_type
            }    

            value = None
            async for output in planning_app.astream(inputs, config):
                for key, value in output.items():
                    logger.info(f"Finished running: {key}")
            
            logger.info(f"value: {value}")
            response = value["answer"]
            logger.info(f"response: {response}")

            urls = value["urls"] if "urls" in value else []
            logger.info(f"urls: {urls}")

        if urls:
            with st.expander(f"수행 결과"):
                st.markdown('\n\n'.join(urls))

        image_url = []
    
    return response, image_url, urls

