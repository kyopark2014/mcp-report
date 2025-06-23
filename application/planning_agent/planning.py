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
import agent

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

async def plan_node(state: State, config):
    logger.info(f"###### plan ######")
    logger.info(f"input: {state['input']}")

    plan_container = config.get("configurable", {}).get("plan_container", None)
    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)    

    status_container.info(get_status_msg(f"plan"))

    if chat.debug_mode=="Enable":
        status_container.info(f"계획을 생성합니다. 요청사항: {state['input']}")
    
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
    plan_container.info(f"Plan: {planning_steps}")

    if chat.debug_mode=="Enable":
        response_container.info(f"생성된 계획: {planning_steps}")
    
    return {
        "input": state["input"],
        "plan": planning_steps
    }

def get_mcp_tools(tools):
    mcp_tools = []
    for tool in tools:
        name = tool.name
        description = tool.description
        description = description.replace("\n", "")
        mcp_tools.append(f"{name}: {description}")
        # logger.info(f"mcp_tools: {mcp_tools}")
    return mcp_tools

async def execute_node(state: State, config):
    logger.info(f"###### execute ######")
    logger.info(f"input: {state['input']}")
    plan = state["plan"]
    logger.info(f"plan: {plan}")
    
    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)
    key_container = config.get("configurable", {}).get("key_container", None)
    tools = config.get("configurable", {}).get("tools", None)

    status_container.info(get_status_msg(f"execute"))    

    task = plan[0]
    logger.info(f"task: {task}")
    if chat.debug_mode == "Enable":
        key_container.info(f"계획을 수행합니다. 현재 계획 {task}")

    global status_msg, response_msg
    result, image_url, status_msg, response_msg = await agent.run_task(
            question = task, 
            tools = tools, 
            system_prompt = None, 
            status_container = status_container, 
            response_container = response_container, 
            key_container = key_container, 
            historyMode = "Disable", 
            previous_status_msg = status_msg, 
            previous_response_msg = response_msg
    )
    
    subresult = f"{task}:\n\n{result}"
    logger.info(f"subresult: {subresult}, image_url: {image_url}")

    if chat.debug_mode=="Enable":
        response_container.info(subresult)
    
    # print('plan: ', state["plan"])
    # print('past_steps: ', task)        
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
    
    status_container = config.get("configurable", {}).get("status_container", None)
    plan_container = config.get("configurable", {}).get("plan_container", None)
    key_container = config.get("configurable", {}).get("key_container", None)
    
    status_container.info(get_status_msg(f"replan"))

    if chat.debug_mode=="Enable":
        key_container.info(f"새로운 계획을 생성합니다.")
    
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

        if chat.debug_mode=="Enable":
            plan_container.info(f"plan: {planning_steps}")

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
    
async def final_answer(state: State, config) -> str:
    logger.info(f"#### final_answer ####")
    
    # get final answer
    context = "".join(f"{info}\n" for info in state['info'])
    logger.info(f"context: {context}")
    
    query = state['input']
    logger.info(f"query: {query}")

    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)
    key_container = config.get("configurable", {}).get("key_container", None)

    status_container.info(get_status_msg(f"final_answer"))

    if chat.debug_mode=="Enable":
        key_container.info(f"최종 답변을 생성합니다.")
    
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
            response_container.info(f"output: {output}")

        return {"answer": output}
        
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

async def run(question: str, tools: list[BaseTool], plan_container, status_container, response_container, key_container, request_id, report_url):
    logger.info(f"request_id: {request_id}")

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg("start"))
        
    # inputs = {
    #     # "messages": [HumanMessage(content=question)],
    #     "input": question,
    #     "final_response": "",
    #     "urls": [report_url]
    # }
    inputs = {
        # "messages": [HumanMessage(content=question)],
        "input": question
    }
    config = {
        "request_id": request_id,
        "recursion_limit": 50,
        "plan_container": plan_container,
        "status_container": status_container,
        "response_container": response_container,
        "key_container": key_container,
        "tools": tools
    }

    # initiate
    global contentList, reference_docs
    contentList = []
    reference_docs = []
    
    # draw a graph
    graph_diagram = planning_app.get_graph().draw_mermaid_png(
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

    # for output in planning_app.stream(inputs, config):   
    #     for key, value in output.items():
    #         logger.info(f"Finished: {key}")
    #         #print("value: ", value)            
    # logger.info(f"value: {value}")

    # reference = ""
    # if reference_docs:
    #     reference = chat.get_references(reference_docs)

    # return value["answer"]+reference, reference_docs

    value = None
    async for output in planning_app.astream(inputs, config):
        for key, value in output.items():
            logger.info(f"Finished running: {key}")
    
    logger.info(f"value: {value}")

    return value["answer"]
    
async def run_planning_agent(query, st):
    logger.info(f"###### run_planning_agent ######")
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

            status_container = st.empty()      
            plan_container = st.empty()
            key_container = st.empty()
            response_container = st.empty()
                                            
            # to do: urls
            urls = []
            response = await run(query, tools, plan_container, status_container, response_container, key_container, request_id, report_url)
            logger.info(f"response: {response}")
            logger.info(f"urls: {urls}")

        if response_msg:
            with st.expander(f"수행 결과"):
                response_msgs = '\n\n'.join(response_msg)
                st.markdown(response_msgs)

        image_url = []
    
    return response, image_url, urls

