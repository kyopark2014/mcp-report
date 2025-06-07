import traceback
import datetime
import utils
import chat
import os
import traceback
import datetime
import agent
import json
import re
import random
import string

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

####################### Agent #######################
# Biology Agent
#########################################################
def get_prompt_template(prompt_name: str) -> str:
    template = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
    return template

# async def run_agent(query, tools, system_prompt, status_container, response_container, key_container, historyMode):
#     @tool
#     async def planning_agent(query: str) -> str:
#         """
#         A specialized planning agent that analyzes the research query and determines 
#         which tools and databases should be used for the investigation.
        
#         Args:
#             query: The research question about drug discovery or target proteins
            
#         Returns:
#             A structured plan outlining tools to use and search queries for each database
#         """
#         try:
#             prompt_name = "planner"
#             planning_system=get_prompt_template(prompt_name)
#             logger.info(f"planning_system: {planning_system}")
            
#             planner_tool = []
#             for tool in tools:
#                 if tool.name == "planning_agent":
#                     planner_tool.append(tool)
#                     break
#             # logger.info(f"planner_tool: {planner_tool}")

#             result, image_url = await agent.run(
#                 query, 
#                 planner_tool, 
#                 planning_system, 
#                 status_container, 
#                 response_container, 
#                 key_container, 
#                 historyMode
#             )
#             logger.info(f"result: {result}")
            
#             return result
#         except Exception as e:
#             logger.error(f"Error in planning agent: {e}")
#             return f"Error in planning agent: {str(e)}"
        
#     @tool
#     async def synthesis_agent(research_results: str) -> str:
#         """
#         Specialized agent for synthesizing research findings into a comprehensive report.
        
#         Args:
#             research_results: Combined results from all research agents
            
#         Returns:
#             A comprehensive, structured scientific report
#         """
#         try:
#             # Create a synthesis agent
#             system_prompt = """
#             You are a specialized synthesis agent for drug discovery research. Your role is to:
            
#             1. Integrate findings from multiple research databases (Arxiv, PubMed, ChEMBL, ClinicalTrials)
#             2. Create a comprehensive, coherent scientific report
#             3. Highlight key insights, connections, and opportunities
#             4. Organize information in a structured, accessible format
#             5. Include proper citations and references
            
#             Your reports should follow this structure:
#             1. Executive Summary (300 words)
#             2. Target Overview (biological function, structure, disease mechanisms)
#             3. Research Landscape (latest findings and research directions)
#             4. Drug Development Status (known compounds, clinical trials)
#             5. References (comprehensive listing of all sources)
#             """
            
#             # Ask synthesis agent to create a report
#             synthesis_prompt = f"""
#             Create a comprehensive scientific report based on the following research findings:
            
#             {research_results}
            
#             Follow the required report structure:
#             1. Executive Summary (300 words)
#             2. Target Overview
#             3. Research Landscape
#             4. Drug Development Status
#             5. References
#             """

#             response = await agent.run(synthesis_prompt, tools, system_prompt, status_container, response_container, key_container, historyMode)
            
#             return str(response)

#         except Exception as e:
#             logger.error(f"Error in synthesis agent: {e}")
#             return f"Error in synthesis agent: {str(e)}"

async def generate_pdf_report(report_content: str, filename: str) -> str:
    """
    Generates a PDF report from the research findings.
    
    Args:
        report_content: The content to be converted into PDF format
        filename: Base name for the generated PDF file
        
    Returns:
        A message indicating the result of PDF generation
    """
    logger.info(f'###### generate_pdf_report ######')
    
    try:
        # Ensure directory exists
        os.makedirs("reports", exist_ok=True)
        
        # Set up the PDF file
        filepath = f"reports/{filename}.pdf"
        logger.info(f"filepath: {filepath}")

        doc = SimpleDocTemplate(filepath, pagesize=letter)
        
        # Register TTF font directly (specify path to NanumGothic font file)
        font_path = "assets/NanumGothic-Regular.ttf"  # Change to actual TTF file path
        pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
        
        # Create styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Normal_KO', 
                                fontName='NanumGothic', 
                                fontSize=10,
                                spaceAfter=12))  # 문단 간격 증가
        styles.add(ParagraphStyle(name='Heading1_KO', 
                                fontName='NanumGothic', 
                                fontSize=16,
                                spaceAfter=20,  # 제목 후 여백 증가
                                textColor=colors.HexColor('#0000FF')))  # 파란색
        styles.add(ParagraphStyle(name='Heading2_KO', 
                                fontName='NanumGothic', 
                                fontSize=14,
                                spaceAfter=16,  # 제목 후 여백 증가
                                textColor=colors.HexColor('#0000FF')))  # 파란색
        styles.add(ParagraphStyle(name='Heading3_KO', 
                                fontName='NanumGothic', 
                                fontSize=12,
                                spaceAfter=14,  # 제목 후 여백 증가
                                textColor=colors.HexColor('#0000FF')))  # 파란색
        
        # Process content
        elements = []
        lines = report_content.split('\n')
        
        for line in lines:
            if line.startswith('# '):
                elements.append(Paragraph(line[2:], styles['Heading1_KO']))
            elif line.startswith('## '):
                elements.append(Paragraph(line[3:], styles['Heading2_KO']))
            elif line.startswith('### '):
                elements.append(Paragraph(line[4:], styles['Heading3_KO']))
            elif line.strip():  # Skip empty lines
                elements.append(Paragraph(line, styles['Normal_KO']))
        
        # Build PDF
        doc.build(elements)
        
        return f"PDF report generated successfully: {filepath}"
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        
        # Fallback to text file
        try:
            text_filepath = f"reports/{filename}.txt"
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            return f"PDF generation failed. Saved as text file instead: {text_filepath}"
        except Exception as text_error:
            return f"Error generating report: {str(e)}. Text fallback also failed: {str(text_error)}"

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

async def Planner(state: State, config: dict) -> dict:
    logger.info(f"###### Planner ######")
    # logger.info(f"state: {state}")

    request_id = config.get("configurable", {}).get("request_id", "")
    logger.info(f"request_id: {request_id}")

    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)
    key_container = config.get("configurable", {}).get("key_container", None)
    tools = config.get("configurable", {}).get("tools", None)

    mcp_tools = get_mcp_tools(tools)
    
    prompt_name = "planner"

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg(f"{prompt_name}"))

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

    prompt = planner_prompt | llm 
    result = prompt.invoke({
        "mcp_tools": mcp_tools,
        "input": state
    })
    logger.info(f"Planner: {result.content}")

    if chat.debug_mode == "Enable":
        key_container.info(result.content)

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

    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)    
    key_container = config.get("configurable", {}).get("key_container", None)
    tools = config.get("configurable", {}).get("tools", None)

    mcp_tools = get_mcp_tools(tools)
    
    last_state = state["messages"][-1].content
    logger.info(f"last_state: {last_state}")

    full_plan = state["full_plan"]
    logger.info(f"full_plan: {full_plan}")

    request_id = config.get("configurable", {}).get("request_id", "")
    prompt_name = "operator"

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg(f"{prompt_name}"))

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
        response_container.info(f"{next}: {task}")

    if next == "FINISHED":
        return
    else:
        tool_info = []
        for tool in tools:
            if tool.name == next:
                tool_info.append(tool)
                logger.info(f"tool_info: {tool_info}")
                
        global status_msg, response_msg

        result, image_url, status_msg, response_msg = await agent.run_manus(
            question = task, 
            tools = tool_info, 
            system_prompt = None, 
            status_container = status_container, 
            response_container = response_container, 
            key_container = key_container, 
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

            response_container.info(f"{task}\n\n{body[:500]}")
        
        else:
            body = f"# {task}\n\n{result}\n\n"

            response_container.info(body[:500])

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

async def Reporter(state: State, config: dict) -> dict:
    logger.info(f"###### Reporter ######")

    prompt_name = "reporter"

    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg(f"{prompt_name}"))

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
        response_container.info(result.content)

    key = f"artifacts/{request_id}_report.md"
    time = f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    appendix = state["appendix"] if "appendix" in state else []
    values = '\n\n'.join(appendix)
    logger.info(f"values: {values}")

    chat.create_object(key, time + result.content + values)

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg("end)"))

    await generate_pdf_report(result.content + values, request_id)

    return {
        "report": result.content
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

async def run(question: str, tools: list[BaseTool], status_container, response_container, key_container, request_id):
    logger.info(f"request_id: {request_id}")

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg("start"))
        
    inputs = {
        "messages": [HumanMessage(content=question)],
        "final_response": ""
    }
    config = {
        "request_id": request_id,
        "recursion_limit": 50,
        "status_container": status_container,
        "response_container": response_container,
        "key_container": key_container,
        "tools": tools
    }

    # draw a graph
    graph_diagram = bio_agent.get_graph().draw_mermaid_png(
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

    value = None
    async for output in bio_agent.astream(inputs, config):
        for key, value in output.items():
            logger.info(f"Finished running: {key}")
    
    logger.info(f"value: {value}")

    if "report" in value:
        return value["report"]
    else:
        return value["final_response"]

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
            key_container = st.empty()
            response_container = st.empty()
                                            
            response = await run(query, tools, status_container, response_container, key_container, request_id)
            logger.info(f"response: {response}")

        if response_msg:
            with st.expander(f"수행 결과"):
                response_msgs = '\n\n'.join(response_msg)
                st.markdown(response_msgs)

        st.markdown(response)

        image_url = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "images": image_url if image_url else []
        })
    
    return response, image_url

