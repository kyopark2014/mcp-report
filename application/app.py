import streamlit as st 
import chat
import json
import knowledge_base as kb
import traceback
import mcp_config as mcp_config
import logging
import sys
import aws_cost.implementation as aws_cost
import random
import string
import os
import pwd
import asyncio
import biology_agent.biology as bio_agent
import planning_agent.planning as planning
import langgraph_agent
import strands_agent

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("streamlit")

try:
    user_info = pwd.getpwuid(os.getuid())
    username = user_info.pw_name
    home_dir = user_info.pw_dir
    logger.info(f"Username: {username}")
    logger.info(f"Home directory: {home_dir}")
except (ImportError, KeyError):
    username = "root"
    logger.info(f"Username: {username}")
    pass  

if username == "root":
    environment = "system"
else:
    environment = "user"
logger.info(f"environment: {environment}")

# title
st.set_page_config(page_title='Report', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

# CSS for adjusting sidebar width and checkbox visibility
st.markdown("""
    <style>
    /* Sidebar width adjustment */
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 400px !important;
    }

    /* Improve checkbox style */
    .stAlert input[type="checkbox"] {
        width: 20px !important;
        height: 20px !important;
        margin: 0 5px 0 0 !important;
        border: 2px solid #666 !important;
        border-radius: 3px !important;
        appearance: none !important;
        -webkit-appearance: none !important;
        background-color: white !important;
    }
    
    .stAlert input[type="checkbox"]:checked {
        background-color: white !important;
        border-color: #666 !important;
        position: relative;
    }
    
    .stAlert input[type="checkbox"]:checked::after {
        content: "✓";
        color: #ff0000;
        position: absolute;
        left: 4px;
        top: -3px;
        font-size: 16px;
        font-weight: bold;
        text-shadow: 1px 1px 1px rgba(0,0,0,0.1);
    }
    
    /* Checkbox label style */
    .stAlert label {
        font-size: 14px !important;
        color: #333 !important;
        margin-left: 5px !important;
        font-weight: 500 !important;
    }
    </style>
""", unsafe_allow_html=True)

mode_descriptions = {
    "일상적인 대화": [
        "대화이력을 바탕으로 챗봇과 일상의 대화를 편안히 즐길수 있습니다."
    ],
    "RAG": [
        "Bedrock Knowledge Base를 이용해 구현한 RAG로 필요한 정보를 검색합니다."
    ],
    "Agent": [
        "MCP를 활용한 Agent를 이용합니다. 왼쪽 메뉴에서 필요한 MCP를 선택하세요."
    ],
    "Agent (Chat)": [
        "MCP를 활용한 Agent를 이용합니다. 채팅 히스토리를 이용해 interative한 대화를 즐길 수 있습니다."
    ],
    "Biology Agent": [
        "Biology Agent를 이용합니다. 왼쪽 메뉴에서 필요한 MCP를 선택하세요."
    ],
    "Planning Agent": [
        "Planning agent를 이용하여 복잡한 문제를 해결합니다."
    ],
    "비용 분석 Agent": [
        "Cloud 사용에 대한 분석을 수행합니다."
    ]
}

def load_image_generator_config():
    config = None
    try:
        with open("image_generator_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            logger.info(f"loaded image_generator_config: {config}")
    except FileNotFoundError:
        config = {"seed_image": ""}
        with open("image_generator_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        logger.info("Create new image_generator_config.json")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")    
    return config

def update_seed_image_url(url):
    with open("image_generator_config.json", "w", encoding="utf-8") as f:
        config = {"seed_image": url}
        json.dump(config, f, ensure_ascii=False, indent=4)

seed_config = load_image_generator_config()
logger.info(f"seed_config: {seed_config}")
seed_image_url = seed_config.get("seed_image", "") if seed_config else ""
logger.info(f"seed_image_url from config: {seed_image_url}")

uploaded_seed_image = None
with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Amazon Bedrock을 이용해 다양한 형태의 대화를 구현합니다." 
        "여기에서는 MCP를 이용해 RAG를 구현하고, Multi agent를 이용해 다양한 기능을 구현할 수 있습니다." 
        "또한 번역이나 문법 확인과 같은 용도로 사용할 수 있습니다."
        "주요 코드는 LangChain과 LangGraph를 이용해 구현되었습니다.\n"
        "상세한 코드는 [Github](https://github.com/kyopark2014/mcp-report)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")
    
    # radio selection
    mode = st.radio(
        label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "RAG", "Agent", "Agent (Chat)", "Biology Agent", "비용 분석 Agent", "Planning Agent"], index=2
    )   
    st.info(mode_descriptions[mode][0])
    
    # mcp selection
    mcp = ""
    if mode=='Agent' or mode=='Agent (Chat)' or mode=='Biology Agent' or mode=='Planning Agent' or mode=='비용 분석 Agent':
        # MCP Config JSON input
        st.subheader("⚙️ MCP Config")

        # Change radio to checkbox
        if environment == "user":        
            mcp_options = [
                "default", "code interpreter", "aws document", "aws cost", "aws cli", 
                "use_aws","aws cloudwatch", "aws storage", "aws diagram", "image generation",
                "knowledge base", "tavily", "perplexity", "ArXiv", "wikipedia", 
                "filesystem", "terminal", "text editor", "context7", "puppeteer", 
                "playwright", "firecrawl", "obsidian", "airbnb", 
                "pubmed", "chembl", "clinicaltrial", "arxiv-manual", "tavily-manual",
                "사용자 설정"
            ]
        else:
            mcp_options = [ 
                "default", "code interpreter", "aws document", "aws cost", "aws cli", 
                "use_aws", "aws cloudwatch", "aws storage", "aws diagram", "image generation",
                "knowledge base", "tavily", "ArXiv", "wikipedia", 
                "filesystem", "terminal", "text editor", "playwright", "airbnb", 
                "pubmed", "chembl", "clinicaltrial", "arxiv-manual", "tavily-manual",
                "사용자 설정"
            ]
        mcp_selections = {}
        default_selections = ["default", "tavily-manual", "use_aws", "code interpreter", "filesystem"]

        if mode=='Agent' or mode=='Agent (Chat)' or mode=='Planning Agent' or mode=='비용 분석 Agent' or mode=='Biology Agent':
            agent_type = st.radio(
                label="Agent 타입을 선택하세요. ",options=["LangGraph", "Strands"], index=0
            )

        if mode == "Biology Agent":
            default_selections = ["pubmed", "chembl", "clinicaltrial", "arxiv-manual", "tavily-manual"]

        with st.expander("MCP 옵션 선택", expanded=True):            
            # Create 2 columns
            col1, col2 = st.columns(2)
            
            # Divide options into two groups
            mid_point = len(mcp_options) // 2
            first_half = mcp_options[:mid_point]
            second_half = mcp_options[mid_point:]
            
            # Display first group in first column
            with col1:
                for option in first_half:
                    default_value = option in default_selections
                    mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)
            
            # Display second group in second column
            with col2:
                for option in second_half:
                    default_value = option in default_selections
                    mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)
        
        if not any(mcp_selections.values()):
            mcp_selections["default"] = True

        if mcp_selections["사용자 설정"]:
            mcp_info = st.text_area(
                "MCP 설정을 JSON 형식으로 입력하세요",
                value=mcp,
                height=150
            )
            logger.info(f"mcp_info: {mcp_info}")

            if mcp_info:
                mcp_config.mcp_user_config = json.loads(mcp_info)
                logger.info(f"mcp_user_config: {mcp_config.mcp_user_config}")
        
        if "image generation" in mcp_selections:
            enable_seed = st.checkbox("Seed Image", value=False)

            if enable_seed:
                st.subheader("🌇 이미지 업로드")
                uploaded_seed_image = st.file_uploader("이미지 생성을 위한 파일을 선택합니다.", type=["png", "jpg", "jpeg"])

                if uploaded_seed_image:
                    url = chat.upload_to_s3(uploaded_seed_image.getvalue(), uploaded_seed_image.name)
                    logger.info(f"uploaded url: {url}")
                    seed_image_url = url
                    update_seed_image_url(seed_image_url)
                
                given_image_url = st.text_input("또는 이미지 URL을 입력하세요", value=seed_image_url, key="seed_image_input")       
                if given_image_url and given_image_url != seed_image_url:       
                    logger.info(f"given_image_url: {given_image_url}")
                    seed_image_url = given_image_url
                    update_seed_image_url(seed_image_url)                    
            else:
                if seed_image_url:
                    logger.info(f"remove seed_image_url")
                    update_seed_image_url("") 
        else:
            enable_seed = False
            if seed_image_url:
                logger.info(f"remove seed_image_url")
                update_seed_image_url("") 

        mcp_servers = [server for server, is_selected in mcp_selections.items() if is_selected]

    # model selection box
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        ("Nova Premier", 'Nova Pro', 'Nova Lite', 'Nova Micro', 'Claude 4 Opus', 'Claude 4 Sonnet', 'Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3.0 Sonnet', 'Claude 3.5 Haiku'), index=7
    )

    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'
    #print('debugMode: ', debugMode)

    # multi region check box
    select_multiRegion = st.checkbox('Multi Region', value=False)
    multiRegion = 'Enable' if select_multiRegion else 'Disable'
    #print('multiRegion: ', multiRegion)

    select_reasoning = st.checkbox('Reasoning', value=False)
    reasoningMode = 'Enable' if select_reasoning and modelName=='Claude 3.7 Sonnet' else 'Disable'
    logger.info(f"reasoningMode: {reasoningMode}")

    # RAG grading
    select_grading = st.checkbox('Grading', value=False)
    gradingMode = 'Enable' if select_grading else 'Disable'
    # logger.info(f"gradingMode: {gradingMode}")

    uploaded_file = None
    if mode=='이미지 분석':
        st.subheader("🌇 이미지 업로드")
        uploaded_file = st.file_uploader("이미지 요약을 위한 파일을 선택합니다.", type=["png", "jpg", "jpeg"])
    elif mode=='RAG' or mode=="Agent" or mode=="Agent (Chat)":
        st.subheader("📋 문서 업로드")
        # print('fileId: ', chat.fileId)
        uploaded_file = st.file_uploader("RAG를 위한 파일을 선택합니다.", type=["pdf", "txt", "py", "md", "csv", "json"], key=chat.fileId)

    chat.update(modelName, debugMode, multiRegion, reasoningMode, gradingMode)

    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")
    # logger.info(f"clear_button: {clear_button}")

st.title('🔮 '+ mode)

if clear_button==True:
    chat.initiate()
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "images" in message:                
                for url in message["images"]:
                    logger.info(f"url: {url}")

                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)
            st.markdown(message["content"])

display_chat_messages()

def show_references(reference_docs):
    if debugMode == "Enable" and reference_docs:
        with st.expander(f"답변에서 참조한 {len(reference_docs)}개의 문서입니다."):
            for i, doc in enumerate(reference_docs):
                st.markdown(f"**{doc.metadata['name']}**: {doc.page_content}")
                st.markdown("---")

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []        
    uploaded_file = None
    
    st.session_state.greetings = False
    chat.clear_chat_history()
    st.rerun()    

# Preview the uploaded image in the sidebar
file_name = ""
state_of_code_interpreter = False
if uploaded_file is not None and clear_button==False:
    logger.info(f"uploaded_file.name: {uploaded_file.name}")
    if uploaded_file.name:
        logger.info(f"csv type? {uploaded_file.name.lower().endswith(('.csv'))}")

    if uploaded_file.name and not mode == '이미지 분석':
        chat.initiate()

        if debugMode=='Enable':
            status = '선택한 파일을 업로드합니다.'
            logger.info(f"status: {status}")
            st.info(status)

        file_name = uploaded_file.name
        logger.info(f"uploading... file_name: {file_name}")
        file_url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
        logger.info(f"file_url: {file_url}")

        kb.sync_data_source()  # sync uploaded files
            
        status = f'선택한 "{file_name}"의 내용을 요약합니다.'
        # my_bar = st.sidebar.progress(0, text=status)
        
        if debugMode=='Enable':
            logger.info(f"status: {status}")
            st.info(status)
    
        msg = chat.get_summary_of_uploaded_file(file_name, st)
        st.session_state.messages.append({"role": "assistant", "content": f"선택한 문서({file_name})를 요약하면 아래와 같습니다.\n\n{msg}"})    
        logger.info(f"msg: {msg}")

        st.write(msg)

    if uploaded_file and clear_button==False and mode == '이미지 분석':
        st.image(uploaded_file, caption="이미지 미리보기", use_container_width=True)

        file_name = uploaded_file.name
        url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
        logger.info(f"url: {url}")

if seed_image_url and clear_button==False and enable_seed==True:
    st.image(seed_image_url, caption="이미지 미리보기", use_container_width=True)
    logger.info(f"preview: {seed_image_url}")
    
if clear_button==False and mode == '비용 분석 Agent':
    request_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    template = open(os.path.join(os.path.dirname(__file__), f"aws_cost/report.html")).read()
    template = template.replace("{request_id}", request_id)
    template = template.replace("{sharing_url}", chat.path)
    key = f"artifacts/{request_id}.html"
    chat.create_object(key, template)
    
    report_url = chat.path + "/artifacts/" + request_id + ".html"
    logger.info(f"report_url: {report_url}")
    st.info(f"report_url: {report_url}")
    
    # show status and response
    status_container = st.empty()
    response_container = st.empty()
    key_container = st.empty()
    
    response = aws_cost.run(request_id, mcp_servers, status_container, response_container, key_container)
    logger.info(f"response: {response}")

    if aws_cost.response_msg:
        with st.expander(f"수행 결과"):
            response_msgs = '\n\n'.join(aws_cost.response_msg)  
            st.markdown(response_msgs)

    st.write(response)

    # if urls:
    #     with st.expander(f"최종 결과"):
    #         url_msg = '\n\n'.join(urls)
    #         st.markdown(url_msg)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Always show the chat input
if mode != "비용 분석 Agent" and (prompt := st.chat_input("메시지를 입력하세요.")):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")
    logger.info(f"prompt: {prompt}")

    with st.chat_message("assistant"):
        if mode == '일상적인 대화':
            stream = chat.general_conversation(prompt)            
            response = st.write_stream(stream)
            logger.info(f"response: {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})

            chat.save_chat_history(prompt, response)

        elif mode == 'RAG':
            with st.status("running...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_rag_with_knowledge_base(prompt, st)                           
                st.write(response)
                logger.info(f"response: {response}")

                st.session_state.messages.append({"role": "assistant", "content": response})

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Agent' or mode == 'Agent (Chat)':
            sessionState = ""

            if mode == 'Agent':
                history_mode = "Disable"
            else:
                history_mode = "Enable"

            with st.status("thinking...", expanded=True, state="running") as status:
                containers = {
                    "tools": st.empty(),
                    "status": st.empty(),
                    "notification": [st.empty() for _ in range(500)]
                }
                if agent_type == "LangGraph":
                    response, image_url = asyncio.run(langgraph_agent.run_agent(prompt, mcp_servers, history_mode, containers))    
                else:
                    response, image_url = asyncio.run(strands_agent.run_agent(prompt, [], mcp_servers, history_mode, containers))

            # if langgraph_agent.response_msg:
            #     with st.expander(f"수행 결과"):
            #         st.markdown('\n\n'.join(langgraph_agent.response_msg))

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "images": image_url if image_url else []
            })

            if agent_type == "LangGraph":
                st.write(response)

            for url in image_url:
                    logger.info(f"url: {url}")
                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)

        elif mode == 'Biology Agent':
            sessionState = ""
            chat.references = []
            chat.image_url = []
            
            response, image_url, urls = asyncio.run(bio_agent.run_biology_agent(prompt, mcp_servers, agent_type, st))

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "images": image_url if image_url else [],
                "urls": urls if urls else []
            })

            st.write(response)
            for url in image_url:
                logger.info(f"url: {url}")
                file_name = url[url.rfind('/')+1:]
                st.image(url, caption=file_name, use_container_width=True)           

            if urls:
                with st.expander(f"최종 결과"):
                    url_msg = '\n\n'.join(urls)
                    st.markdown(url_msg)

        elif mode == 'Planning Agent':
            sessionState = ""
            chat.references = []
            chat.image_url = []
            
            response, image_url, urls = asyncio.run(planning.run_planning_agent(prompt, mcp_servers, agent_type, st))

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "images": image_url if image_url else [],
                "urls": urls if urls else []
            })

            st.write(response)
            for url in image_url:
                logger.info(f"url: {url}")
                file_name = url[url.rfind('/')+1:]
                st.image(url, caption=file_name, use_container_width=True)           

            if urls:
                with st.expander(f"최종 결과"):
                    url_msg = '\n\n'.join(urls)
                    st.markdown(url_msg)