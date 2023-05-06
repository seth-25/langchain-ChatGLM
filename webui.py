import gradio as gr
import os
import shutil
from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 6

# LLM input history length
LLM_HISTORY_LEN = 3


def get_vs_list():
    if not os.path.exists(VS_ROOT_PATH):
        return []
    return ["新建知识库"] + os.listdir(VS_ROOT_PATH)


vs_list = get_vs_list()

embedding_model_dict_list = list(embedding_model_dict.keys())

llm_model_dict_list = list(llm_model_dict.keys())

local_doc_qa = LocalDocQA()


def get_answer(query, vs_path, history, mode):
    if mode == "知识库问答":
        resp, history = local_doc_qa.get_knowledge_based_answer(
            query=query, chat_history=history)
        source = "".join([f"""<details> <summary>出处[{i + 1}] <b>所属文件：</b>{doc.metadata["metadata"]["source_id"]}</summary>
{doc.page_content}
</details>""" for i, doc in enumerate(resp["source_documents"])])
        history[-1][-1] += source
    else:
        resp = local_doc_qa.llm(query)
        history = history + [[query, resp + ("\n\n当前知识库为空，如需基于知识库进行问答，请先加载知识库后，再进行提问。" if mode == "知识库问答" else "")]]
    return history, ""


def update_status(history, status):
    history = history + [[None, status]]
    print(status)
    return history


def init_model():
    try:
        local_doc_qa.init_cfg()
    except Exception as e:
        print(e)
        return """模型未成功加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""


def reinit_model(llm_model, embedding_model, llm_history_len, use_ptuning_v2, top_k, history):
    try:
        local_doc_qa.init_cfg(top_k=top_k)
    except Exception as e:
        print(e)
        model_status = """模型未成功重新加载，请到页面左上角"模型配置"选项卡中重新选择后点击"加载模型"按钮"""
    return history + [[None, model_status]]


def change_vs_name_input(vs_id):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), VS_ROOT_PATH + vs_id


def change_mode(mode):
    if mode == "知识库问答":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def add_vs_name(vs_name, vs_list, chatbot):
    if vs_name in vs_list:
        chatbot = chatbot + [[None, "与已有知识库名称冲突，请重新选择其他名称后提交"]]
        return gr.update(visible=True), vs_list, chatbot
    else:
        chatbot = chatbot + [
            [None, f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """]]
        return gr.update(visible=True, choices=vs_list + [vs_name], value=vs_name), vs_list + [vs_name], chatbot


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}

.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉langchain-ChatBot WebUI🎉

"""

init_message = """欢迎使用 langchain-ChatBot Web UI！

"""

model_status = init_model()

with gr.Blocks(css=block_css) as demo:
    vs_path, file_status, model_status, vs_list = gr.State(""), gr.State(""), gr.State(model_status), gr.State(vs_list)
    gr.Markdown(webui_title)
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status.value]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交",
                                   ).style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["LLM 对话", "知识库问答"],
                                label="请选择使用模式",
                                value="知识库问答", )
                query.submit(get_answer,
                                 [query, vs_path, chatbot, mode],
                                 [chatbot, query],
                                 )

demo.queue(concurrency_count=3
           ).launch(server_name='0.0.0.0',
                    server_port=7860,
                    show_api=False,
                    share=False,
                    inbrowser=False)
