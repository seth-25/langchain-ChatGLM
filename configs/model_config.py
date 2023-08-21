import torch.cuda
import torch.backends
import os
import logging
import uuid

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

LANGCHAIN_DEFAULT_EMBEDDING_DIM = 1024
LANGCHAIN_DEFAULT_KNOWLEDGE_NAME = "langchain_document"
LANGCHAIN_DEFAULT_COLLECTIONS_NAME = "langchain_collections"

# 在以下字典中修改属性值，以指定本地embedding模型存储位置
# 如将 "text2vec": "GanymedeNil/text2vec-large-chinese" 修改为 "text2vec": "User/Downloads/text2vec-large-chinese"
# 此处请写绝对路径

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec-paraphrase": "shibing624/text2vec-base-chinese-paraphrase",
    "text2vec-sentence": "shibing624/text2vec-base-chinese-sentence",
    "text2vec-multilingual": "shibing624/text2vec-base-multilingual",
    "m3e-small": "moka-ai/m3e-small",
    "m3e-base": "moka-ai/m3e-base",
    "m3e-large": "moka-ai/m3e-large",
    "bge-small-zh": "BAAI/bge-small-zh",
    "bge-base-zh": "BAAI/bge-base-zh",
    "bge-large-zh": "BAAI/bge-large-zh"
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"  # dim=1024
# EMBEDDING_MODEL = "m3e-base"  # dim=768

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# supported LLM models
# llm_model_dict 处理了loader的一些预设行为，如加载位置，模型名称，模型处理器实例
# 在以下字典中修改属性值，以指定本地 LLM 模型存储位置
# 如将 "chatglm-6b" 的 "local_model_path" 由 None 修改为 "User/Downloads/chatglm-6b"
# 此处请写绝对路径,且路径中必须包含repo-id的模型名称，因为FastChat是以模型名匹配的
llm_model_dict = {
    "chatglm-6b-int4-qe": {
        "name": "chatglm-6b-int4-qe",
        "pretrained_model_name": "THUDM/chatglm-6b-int4-qe",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm-6b-int4": {
        "name": "chatglm-6b-int4",
        "pretrained_model_name": "THUDM/chatglm-6b-int4",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm-6b-int8": {
        "name": "chatglm-6b-int8",
        "pretrained_model_name": "THUDM/chatglm-6b-int8",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm-6b": {
        "name": "chatglm-6b",
        "pretrained_model_name": "THUDM/chatglm-6b",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    # langchain-ChatGLM 用户“帛凡” @BoFan-tunning 基于ChatGLM-6B 训练并提供的权重合并模型和 lora 权重文件 chatglm-fitness-RLHF
    # 详细信息见 HuggingFace 模型介绍页 https://huggingface.co/fb700/chatglm-fitness-RLHF
    # 使用该模型或者lora权重文件，对比chatglm-6b、chatglm2-6b、百川7b，甚至其它未经过微调的更高参数的模型，在本项目中，总结能力可获得显著提升。
    "chatglm-fitness-RLHF": {
        "name": "chatglm-fitness-RLHF",
        "pretrained_model_name": "fb700/chatglm-fitness-RLHF",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm2-6b": {
        "name": "chatglm2-6b",
        "pretrained_model_name": "THUDM/chatglm2-6b",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm2-6b-32k": {
        "name": "chatglm2-6b-32k",
        "pretrained_model_name": "THUDM/chatglm2-6b-32k",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    # 注：chatglm2-cpp已在mac上测试通过，其他系统暂不支持
    "chatglm2-cpp": {
        "name": "chatglm2-cpp",
        "pretrained_model_name": "cylee0909/chatglm2cpp",
        "local_model_path": None,
        "provides": "ChatGLMCppLLMChain"
    },
    "chatglm2-6b-int4": {
        "name": "chatglm2-6b-int4",
        "pretrained_model_name": "THUDM/chatglm2-6b-int4",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatglm2-6b-int8": {
        "name": "chatglm2-6b-int8",
        "pretrained_model_name": "THUDM/chatglm2-6b-int8",
        "local_model_path": None,
        "provides": "ChatGLMLLMChain"
    },
    "chatyuan": {
        "name": "chatyuan",
        "pretrained_model_name": "ClueAI/ChatYuan-large-v2",
        "local_model_path": None,
        "provides": "MOSSLLMChain"
    },
    "moss": {
        "name": "moss",
        "pretrained_model_name": "fnlp/moss-moon-003-sft",
        "local_model_path": None,
        "provides": "MOSSLLMChain"
    },
    "moss-int4": {
        "name": "moss",
        "pretrained_model_name": "fnlp/moss-moon-003-sft-int4",
        "local_model_path": None,
        "provides": "MOSSLLM"
    },
    "vicuna-13b-hf": {
        "name": "vicuna-13b-hf",
        "pretrained_model_name": "vicuna-13b-hf",
        "local_model_path": None,
        "provides": "LLamaLLMChain"
    },
    "vicuna-7b-hf": {
        "name": "vicuna-13b-hf",
        "pretrained_model_name": "vicuna-13b-hf",
        "local_model_path": None,
        "provides": "LLamaLLMChain"
    },
    # 直接调用返回requests.exceptions.ConnectionError错误，需要通过huggingface_hub包里的snapshot_download函数
    # 下载模型，如果snapshot_download还是返回网络错误，多试几次，一般是可以的，
    # 如果仍然不行，则应该是网络加了防火墙(在服务器上这种情况比较常见)，基本只能从别的设备上下载，
    # 然后转移到目标设备了.
    "bloomz-7b1": {
        "name": "bloomz-7b1",
        "pretrained_model_name": "bigscience/bloomz-7b1",
        "local_model_path": None,
        "provides": "MOSSLLMChain"

    },
    # 实测加载bigscience/bloom-3b需要170秒左右，暂不清楚为什么这么慢
    # 应与它要加载专有token有关
    "bloom-3b": {
        "name": "bloom-3b",
        "pretrained_model_name": "bigscience/bloom-3b",
        "local_model_path": None,
        "provides": "MOSSLLMChain"

    },
    "baichuan-7b": {
        "name": "baichuan-7b",
        "pretrained_model_name": "baichuan-inc/baichuan-7B",
        "local_model_path": None,
        "provides": "MOSSLLMChain"
    },
    # llama-cpp模型的兼容性问题参考https://github.com/abetlen/llama-cpp-python/issues/204
    "ggml-vicuna-13b-1.1-q5": {
        "name": "ggml-vicuna-13b-1.1-q5",
        "pretrained_model_name": "lmsys/vicuna-13b-delta-v1.1",
        # 这里需要下载好模型的路径,如果下载模型是默认路径则它会下载到用户工作区的
        # /.cache/huggingface/hub/models--vicuna--ggml-vicuna-13b-1.1/
        # 还有就是由于本项目加载模型的方式设置的比较严格，下载完成后仍需手动修改模型的文件名
        # 将其设置为与Huggface Hub一致的文件名
        # 此外不同时期的ggml格式并不兼容，因此不同时期的ggml需要安装不同的llama-cpp-python库，且实测pip install 不好使
        # 需要手动从https://github.com/abetlen/llama-cpp-python/releases/tag/下载对应的wheel安装
        # 实测v0.1.63与本模型的vicuna/ggml-vicuna-13b-1.1/ggml-vic13b-q5_1.bin可以兼容
        "local_model_path": f'''{"/".join(os.path.abspath(__file__).split("/")[:3])}/.cache/huggingface/hub/models--vicuna--ggml-vicuna-13b-1.1/blobs/''',
        "provides": "LLamaLLMChain"
    },

    # 通过 fastchat 调用的模型请参考如下格式
    "fastchat-chatglm-6b": {
        "name": "chatglm-6b",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "chatglm-6b",
        "local_model_path": None,
        "provides": "FastChatOpenAILLMChain",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLMChain"
        "api_base_url": "http://localhost:8000/v1",  # "name"修改为fastchat服务中的"api_base_url"
        "api_key": "EMPTY"
    },
    # 通过 fastchat 调用的模型请参考如下格式
    "fastchat-chatglm-6b-int4": {
        "name": "chatglm-6b-int4",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "chatglm-6b-int4",
        "local_model_path": None,
        "provides": "FastChatOpenAILLMChain",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLMChain"
        "api_base_url": "http://localhost:8001/v1",  # "name"修改为fastchat服务中的"api_base_url"
        "api_key": "EMPTY"
    },
    "fastchat-chatglm2-6b": {
        "name": "chatglm2-6b",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "chatglm2-6b",
        "local_model_path": None,
        "provides": "FastChatOpenAILLMChain",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLMChain"
        "api_base_url": "http://localhost:8000/v1"  # "name"修改为fastchat服务中的"api_base_url"
    },

    # 通过 fastchat 调用的模型请参考如下格式
    "fastchat-vicuna-13b-hf": {
        "name": "vicuna-13b-hf",  # "name"修改为fastchat服务中的"model_name"
        "pretrained_model_name": "vicuna-13b-hf",
        "local_model_path": None,
        "provides": "FastChatOpenAILLMChain",  # 使用fastchat api时，需保证"provides"为"FastChatOpenAILLMChain"
        "api_base_url": "http://localhost:8000/v1",  # "name"修改为fastchat服务中的"api_base_url"
        "api_key": "EMPTY"
    },
    # 调用chatgpt时如果报出： urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='api.openai.com', port=443):
    #  Max retries exceeded with url: /v1/chat/completions
    # 则需要将urllib3版本修改为1.25.11
    # 如果依然报urllib3.exceptions.MaxRetryError: HTTPSConnectionPool，则将https改为http
    # 参考https://zhuanlan.zhihu.com/p/350015032

    # 如果报出：raise NewConnectionError(
    # urllib3.exceptions.NewConnectionError: <urllib3.connection.HTTPSConnection object at 0x000001FE4BDB85E0>:
    # Failed to establish a new connection: [WinError 10060]
    # 则是因为内地和香港的IP都被OPENAI封了，需要切换为日本、新加坡等地
    "openai-chatgpt-3.5": {
        "name": "gpt-3.5-turbo",
        "pretrained_model_name": "gpt-3.5-turbo",
        "provides": "FastChatOpenAILLMChain",
        "local_model_path": None,
        "api_base_url": "https://api.openai.com/v1",
        "api_key": ""
    },

}

# LLM 名称
# LLM_MODEL = "chatglm2-6b-32k"
# LLM_MODEL = "chatglm2-6b"
LLM_MODEL = "chatglm-fitness-RLHF"

# LLM的最大token数
MAX_LENGTH = 20480

# LLM回答多样性，越低越精确相关，越高越发散
TEMPERATURE = 0.05
TOP_P = 0.3

# 传入LLM的历史记录长度/对话长度
HISTORY_LEN = 0

# 量化加载8bit 模型
LOAD_IN_8BIT = False
# Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.
BF16 = False
# 本地lora存放的位置
LORA_DIR = "loras/"

# LORA的名称，如有请指定为列表

LORA_NAME = ""
USE_LORA = True if LORA_NAME else False

# LLM streaming reponse
STREAMING = True

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False
PTUNING_DIR = './ptuning-v2'
# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 知识库文件暂存路径
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"

PROMPT_TEMPLATE = """【已知信息】{context} 

【问题】{question}

【指令】根据已知信息，详细和准确的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。"""

# PROMPT_TEMPLATE = """
# 根据已知信息，简洁和专业的来回答用户的问题。不允许在答案中添加编造成分，不要让【】内的内容出现在答案里，答案请使用中文。
#
# 示例：
#
# 用户输入：如何设置白名单
#
# 已知信息：你好
#
# 答案：根据已知信息无法回答该问题
#
# 用户输入：如何设置白名单
#
# 已知信息：
# 【下文与(设置白名单)有关】AnalyticDB PostgreSQL版实例默认禁止所有外部IP访问，连接并使用实例前，请先将客户端的IP地址或IP地址段加入AnalyticDB PostgreSQL版的白名单。
# 【下文与(设置白名单)有关】此步骤为可选步骤，如果您希望通过本地的IDE环境访问数据库，则需要设置白名单。如果您希望通过数据管理DMS访问数据库，则可跳过此步骤。
# 【下文与(设置白名单,前提条件)有关】已根据快速入门，完成了[创建实例]。
# 【下文与(设置白名单,操作步骤)有关】1. 登录云原生数据仓库AnalyticDB PostgreSQL版控制台。
# 2. 在控制台左上角，选择实例所在地域。
# 3. 找到目标实例，单击实例ID。
# 【下文与(设置白名单,操作步骤)有关】4. 在基本信息页面的右上方，单击白名单设置。
# 5. 单击default分组右侧的修改。
# 6. 在组内白名单下方的文本框中填写需要加入白名单的IP地址或IP地址段。
# 7. 单击确定。
#
# 答案：
# 设置白名单的步骤如下：
# 1. 登录云原生数据仓库AnalyticDB PostgreSQL版控制台。
# 2. 在控制台左上角，选择实例所在地域。
# 3. 找到目标实例，单击实例ID。
# 4. 在基本信息页面的右上方，单击白名单设置。
# 5. 单击default分组右侧的修改。
# 6. 在组内白名单下方的文本框中填写需要加入白名单的IP地址或IP地址段。
# 7. 单击确定。
#
# 用户输入：{question}
#
# 已知信息：{context}
#
# 答案：
# """

# 缓存知识库数量,如果是ChatGLM2,ChatGLM2-int4,ChatGLM2-int8模型若检索效果不好可以调成’10’
CACHED_VS_NUM = 1

# 文本分句长度
SENTENCE_SIZE = 150
# 文本重叠长度
SENTENCE_OVERLAP = 0

# 匹配后单段上下文长度
CHUNK_SIZE = 600

# 知识库检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 3

# 知识检索内容距离 Score, 数值范围约为0-1100，如果为0，则不生效，建议设置为500左右，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 0

NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

FLAG_USER_NAME = uuid.uuid4().hex

logger.info(f"""
loading model config
llm device: {LLM_DEVICE}
embedding device: {EMBEDDING_DEVICE}
dir: {os.path.dirname(os.path.dirname(__file__))}
flagging username: {FLAG_USER_NAME}
""")

# 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = False

# Bing 搜索必备变量
# 使用 Bing 搜索需要使用 Bing Subscription Key,需要在azure port中申请试用bing search
# 具体申请方式请见
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource
# 使用python创建bing api 搜索实例详见:
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
# 注意不是bing Webmaster Tools的api key，

# 此外，如果是在服务器上，报Failed to establish a new connection: [Errno 110] Connection timed out
# 是因为服务器加了防火墙，需要联系管理员加白名单，如果公司的服务器的话，就别想了GG
BING_SUBSCRIPTION_KEY = ""

# 是否开启中文标题加强，以及标题增强的相关配置
# 通过增加标题判断，判断哪些文本为标题，并在metadata中进行标记；
# 然后将文本与往上一级的标题进行拼合，实现文本信息的增强。
ZH_TITLE_ENHANCE = False

# 中文按句切分后，是否需要合并较短的句子
CHINESE_TEXT_SPLITTER_MERGE_SPLIT = False

MD_TITLE_ENHANCE = True  # 将markdown标题和文本融合
MD_TITLE_ENHANCE_ADD_FILENAME = True  # 是否将文件名也加入markdown标题
REMOVE_TITLE = False  # 向模型输入提问时，移除拼接文本重复的标题，在开启上下文拼接功能，且开启MD_TITLE_ENHANCE或ZH_TITLE_ENHANCE时才有效
# MD_TITLE_SPLIT = 1  # 上下文拼接时，几级标题不同就不再拼接，值为1～6
MD_REPLACE_CODE_AND_URL = True  # 切分前将code和url写入metatdata，并在原文中用占位符代替，避免被切分。回答时再将占位符换回来。


SORT_BY_DISTANCE = False  # 将提供的多条知识按照score排序，可能会打乱输入顺序，但是当信息量过大（无关信息多时），优先给模型有关的信息有助于理解
