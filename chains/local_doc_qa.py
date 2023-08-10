from langchain.embeddings.base import Embeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from textsplitter.my_markdown_splitter import my_md_split
from vectorstores import MyFAISS, MyAnalyticDB
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, CSVLoader
from configs.model_config import *
from textsplitter import ChineseTextSplitter
from typing import List
from utils import torch_gc
from tqdm import tqdm
from agent import bing_search
from langchain.docstore.document import Document
from textsplitter.zh_title_enhance import zh_title_enhance
from langchain.chains.base import Chain
from urllib.parse import quote_plus


# patch HuggingFaceEmbeddings to make it hashable
def _embeddings_hash(self):
    return hash(self.model_name)


HuggingFaceEmbeddings.__hash__ = _embeddings_hash

CONNECTION_STRING = MyAnalyticDB.connection_string_from_db_params(
    driver=os.environ.get("PG_DRIVER", "psycopg2"),
    host=os.environ.get("PG_HOST", "localhost"),
    port=int(os.environ.get("PG_PORT", "5432")),
    database=os.environ.get("PG_DATABASE", "postgres"),
    user=os.environ.get("PG_USER", "postgres"),
    password=quote_plus(os.environ.get("PG_PASSWORD", "postgres")),
)


def load_file(filepath, sentence_size=SENTENCE_SIZE, using_zh_title_enhance=ZH_TITLE_ENHANCE, url: str = ""):
    if filepath.lower().endswith(".md"):
        # loader = UnstructuredFileLoader(filepath, mode="elements")
        # docs = loader.load()
        docs = my_md_split(filepath, sentence_size=sentence_size)
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath, autodetect_encoding=True)
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".pdf"):
        # 暂且将paddle相关的loader改为动态加载，可以在不上传pdf/image知识文件的前提下使用protobuf=4.x
        from loader import UnstructuredPaddlePDFLoader
        loader = UnstructuredPaddlePDFLoader(filepath)
        # print("pdf加载完成")
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
        # print("切分完成")
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        # 暂且将paddle相关的loader改为动态加载，可以在不上传pdf/image知识文件的前提下使用protobuf=4.x
        from loader import UnstructuredPaddleImageLoader
        loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    elif filepath.lower().endswith(".csv"):
        loader = CSVLoader(filepath, encoding='gb18030')
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    if using_zh_title_enhance:
        docs = zh_title_enhance(docs)
    # write_check_file(filepath, docs)
    if len(url) > 0:
        for doc in docs:
            doc.metadata['url'] = url
    return docs


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


def generate_prompt(related_docs: List[Document],
                    query: str,
                    prompt_template: str = PROMPT_TEMPLATE, ) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


class LocalDocQA:
    llm_model_chain: Chain = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_content: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD

    def __init__(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 ):
        self.embeddings: Embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                            model_kwargs={'device': embedding_device})
        self.myAnalyticDB = MyAnalyticDB(embedding_function=self.embeddings,
                                         connection_string=CONNECTION_STRING,
                                         pre_delete_collection=False)

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 llm_model: Chain = None,
                 top_k=VECTOR_SEARCH_TOP_K,
                 ):
        self.llm_model_chain = llm_model
        self.embeddings: Embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                            model_kwargs={'device': embedding_device})
        self.top_k = top_k
        self.myAnalyticDB = MyAnalyticDB(embedding_function=self.embeddings,
                                         connection_string=CONNECTION_STRING,
                                         pre_delete_collection=False)

    def load_vector_store(self, knowledge_name):
        self.myAnalyticDB.set_collection_name(knowledge_name)
        return self.myAnalyticDB

    def create_knowledge_vector_store(self, knowledge_name: str or os.PathLike = None):
        print(f"创建 {knowledge_name}")
        _, table_is_exist = self.myAnalyticDB.create_table_if_not_exists(knowledge_name)
        return table_is_exist

    # 上传文件并创建知识库
    def init_knowledge_vector_store(self,
                                    filepath: List[str],
                                    knowledge_name: str,
                                    sentence_size=SENTENCE_SIZE, url: str = ""):
        print(f"初始化 {knowledge_name}")
        loaded_files = []
        docs = []
        for file in filepath:
            try:
                docs += load_file(file, sentence_size, url=url)
                logger.info(f"{file} 已成功加载")
                loaded_files.append(file)
            except Exception as e:
                logger.error(e)
                logger.info(f"{file} 未能成功加载")
        if len(docs) > 0:
            logger.info("文件加载完毕，正在生成向量库")
            if not knowledge_name:
                knowledge_name = LANGCHAIN_DEFAULT_KNOWLEDGE_NAME
            vector_store = self.load_vector_store(knowledge_name)
            vector_store.add_documents(docs)  # docs 为Document列表
            torch_gc()
            return knowledge_name, loaded_files
        else:
            logger.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")

            return None, []

    def one_knowledge_add(self, knowledge_name, one_title, one_content, one_content_segmentation, sentence_size):
        try:
            if not knowledge_name or not one_title or not one_content:
                logger.info("知识库添加错误，请确认知识库名字、标题、内容是否正确！")
                return None, [one_title]
            docs = [Document(page_content=one_content + "\n", metadata={"source": one_title})]
            if not one_content_segmentation:
                text_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
                docs = text_splitter.split_documents(docs)

            vector_store = self.load_vector_store(knowledge_name)
            vector_store.add_documents(docs)  # docs 为Document列表
            torch_gc()
            return knowledge_name, [one_title]
        except Exception as e:
            logger.error(e)
            return None, [one_title]

    def get_knowledge_based_answer(self, query, knowledge_name, chat_history=[], streaming: bool = STREAMING,
                                   keyword: str = ""):
        print(f"查询：知识库【{knowledge_name}】；问题【{query}】；历史【{chat_history}】；关键词【{keyword}】")
        if not knowledge_name:
            logger.error("知识库名称错误")
            return None
        vector_store = self.load_vector_store(knowledge_name)
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_content = self.chunk_content
        vector_store.score_threshold = self.score_threshold
        print("score_threshold", vector_store.score_threshold, "chunk_size", vector_store.chunk_size, "chunk_content",
              vector_store.chunk_content)
        if len(keyword) == 0:
            related_docs_with_score = vector_store.similarity_search(query, k=self.top_k)
        else:
            related_docs_with_score = vector_store.similarity_search(keyword, k=self.top_k)
        torch_gc()
        if len(related_docs_with_score) > 0:
            prompt = generate_prompt(related_docs_with_score, query)
        else:
            prompt = query

        # answer_result_stream_result：4个key的字典，prompt、history、streaming、answer_result_stream
        answer_result_stream_result = self.llm_model_chain(
            {"prompt": prompt, "history": chat_history, "streaming": streaming})

        # answer_result_stream：多个AnswerResult（models/base/base.py），AnswerResult包括history(List[List[str]])和llm_output(Optional[dict])
        # 如果streaming为True，则有多个AnswerResult，后面一个AnswerResult比前面一个多一个字符。streaming为false，只有一个AnswerResult
        # 返回的history存的每个list都代表一轮问答，每个list有两个str，前者是prompt，后者是回答。之后将prompt替换成query，回答在webui里还会加上知识出处source。
        # llm_output只有一个key "answer"，存了回答
        # 具体见ChatGLMLLMChain (models/chatglm_llm.py)的_generate_answer
        for answer_result in answer_result_stream_result['answer_result_stream']:
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query  # 原本的history[-1][0]是prompt，现在换成query
            response = {"query": query,
                        "result": resp,
                        "source_documents": related_docs_with_score}
            yield response, history

    # query      查询内容
    # vs_path    知识库路径
    # chunk_content   是否启用上下文关联
    # score_threshold    搜索匹配score阈值
    # vector_search_top_k   搜索知识库内容条数，默认搜索5条结果
    # chunk_sizes    匹配单段内容的连接上下文长度
    def get_knowledge_based_content_test(self, query, knowledge_name):
        print(f"测试：知识库 {knowledge_name}，问题 {query}")
        if not knowledge_name:
            logger.error("知识库名称错误")
            return None
        vector_store = self.load_vector_store(knowledge_name)
        vector_store.chunk_content = self.chunk_content
        vector_store.score_threshold = self.score_threshold
        vector_store.chunk_size = self.chunk_size
        print("score_threshold", vector_store.score_threshold, "chunk_size", vector_store.chunk_size, "chunk_content",
              vector_store.chunk_content)
        related_docs_with_score = vector_store.similarity_search(query, k=self.top_k)
        if not related_docs_with_score:
            response = {"query": query,
                        "source_documents": []}
            return response, ""
        torch_gc()
        prompt = "\n".join([doc.page_content for doc in related_docs_with_score])
        response = {"query": query,
                    "source_documents": related_docs_with_score}
        return response, prompt

    def get_search_result_based_answer(self, query, chat_history=[], streaming: bool = STREAMING):
        results = bing_search(query)
        result_docs = search_result2docs(results)
        prompt = generate_prompt(result_docs, query)

        answer_result_stream_result = self.llm_model_chain(
            {"prompt": prompt, "history": chat_history, "streaming": streaming})

        for answer_result in answer_result_stream_result['answer_result_stream']:
            resp = answer_result.llm_output["answer"]
            history = answer_result.history
            history[-1][0] = query
            response = {"query": query,
                        "result": resp,
                        "source_documents": result_docs}
            yield response, history

    def delete_file_from_vector_store(self,
                                      filename: str or List[str],
                                      knowledge_name):
        print(f"删除 {knowledge_name} 的文件 {filename}")
        vector_store = self.load_vector_store(knowledge_name)
        status = vector_store.delete_doc(filename)
        return status

    def update_file_from_vector_store(self,
                                      filepath: str or List[str],
                                      knowledge_name,
                                      docs: List[Document], ):
        print(f"更新 {knowledge_name} 的文件 {filepath}")
        if not knowledge_name:
            logger.error("知识库名称错误")
            return f"docs update fail"
        vector_store = self.load_vector_store(knowledge_name)
        status = vector_store.update_doc(filepath, docs)
        return status

    def list_file_from_vector_store(self,
                                    knowledge_name):
        print(f"列出 {knowledge_name} 内的文件")
        if not knowledge_name:
            logger.error("知识库名称错误")
            return None
        vector_store = self.load_vector_store(knowledge_name)
        docs = vector_store.list_docs()
        return docs

    def check_knowledge_in_collections(self, knowledge_name):
        print(f"检查 {knowledge_name} 是否存在")
        if not knowledge_name:
            logger.error("知识库名称错误")
            return None
        return self.myAnalyticDB.check_collection_if_exists(knowledge_name)

    def get_knowledge_list(self):
        print("获取knowledge列表")
        return self.myAnalyticDB.get_collections()

    def delete_knowledge(self, knowledge_name):
        print(f"删除 {knowledge_name}")
        vector_store = self.load_vector_store(knowledge_name)
        return vector_store.delete_collection()

    def change_knowledge(self, knowledge_name, new_knowledge_name):
        print(f"修改 {knowledge_name} 为 {new_knowledge_name}")
        vector_store = self.load_vector_store(knowledge_name)
        return vector_store.change_collection(new_knowledge_name)

# if __name__ == "__main__":
#     # 初始化消息
#     args = None
#     args = parser.parse_args(args=['--model-dir', '/media/checkpoint/', '--model', 'chatglm-6b', '--no-remote-model'])
#
#     args_dict = vars(args)
#     shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
#     llm_model_ins = shared.loaderLLM()
#
#     local_doc_qa = LocalDocQA()
#     local_doc_qa.init_cfg(llm_model=llm_model_ins)
#     query = "本项目使用的embedding模型是什么，消耗多少显存"
#     vs_path = "/media/gpt4-pdf-chatbot-langchain/dev-langchain-ChatGLM/vector_store/test"
#     last_print_len = 0
#     # for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
#     #                                                              vs_path=vs_path,
#     #                                                              chat_history=[],
#     #                                                              streaming=True):
#     for resp, history in local_doc_qa.get_search_result_based_answer(query=query,
#                                                                      chat_history=[],
#                                                                      streaming=True):
#         print(resp["result"][last_print_len:], end="", flush=True)
#         last_print_len = len(resp["result"])
#     source_text = [f"""出处 [{inum + 1}] {doc.metadata['source'] if doc.metadata['source'].startswith("http")
#     else os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
#                    # f"""距离：{doc.metadata['score']}\n\n"""
#                    for inum, doc in
#                    enumerate(resp["source_documents"])]
#     logger.info("\n\n" + "\n\n".join(source_text))
#     pass
