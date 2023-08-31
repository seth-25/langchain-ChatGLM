import os

from langchain.document_loaders import TextLoader, CSVLoader, UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from configs.model_config import (
    embedding_model_dict,
    KB_ROOT_PATH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE, MD_REPLACE_CODE_AND_URL, MD_TITLE_ENHANCE
)
from functools import lru_cache
import importlib
from text_splitter import zh_title_enhance, ChineseTextSplitter
from text_splitter.markdown_header_splitter import MarkdownHeaderTextSplitter
from text_splitter.markdown_splitter import md_headers, MyMarkdownTextSplitter, _md_title_enhance, \
    _md_write_code_url_in_metadata, _md_code_url_replace


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_vs_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store")


def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)


def list_kbs_from_folder():
    return [f for f in os.listdir(KB_ROOT_PATH)
            if os.path.isdir(os.path.join(KB_ROOT_PATH, f))]


def list_docs_from_folder(kb_name: str):
    doc_path = get_doc_path(kb_name)
    return [file for file in os.listdir(doc_path)
            if os.path.isfile(os.path.join(doc_path, file))]


@lru_cache(1)
def load_embeddings(model: str, device: str):
    if model == "text-embedding-ada-002":  # openai text-embedding-ada-002
        embeddings = OpenAIEmbeddings(openai_api_key=embedding_model_dict[model], chunk_size=CHUNK_SIZE)
    elif 'bge-' in model:
        embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_dict[model],
                                              model_kwargs={'device': device},
                                              query_instruction="为这个句子生成表示以用于检索相关文章：")
        if model == "bge-large-zh-noinstruct":  # bge large -noinstruct embedding
            embeddings.query_instruction = ""
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[model], model_kwargs={'device': device})
    return embeddings


LOADER_DICT = {"UnstructuredFileLoader": ['.eml', '.html', '.json', '.md', '.msg', '.rst',
                                          '.rtf', '.txt', '.xml',
                                          '.doc', '.docx', '.epub', '.odt', '.pdf',
                                          '.ppt', '.pptx', '.tsv'],  # '.pdf', '.xlsx', '.csv'
               "CSVLoader": [".csv"],
               "PyPDFLoader": [".pdf"],
               }
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass


class KnowledgeFile:
    def __init__(
            self,
            filename: str,
            knowledge_base_name: str
    ):
        self.kb_name = knowledge_base_name
        self.filename = filename
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.ext}")
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.docs = None
        self.document_loader_name = get_LoaderClass(self.ext)

        # TODO: 增加依据文件格式匹配text_splitter
        self.text_splitter_name = None

    def md_file2text(self):
        # 获取文本
        loader = TextLoader(self.filepath)
        markdown_document = loader.load()[0]
        page_content = markdown_document.page_content

        url_dict = {}
        code_dict = {}
        if MD_REPLACE_CODE_AND_URL:
            # 先将url和code替换成占位符
            page_content = _md_code_url_replace(page_content, url_dict, code_dict)

        # 每段文本前加如标题
        if MD_TITLE_ENHANCE:
            # 按标题拆分，并将标题写入metadata
            header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers, return_each_line=False)
            docs = header_splitter.split_text(page_content)

            # 按separators递归拆分
            text_splitter = MyMarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE,
                                                   keep_separator=True)
            docs = text_splitter.split_documents(docs)

            docs = _md_title_enhance(docs, self.filepath)
        else:
            # 按separators递归拆分
            text_splitter = MyMarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE,
                                                   keep_separator=True)
            docs = text_splitter.split_documents([markdown_document])

        if MD_REPLACE_CODE_AND_URL:
            # 根据占位符，将url和code写入metadata
            _md_write_code_url_in_metadata(docs, url_dict, code_dict)

        for doc in docs:
            doc.metadata["source"] = self.filepath
        return docs

    def file2text(self, using_zh_title_enhance=ZH_TITLE_ENHANCE):
        print(self.document_loader_name)

        # todo 以下是旧版的调用方式，可等后续社区增加自定义切分器再改调用形式，Document粒度的metadata内容可能需要split后再写入，无法在load阶段完成
        # todo loader文件夹下的内容也是旧版的，如果新版用其他调用loader的形式，可更换后删除loader文件夹
        if self.ext == ".md":
            self.text_splitter_name = "MarkdownTextSplitter"
            return self.md_file2text()
        elif self.ext == ".txt":
            loader = TextLoader(self.filepath, autodetect_encoding=True)
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=CHUNK_SIZE)
            docs = loader.load_and_split(textsplitter)
            self.text_splitter_name = "ChineseTextSplitter"
        elif self.ext == ".pdf":
            # 暂且将paddle相关的loader改为动态加载，可以在不上传pdf/image知识文件的前提下使用protobuf=4.x
            from text_splitter.loader import UnstructuredPaddlePDFLoader
            loader = UnstructuredPaddlePDFLoader(self.filepath)
            textsplitter = ChineseTextSplitter(pdf=True, sentence_size=CHUNK_SIZE)
            docs = loader.load_and_split(textsplitter)
            self.text_splitter_name = "ChineseTextSplitter"
        elif self.ext == ".jpg" or self.ext == ".png":
            # 暂且将paddle相关的loader改为动态加载，可以在不上传pdf/image知识文件的前提下使用protobuf=4.x
            from text_splitter.loader import UnstructuredPaddleImageLoader
            loader = UnstructuredPaddleImageLoader(self.filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=CHUNK_SIZE)
            docs = loader.load_and_split(text_splitter=textsplitter)
            self.text_splitter_name = "ChineseTextSplitter"
        elif self.ext == ".csv":
            loader = CSVLoader(self.filepath, encoding='gb18030')
            docs = loader.load()
        else:
            loader = UnstructuredFileLoader(self.filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False, sentence_size=CHUNK_SIZE)
            docs = loader.load_and_split(text_splitter=textsplitter)
            self.text_splitter_name = "ChineseTextSplitter"
        if using_zh_title_enhance:
            docs = zh_title_enhance(docs)
        return docs


    """以下是社区版的调用方式"""
    def file2text_bak(self, using_zh_title_enhance=ZH_TITLE_ENHANCE):
        print(self.document_loader_name)
        try:
            document_loaders_module = importlib.import_module('langchain.document_loaders')
            DocumentLoader = getattr(document_loaders_module, self.document_loader_name)
        except Exception as e:
            print(e)
            document_loaders_module = importlib.import_module('langchain.document_loaders')
            DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")
        if self.document_loader_name == "UnstructuredFileLoader":
            loader = DocumentLoader(self.filepath, autodetect_encoding=True)
        else:
            loader = DocumentLoader(self.filepath)

        try:
            if self.text_splitter_name is None:
                text_splitter_module = importlib.import_module('langchain.text_splitter')
                TextSplitter = getattr(text_splitter_module, "SpacyTextSplitter")
                text_splitter = TextSplitter(
                    pipeline="zh_core_web_sm",
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=OVERLAP_SIZE,
                )
                self.text_splitter_name = "SpacyTextSplitter"
            else:
                text_splitter_module = importlib.import_module('langchain.text_splitter')
                TextSplitter = getattr(text_splitter_module, self.text_splitter_name)
                text_splitter = TextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=OVERLAP_SIZE)
        except Exception as e:
            print(e)
            text_splitter_module = importlib.import_module('langchain.text_splitter')
            TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
            text_splitter = TextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=OVERLAP_SIZE,
            )

        docs = loader.load_and_split(text_splitter)
        print(docs[0])
        if using_zh_title_enhance:
            docs = zh_title_enhance(docs)
        return docs
