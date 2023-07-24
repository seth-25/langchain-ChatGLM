import re
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.text_splitter import MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain.docstore.document import Document

from typing import (
    Dict,
    List,
    Tuple,
    TypedDict,
)

from langchain.docstore.document import Document

md_headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
]


def md_title_enhance(docs: List[Document]) -> List[Document]:
    if len(docs) > 0:
        for doc in docs:
            title_list = []
            for key, value in doc.metadata.items():
                if re.match("Header", key):
                    # print(key, value)
                    title_list.append(value)
            doc.page_content = f"下文与({','.join(title_list)})有关。{doc.page_content}"
        return docs
    else:
        print("文件不存在")


def my_md_split(filepath):
    # 获取文本
    loader = TextLoader(filepath)
    markdown_document: Document = loader.load()[0]

    # 按标题拆分，并将标题写入metadata
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers, return_each_line=False)
    md_header_splits = header_splitter.split_text(markdown_document.page_content)

    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(md_header_splits)

    docs = md_title_enhance(docs)
    return docs
