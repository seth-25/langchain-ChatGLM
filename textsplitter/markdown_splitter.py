import re
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Any, Optional, Iterable
from .markdown_header_splitter import MarkdownHeaderTextSplitter

from configs.model_config import *
from utils.file_util import get_filename_from_source

md_headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]


def _split_text_by_code_blocks(text, code_pattern):
    # 找到所有代码块的位置
    code_blocks = [match.span() for match in re.finditer(code_pattern, text, flags=re.DOTALL)]

    # 对文本进行切分，根据代码块位置将文本分成块
    text_lists = []
    start_index = 0
    for code_start, code_end in code_blocks:
        text_lists.append(text[start_index:code_start])  # 代码块前的文本
        text_lists.append(text[code_start:code_end])  # 代码块
        start_index = code_end
    if start_index < len(text):
        text_lists.append(text[start_index:])  # 最后一个代码块后的文本
    return text_lists


def _split_text_with_regex(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class MyRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,  # 需要在文本中保留切分符号，到时候方便拼接起来显示
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators


    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        # text = text.strip()   # 不去除换行，保留markdown格式
        if text == "":
            return None
        else:
            return text

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if re.search(_s, text):
                separator = _s
                new_separators = separators[i + 1:]  # 下几级的separators
                break
        if separator == self._separators[1]:  # 有代码
            splits = _split_text_by_code_blocks(text, self._separators[1])
        else:
            # text = re.sub(r" {3,}", r" ", text)  # 超过2个的空格替换成一个（表格经常包含大量空格）
            text = re.sub(r" {20,}", r"", text)  # 超过20个的空格就不太可能是缩进了（表格经常包含大量空格），较少的空格则保留markdown缩进
            text = re.sub(r"-{4,}", r"---", text)  # 超过3个的-替换成一个（表格经常包含大量-）
            splits = _split_text_with_regex(text, separator, self._keep_separator)
        # print([text])
        # print("separator:", [separator])
        # print(splits)
        # print("==================")
        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator

        for s in splits:
            if self._length_function(s) < self._chunk_size or re.search(self._separators[1], s):  # 长度满足要求或是代码块，不递归切分
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)  # 切分完还是不够小，递归切分
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)

        return final_chunks


class MyMarkdownTextSplitter(MyRecursiveCharacterTextSplitter):
    """Attempts to split the text along Markdown-formatted headings."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a MarkdownTextSplitter."""
        separators = [
            # First, try to split along Markdown headings (starting with level 2)
            "\n#{1,6} ",
            # Note the alternative syntax for headings (below) is not handled here
            # Heading level 2
            # ---------------
            # Code block
            r'```[^`]*?```\s*',  # .*?非贪心匹配，从```开始匹配到最近的```
            # Horizontal lines
            "\n\*\*\*+\n",
            "\n---+\n",
            "\n___+\n",
            # Note that this splitter doesn't handle horizontal lines defined
            # by *three or more* of ***, ---, or ___, but this is not handled
            "\n\n",
            "\n[^\n]",
            "。",
            "；",
            ";",
            "，",
            ",",
            "、",
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)


def md_title_enhance(docs: List[Document], filepath) -> List[Document]:
    if len(docs) > 0:
        for doc in docs:
            title_list = []
            for key, value in doc.metadata.items():
                if re.match("Header", key):
                    # print(key, value)
                    title_list.append(value)
            if MD_TITLE_ENHANCE_ADD_FILENAME:
                filename = get_filename_from_source(filepath)
                doc.page_content = f"【下文与({filename},{','.join(title_list)})有关】{doc.page_content}"
            else:
                doc.page_content = f"【下文与({','.join(title_list)})有关】{doc.page_content}"
        return docs
    else:
        print("文件不存在")


def my_md_split(filepath, sentence_size=SENTENCE_SIZE, sentence_overlap=SENTENCE_OVERLAP):
    # 获取文本
    loader = TextLoader(filepath)
    markdown_document: Document = loader.load()[0]

    # 每段文本前加如标题
    if MD_TITLE_ENHANCE:
        # 按标题拆分，并将标题写入metadata
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers, return_each_line=False)
        md_header_splits = header_splitter.split_text(markdown_document.page_content)

        # 按separators递归拆分
        text_splitter = MyMarkdownTextSplitter(chunk_size=sentence_size, chunk_overlap=sentence_overlap, keep_separator=True)
        docs = text_splitter.split_documents(md_header_splits)

        docs = md_title_enhance(docs, filepath)
    else:
        # 按separators递归拆分
        text_splitter = MyMarkdownTextSplitter(chunk_size=sentence_size, chunk_overlap=sentence_overlap, keep_separator=True)
        docs = text_splitter.split_documents([markdown_document])

    for doc in docs:
        doc.metadata["source"] = filepath
    return docs


if __name__ == "__main__":
    filepath = "../docs/CHANGELOG.md"
    docs = my_md_split(filepath)
    print("doc ===========================")
    for doc in docs:
        print(doc)
    print("doc ===========================")

    for doc in docs:
        if len(doc.page_content) < 20:
            print(doc)
