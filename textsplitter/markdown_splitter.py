import re
from langchain.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Any, Optional, Iterable
from .markdown_header_splitter import MarkdownHeaderTextSplitter

from configs.model_config import *
from utils.file_util import get_filename_from_source

"""
对url和代码段进行替换的为占位符，url和代码段的内容写入metadata里，不会进入embedding
"""

md_headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]

md_url_placeholder = "URL_P"
md_code_placeholder = "CODE_P"


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
            _splits = re.split(f"({separator})", text, flags=re.DOTALL)
            splits = [_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text, flags=re.DOTALL)
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

    # todo 改成优先合并长度短的
    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

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
            if re.search(_s, text, flags=re.DOTALL):
                separator = _s
                new_separators = separators[i + 1:]  # 下几级的separators
                break
        if separator == self._separators[1]:  # 有代码
            splits = _split_text_by_code_blocks(text, self._separators[1])
        else:
            text = re.sub(r" {20,}", r"", text)  # 超过20个的空格就不太可能是缩进了（表格经常包含大量空格）；较少的空格则是markdown缩进，需要保留
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
            if self._length_function(s) < self._chunk_size or re.search(self._separators[1], s,
                                                                        flags=re.DOTALL):  # 长度满足要求或是代码块，不递归切分
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
            r"```[^`]*?```\s*",  # *?非贪心匹配，从```开始匹配到最近的```
            # Horizontal lines
            "\n\*\*\*+\n",
            "\n---+\n",
            "\n___+\n",
            # Note that this splitter doesn't handle horizontal lines defined
            # by *three or more* of ***, ---, or ___, but this is not handled
            "\n\n",
            "\n+",  # 因为保留了两个回车换行，此处匹配要防止对两个回车的前一个回车切分
            "。",
            "；",
            ";",
            "，",
            ",",
            "、",
            " ",
            rf"{{{md_url_placeholder}\d}}|{{{md_code_placeholder}\d}}|.",  # 匹配任何字符，在任意位置均可切分，除了url和code的占位符
        ]
        super().__init__(separators=separators, **kwargs)


def _md_title_enhance(docs: List[Document], filepath) -> List[Document]:
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


def _md_code_url_replace(page_content: str, url_dict: dict, code_dict: dict):
    # ()除了用于分组，还有匹配子表达式的功能，配合?:防止匹配子表达式http/ftp/file
    url_pattern = r"(?:https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"
    url_blocks = re.findall(url_pattern, page_content, re.DOTALL)
    for url_block1 in url_blocks:   # 文档可能有重复的url，或者一个url是另一个url的子链，去重
        for url_block2 in url_blocks:
            if url_block2 in url_block1 and url_block2 != url_block1:
                url_blocks.remove(url_block2)
    url_blocks = set(url_blocks)
    for i, url_block in enumerate(url_blocks):
        # print(i, url_block)
        page_content = page_content.replace(url_block, f'{{{md_url_placeholder}{i}}}')
        url_dict[i] = url_block

    code_pattern = r"```.*?````*"
    code_blocks = re.findall(code_pattern, page_content, re.DOTALL)
    code_blocks = set(code_blocks)  # 文档可能有重复的code，去重。一般不会出现代码块套代码块
    for i, code_block in enumerate(code_blocks):
        page_content = page_content.replace(code_block, f'{{{md_code_placeholder}{i}}}')
        code_dict[i] = code_block
    return page_content


def _md_write_code_url_in_metadata(docs: list[Document], url_dict: dict, code_dict: dict):
    for i, doc in enumerate(docs):
        docs[i].metadata["CODE_NUM"] = []
        docs[i].metadata["URL_NUM"] = []
        docs[i].metadata["CODE_LEN"] = 0    # todo 可根据code_len考虑是否召回
        for j, code in code_dict.items():
            code_placeholder = f"{{{md_code_placeholder}{j}}}"
            if code_placeholder in doc.page_content:
                docs[i].metadata["CODE_NUM"].append(j)
                docs[i].metadata["CODE_LEN"] += len(code)
                docs[i].metadata[code_placeholder] = code

        for j, url in url_dict.items():
            url_placeholder = f"{{{md_url_placeholder}{j}}}"
            if url_placeholder in doc.page_content:
                docs[i].metadata["URL_NUM"].append(j)
                docs[i].metadata[url_placeholder] = url


def my_md_split(filepath, sentence_size=SENTENCE_SIZE, sentence_overlap=SENTENCE_OVERLAP):
    # 获取文本
    loader = TextLoader(filepath)
    markdown_document: Document = loader.load()[0]
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
        text_splitter = MyMarkdownTextSplitter(chunk_size=sentence_size, chunk_overlap=sentence_overlap,
                                               keep_separator=True)
        docs = text_splitter.split_documents(docs)

        docs = _md_title_enhance(docs, filepath)
    else:
        # 按separators递归拆分
        text_splitter = MyMarkdownTextSplitter(chunk_size=sentence_size, chunk_overlap=sentence_overlap,
                                               keep_separator=True)
        docs = text_splitter.split_documents([markdown_document])

    if MD_REPLACE_CODE_AND_URL:
        # 根据占位符，将url和code写入metadata
        _md_write_code_url_in_metadata(docs, url_dict, code_dict)

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
