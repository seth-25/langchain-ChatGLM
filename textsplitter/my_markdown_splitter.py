import re
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import Dict, List, Tuple, TypedDict, Any, Optional

from configs.model_config import *

md_headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]

def _split_text_with_regex(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            # print("_splits", _splits)
            splits = [_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # if len(_splits) % 2 == 0:
            #     splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]

class MyRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]

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
                new_separators = separators[i + 1:]
                break
        # print()
        # print("separator", [separator], "new_separators", [new_separators])

        splits = _split_text_with_regex(text, separator, self._keep_separator)

        # print("splits", splits)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)

        # print("final_chunks", final_chunks)
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
            # End of code block
            "```\n",
            # Horizontal lines
            "\n\*\*\*+\n",
            "\n---+\n",
            "\n___+\n",
            # Note that this splitter doesn't handle horizontal lines defined
            # by *three or more* of ***, ---, or ___, but this is not handled
            "\n\n",
            "\n",
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


def md_title_enhance(docs: List[Document]) -> List[Document]:
    if len(docs) > 0:
        for doc in docs:
            title_list = []
            for key, value in doc.metadata.items():
                if re.match("Header", key):
                    # print(key, value)
                    title_list.append(value)
            doc.page_content = f"【下文与({','.join(title_list)})有关】{doc.page_content}"

        return docs
    else:
        print("文件不存在")


def my_md_split(filepath, sentence_size=SENTENCE_SIZE, sentence_overlap=SENTENCE_OVERLAP):
    # 获取文本
    loader = TextLoader(filepath)
    markdown_document: Document = loader.load()[0]

    # 按标题拆分，并将标题写入metadata
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_headers, return_each_line=False)
    md_header_splits = header_splitter.split_text(markdown_document.page_content)

    text_splitter = MyMarkdownTextSplitter(chunk_size=sentence_size, chunk_overlap=sentence_overlap)
    docs = text_splitter.split_documents(md_header_splits)

    if MD_TITLE_ENHANCE:
        docs = md_title_enhance(docs)
    for doc in docs:
        doc.metadata["source"] = filepath
    return docs


if __name__ == "__main__":
    filepath = "../docs/Serverless快速入门.md"
    # filepath = "../docs/CHANGELOG.md"
    # filepath = "../docs/test.md"
    docs = my_md_split(filepath)
    print("doc ===========================")
    for doc in docs:
        print(doc)
    print("doc ===========================")

    for doc in docs:
        if len(doc.page_content) < 20:
            print(doc)