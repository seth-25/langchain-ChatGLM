from configs.model_config import *
from langchain.docstore.document import Document
from text_splitter.markdown_splitter import md_url_placeholder, md_code_placeholder


import re

def get_filename_from_source(file_source) -> str:
    """
    从文件路径名称中获取文件名
    """
    return os.path.split(file_source)[-1]

def match_brackets_at_start(text):
    pattern = r'^【.*?】'  # 正则表达式匹配以【】开头的部分
    match = re.match(pattern, text)
    if match:
        return match.group(0)  # 返回匹配到的部分
    else:
        return None


def remove_brackets_at_start(text):
    pattern = r'^【.*?】'
    result = re.sub(pattern, '', text, count=1)  # 移除匹配的部分
    return result


def add_enter_after_brackets(text, markdown=False):
    """在标题后加回车，方便在webui显示"""
    pattern = r'^【.*?】'
    if markdown:
        replacement = r'\g<0>\n\n'
    else:
        replacement = r'\g<0>\n'
    return re.sub(pattern, replacement, text, flags=re.MULTILINE)


def merge_ids(id_set):
    """ 连续的id分在一起，成为一个id seq """
    id_list = sorted(list(id_set))

    id_seqs = []  # 存一个个连续的id seq
    id_seq = [id_list[0]]
    for i in range(1, len(id_list)):
        if id_list[i - 1] + 1 == id_list[i]:
            id_seq.append(id_list[i])
        else:
            id_seqs.append(id_seq)
            id_seq = [id_list[i]]
    id_seqs.append(id_seq)
    return id_seqs



def md_write_code_url_in_content(result) -> str:
    if "CODE_NUM" not in result.metadata.keys() or "URL_NUM" not in result.metadata.keys():  # 兼容之前没对url和code处理的向量
        return result.document
    code_num_list = result.metadata["CODE_NUM"]
    result_document = result.document
    for num in code_num_list:
        result_document = result_document.replace(f'{{{md_code_placeholder}{num}}}', result.metadata[f'{{{md_code_placeholder}{num}}}'])

    url_num_list = result.metadata["URL_NUM"]
    for num in url_num_list:
        result_document = result_document.replace(f'{{{md_url_placeholder}{num}}}', result.metadata[f'{{{md_url_placeholder}{num}}}'])
    return result_document


def generate_doc_with_score(id_seqs, id_map) -> list:
    """ 将一个连续的id seq拼成一个doc，并合并相邻的doc，做一些处理 """
    documents_with_scores = []
    for id_seq in id_seqs:
        doc: Document = None
        doc_score = None
        for id in id_seq:
            if id == id_seq[0]:
                result = id_map[id]
                result_document = result.document
                if MD_REPLACE_CODE_AND_URL:
                    result_document = md_write_code_url_in_content(result)
                doc = Document(
                    page_content=result_document,
                    metadata=result.metadata,
                )
                # metadata["content"]是在前端显示的内容，在标题后加回车，方便在webui显示
                if result.filename.lower().endswith(".md"):
                    doc.metadata["content"] = add_enter_after_brackets(doc.page_content, markdown=True)
                else:
                    doc.metadata["content"] = add_enter_after_brackets(doc.page_content, markdown=False)
                doc_score = result.distance
                if result.url:
                    doc.metadata["url"] = result.url
            else:
                result = id_map[id]
                result_document = result.document
                if MD_REPLACE_CODE_AND_URL:
                    result_document = md_write_code_url_in_content(result)
                remove_brackets_page_content = result_document

                last_res = id_map[id - 1]  # 上一个文本

                # 开启标题增强的情况下，如果当前文本和上一个文本标题相同，去掉当前文本的标题。
                if match_brackets_at_start(last_res.document) == match_brackets_at_start(result_document):
                    remove_brackets_page_content = remove_brackets_at_start(result_document)

                if result.filename.lower().endswith(".md"):  # 是markdown，文本切分自带换行，添加的上下文不需要换行
                    if REMOVE_TITLE:  # 有时候大模型会把标题也混入答案中。可选择去掉相同标题，但文本太长回答可能不全，模型无法理解哪些与标题有关
                        doc.page_content += remove_brackets_page_content
                    else:
                        doc.page_content += result_document
                    doc.metadata["content"] += add_enter_after_brackets(remove_brackets_page_content, markdown=True)
                else:  # 不是markdown，文本切分会去掉换行空格，添加上下文需要换行
                    if REMOVE_TITLE:  # 有时候大模型会把标题也混入答案中。可选择去掉相同标题，但文本太长回答可能不全，模型无法理解哪些与标题有关
                        doc.page_content += "\n" + remove_brackets_page_content
                    else:
                        doc.page_content += "\n" + result_document
                    doc.metadata["content"] += "\n" + add_enter_after_brackets(remove_brackets_page_content,
                                                                               markdown=False)

                doc_score = min(doc_score, result.distance)
        if not isinstance(doc, Document) or doc_score is None:
            raise ValueError(f"Could not find document, got {doc}")

        documents_with_scores.append((doc, doc_score))

        if SORT_BY_DISTANCE:
            documents_with_scores = sorted(documents_with_scores,
                                           key=lambda documents_with_scores: documents_with_scores[1])
    return documents_with_scores
