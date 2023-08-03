from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List
from configs.model_config import SENTENCE_SIZE, CHINESE_TEXT_SPLITTER_MERGE_SPLIT


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, sentence_size: int = SENTENCE_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    # 社区版的切分
    def split_text1(self, text: str) -> List[str]:  ##此处需要进一步优化逻辑
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)

        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        ls = [i for i in text.split("\n") if i]
        for ele in ls:
            if len(ele) > self.sentence_size:
                ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) > self.sentence_size:
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")
                        for ele_ele2 in ele2_ls:
                            if len(ele_ele2) > self.sentence_size:
                                ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
                                                                                                       ele2_id + 1:]
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
        return ls

    def merge_text(self, text_list: List[str]) -> List[str]:
        merged_texts_list = []
        merged_texts = ""
        for text in text_list:
            if len(merged_texts) + len(text) <= self.sentence_size:
                merged_texts += text
            else:
                merged_texts_list.append(merged_texts)
                merged_texts = text  # 创建新字符串，不用担心merged_texts_list内被修改
        if merged_texts:
            merged_texts_list.append(merged_texts)
        return merged_texts_list

    def split_text(self, text: str) -> List[str]:  # 此处需要进一步优化逻辑
        if self.pdf:
            text = re.sub(r'(\s*\n)+', r'\n', text)  # 多行空白行换成一行
            text = re.sub(r'\s+', r' ', text)  # ocr会将很多连贯内容分割成不同行，替换成只用空格隔开，使其分割优先级小于句号
            text = re.sub(r"\.{7,}", "", text)  # 多余6个省略号，可能是目录识别乱码

        # 单字符断句符，将断句符前后拆成两段。
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
        text = re.sub(r'([;；!?。！？]["’”」』]{0,2})([^;；!?。！？])', r'\1\n\2', text)  # 排除两个符号紧挨着的情况，如?!
        text = re.sub(r'(\.)( )', r"\1\n\2", text)  # .后面跟的是空格才是英文句号，才需要分开，否则可能是小数或者网址
        text = re.sub(r'(…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(r'(\s*\n)+', r'\n', text)  # 多行空白行换成一行
        text_list = [i for i in text.split("\n") if i]
        for t in text_list:
            if len(t) > self.sentence_size:   # 分开后仍然超长度
                t1_text = re.sub(r'([,，]["’”」』]{0,2})([^,，\d])', r'\1\n\2', t)   # 用逗号分，排除数字用,隔开的情况
                t1_list = t1_text.split("\n")
                for t1 in t1_list:
                    if len(t1) > self.sentence_size:
                        t2_text = re.sub(r'(\s+)(\S)', r'\1\n\2', t1)   # 用空格分
                        t2_list = t2_text.split("\n")

                        t2_list = self.merge_text(t2_list)
                        t1_id = t1_list.index(t1)
                        t1_list = t1_list[:t1_id] + [i for i in t2_list if i] + t1_list[t1_id + 1:]

                t1_list = self.merge_text(t1_list)
                id = text_list.index(t)  # 在list中找到t的位置
                text_list = text_list[:id] + [i for i in t1_list if i] + text_list[id + 1:]  # 删掉t，加入t1_list的各个元素
        if CHINESE_TEXT_SPLITTER_MERGE_SPLIT:
            text_list = self.merge_text(text_list)
        return text_list
