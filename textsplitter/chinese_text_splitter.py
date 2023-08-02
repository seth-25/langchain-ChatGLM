from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List
from configs.model_config import SENTENCE_SIZE


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, sentence_size: int = SENTENCE_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    def split_text1(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list

    def split_text(self, text: str) -> List[str]:  # 此处需要进一步优化逻辑
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub(r"\s", " ", text)
            text = re.sub(r"\n\n", "", text)
            text = re.sub(r"\.{7,}", "", text)  # 多余6个省略号，可能是目录识别乱码

        # 单字符断句符，将断句符前后拆成两段。
        text = re.sub(r'([;；!?。！？]["’”」』]{0,2})([^;；!?，。！？])', r'\1\n\2', text)  # 排除两个符号紧挨着的情况，如?!
        text = re.sub(r'(\.)([^"”’\d.])', r"\1\n\2", text)  # 排除.后面跟的\d防止将小数分割开，排除.跟着.的英文省略号
        text = re.sub(r'(…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号

        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        ls = [i for i in text.split("\n") if i]
        for ele in ls:
            if len(ele) > self.sentence_size:   # 分开后仍然超长度
                ele1 = re.sub(r'([,，]["’”」』]{0,2})([^,，])', r'\1\n\2', ele)   # 用逗号分
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

                id = ls.index(ele)  # 在list中找到ele的位置
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]  # 删掉ele，加入ele1_ls的各个元素
        return ls
