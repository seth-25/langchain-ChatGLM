import re


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
