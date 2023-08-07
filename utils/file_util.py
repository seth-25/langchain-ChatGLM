import os
def get_filename_from_source(file_source):
    """
    从文件路径名称中获取文件名
    """
    return os.path.split(file_source)[-1]
