import os


def get_filename_from_source(file_source) -> str:
    """
    从文件路径名称中获取文件名
    """
    return os.path.split(file_source)[-1]


def get_filename_no_suffix_from_source(file_source) -> str:
    """
     从文件路径名称中获取文件名，无后缀
     """
    filename = os.path.split(file_source)[-1]
    return filename.split('.')[0]


if __name__ == "__main__":
    print(get_filename_no_suffix_from_source("home/admin/test.md"))