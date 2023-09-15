def readPreprocessing(path):
    with open(path, 'r') as file:
        # 读取文件内容
        file_contents = file.read()
    return file_contents
