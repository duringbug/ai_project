<!--
 * @Description: 
 * @Author: 唐健峰
 * @Date: 2023-09-28 21:53:51
 * @LastEditors: ${author}
 * @LastEditTime: 2023-09-28 22:16:29
-->
## 预处理
- 1. 创建虚拟环境：使用命令创建一个虚拟环境。
```python
python -m venv venv
```
- 2. 激活虚拟环境：激活虚拟环境以开始在该环境中工作。在 macOS 和 Linux 上，激活命令是：
```python
source venv/bin/activate
```
- 3. 在虚拟环境中安装依赖：使用 pip 在虚拟环境中安装项目所需的依赖包。
```python
pip install package_name
```
- 4. 在虚拟环境中运行项目：在虚拟环境中执行你的 Python 项目，它将使用虚拟环境中的 Python 解释器和依赖库。
- 5. 退出虚拟环境：在项目工作完成后，可以使用以下命令退出虚拟环境：
```python
deactivate
```
- 6. 要将已安装的依赖项添加到 requirements.txt 文件中
```python
pip freeze > requirements.txt
```