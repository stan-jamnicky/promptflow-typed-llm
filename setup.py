from setuptools import find_packages, setup

PACKAGE_NAME = "promptflow_typed_llm"

setup(
    name=PACKAGE_NAME,
    version="0.0.9",
    description="Package for Promptflow tool that sends typed LLM requests",
    packages=find_packages(),
    entry_points={
        "package_tools": ["typed_llm = typed_llm.tools.utils:list_package_tools"],
    },
    include_package_data=True,   # This line tells setuptools to include files from MANIFEST.in
    install_requires=[
        'promptflow>=1.15.0',
        'promptflow[azure]>=1.15.0',
        'promptflow-tools>=1.4.0',
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    author="Tanya Borisova",
    author_email="tanyatik@yandex.ru",
    url="https://github.com/tanya-borisova/promptflow-typed-llm-tool",
)