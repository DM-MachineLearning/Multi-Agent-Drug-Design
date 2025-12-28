from setuptools import setup, find_namespace_packages # Change this line

setup(
    name="multi_agent_drug_design",
    version="0.1",
    packages=find_namespace_packages(include=["Agents*", "Generators*", "Datasets*", "src*", "utils*"]),
)