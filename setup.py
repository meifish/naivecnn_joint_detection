from setuptools import setup, find_packages

setup(
    name = 'naive-joints',
    version = 0.0,
    description = "using Keras to implement human joint detection",
    url="https://github.com/meifish/naivecnn-joint-detection",
    packages = find_packages(),
    install_requires = [
        'numpy'
    ]
)