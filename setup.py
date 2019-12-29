from setuptools import setup, find_packages

required = [
    "numpy"
]

tf_options = dict(
    gpu="tensorflow-gpu>=2.0.0",
    cpu="tensorflow>=2.0.0"
)

setup(
    name="text2vec",
    version="0.1",
    description="Building blocks for text vectorization and embedding",
    author="Dave Hollander",
    author_url="https://github.com/brainsqueeze",
    url="https://github.com/brainsqueeze/text2vec",
    license="BSD 2-Clause License",
    install_requires=required,
    extras_require=tf_options,
    packages=find_packages()
)
