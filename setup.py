from setuptools import setup, find_packages


setup(
    name="text2vec",
    version="1.0.6",
    description="Building blocks for text vectorization and embedding",
    author="Dave Hollander",
    author_url="https://github.com/brainsqueeze",
    url="https://github.com/brainsqueeze/text2vec",
    license="BSD 2-Clause License",
    install_requires=[
        "numpy",
        "pyyaml",
        "tokenizers",
        "datasets"
    ],
    extras_require=dict(
        serving=[
            "flask",
            "flask-cors",
            "nltk",
            "tornado"
        ]
    ),
    packages=find_packages(exclude=["bin"]),
    entry_points={
        "console_scripts": [
            "text2vec_main=text2vec.bin.main:main",
        ],
    }
)
