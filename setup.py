from setuptools import setup, find_packages


setup(
    name="text2vec",
    version="2.0.1",
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
            "tornado"
        ]
    ),
    packages=find_packages(exclude=["bin"]),
)
