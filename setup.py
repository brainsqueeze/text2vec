from setuptools import setup, find_packages


setup(
    name="text2vec",
    version="0.4.1",
    description="Building blocks for text vectorization and embedding",
    author="Dave Hollander",
    author_url="https://github.com/brainsqueeze",
    url="https://github.com/brainsqueeze/text2vec",
    license="BSD 2-Clause License",
    install_requires=[
        "numpy",
        "pyyaml"
    ],
    extras_require=dict(
        serving=[
            "flask",
            "flask-cors",
            "nltk",
            "tornado"
        ],
        gpu="tensorflow-gpu>=2.1.0",
        cpu="tensorflow>=2.1.0"
    ),
    packages=find_packages(exclude=["bin"]),
    entry_points={
        "console_scripts": [
            "text2vec_main=text2vec.bin.main:main",
        ],
    }
)
