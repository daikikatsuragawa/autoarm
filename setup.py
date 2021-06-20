import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="autoarm",
    version="0.1.0",
    author="Daiki Katsuragawa",
    author_email="daikikatsuragawa@gmail.com",
    description="AutoARM simplifies and automates association rule mining and related tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daikikatsuragawa/autoarm",
    install_requires=["pandas", "mlxtend", "numpy", "df4loop"],
    packages=setuptools.find_packages(),
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
