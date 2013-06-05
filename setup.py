from setuptools import setup, find_packages

setup(
    name='femToybox',
    version='0.1.2',
    author='Lyman Gillispie',
    author_email="lyman.gillispie@gmail.com",
    url="https://github.com/fmuzf/python-femToybox",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.7.0",
        "scipy >= 0.10"
    ]
)
