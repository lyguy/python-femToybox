from setuptools import setup, find_packages

setup(
    name='femToybox',
    version='0.1.3-0.0.3',
    author='Lyman Gillispie',
    author_email="lyman.gillispie@gmail.com",
    url="https://github.com/fmuzf/python-femToybox",
    packages=find_packages(),
    license='MIT',
    install_requires=[
        "numpy >= 1.7.0",
        "scipy >= 0.10"
    ],
    test_suite='nose.collector',
    tests_require=['nose']
)
