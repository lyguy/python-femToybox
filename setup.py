from distutils.core import setup

setup(
    name='femToybox',
    version='0.0.1',
    author='Lyman Gillispie',
    author_email="lyman.gillispie@gmail.com",
    packages=['femtoybox'],
    install_requires=[
        "numpy >= 1.7.0",
        "scipy >= 0.10"
    ]
)
