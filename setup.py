from distutils.core import setup

setup(
    name='keggler',
    version='0.1.0',
    author='M. Lisovyi',
    author_email='mlisovyiATgmail.com',
    packages=['keggler', 'keggler.ensemble', 'keggler.plotting', 'keggler.preprocess'],
    scripts=[],
    url='http://pypi.python.org/pypi/keggler/',
    license='LICENSE.txt',
    description='A personal kaggle toolkit',
    long_description=open('README.md').read(),
    python_requires='~=3.5',
    install_requires=[
        "numpy >= 1.14",
        "pandas >= 0.22.0",
        "scikit-learn >= 0.19.1",
        "lightgbm >= 2.1.0",
#        "python >= 3.5",
        "matplotlib >= 2.2.0"
    ],
)
