import os
from setuptools import setup

def get_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        return f.read()


def do_setup():
    setup(
        name="easybt",
        version="1.0",
        author="Kirill Fedorenko",
        author_email="емейл поставь!",
        url='https://github.com/Fsa55/easybt',
        description="Kafka CLI",
        long_description=get_readme(),

        packages=['easybt'],

        classifiers=[
            'Development Status :: 5 - Production/Stable',

            'Programming Language :: Python :: 3',
        ],
        install_requires=['plotly', 'pandas', 'numpy'],
        include_package_data=True
    )


if __name__ == "__main__":
    do_setup()
