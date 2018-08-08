import os
import subprocess
import sys

from setuptools import setup, find_packages


def __path(filename):
    return os.path.join(os.path.dirname(__file__),
                        filename)

def generate_proto():
    protoc_command = ['python', '-m', 'grpc_tools.protoc', '-I', 'protos', '--proto_path=service/protos/',
                      '--python_out=service/protos/gen-py',
                      '--grpc_python_out=service/protos/gen-py', 'service/protos/sentiments.proto']
    if subprocess.call(protoc_command) != 0:
        sys.exit(-1)


generate_proto()

if os.path.exists(__path('version.properties')):
    version = open(__path('version.properties')).read().strip()

setup(
    name='sentiment-service',
    version=version,
    author='Sujay Bhowmick',
    author_email='sujaybhowmick@gmail.com',
    description='Sentiment Analysis Service',
    license='',
    keywords='sentiment analysis',
    url='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    package_data={'sentiments.server': ['conf/*']},
    long_description='',
    classifiers=[
        'Programming Language :: Python :: 3.6'
        ],
    install_requires=[
        'grpcio==1.6.0',
        'futures==3.2',
        'pyyaml==3.12'
    ],
    include_package_data=True
)
