# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

setuptools.setup(
    name = 'nni-tool',
    version = '999.0.0-developing',
    packages = setuptools.find_packages(exclude=['*test*']),

    python_requires = '>=3.5',
    install_requires = [
        'requests',
        'ruamel.yaml',
        'psutil',
        'astor',
        'schema',
        'PythonWebHDFS',
        'colorama'
    ],

    author = 'Pengcheng Laboratory AAH Team',
    author_email = 'renzhx@pcl.ac.cn',
    description = 'NNI control for Automated Machine Learning As An AI HPC Benchmark',
    license = 'MIT',
    url = 'https://github.com/pcl-ai-public/AAH',
    entry_points = {
        'console_scripts' : [
            'nnictl = nni_cmd.nnictl:parse_args'
        ]
    }
)
