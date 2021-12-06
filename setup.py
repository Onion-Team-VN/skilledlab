import setuptools
import skilledlab

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='skilledlab',
    version=skilledlab.__version__,
    author="skilledin team",
    author_email="contact@skilledin.ai",
    description="Organize Machine Learning Experiments",
    long_description=long_description,
    long_description_content_type="",
    project_urls={
        'Documentation': ''
    },
    packages=setuptools.find_packages(exclude=('skilledlab_helpers',
                                               'skilledlab_helpers.*',
                                               'test',
                                               'test.*')),
    install_requires=['gitpython',
                      'pyyaml',
                      'numpy'],
    entry_points={
        'console_scripts': [''],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='machine learning',
)