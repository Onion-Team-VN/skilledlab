import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='skilledlab',
    version='0.0.1',
    author="skilledin.ai",
    author_email="contact@skilledin.ai",
    description="🧑‍🏫 Implementations/tutorials of deep learning papers with side-by-side notes 📝",
    long_description=long_description,
    long_description_content_type="",
    # project_urls={''
    # },
    packages=setuptools.find_packages(),
    install_requires=['labml>=0.4.133',
                      'labml-helpers>=0.4.84',
                      'torch',
                      'torchtext',
                      'torchvision',
                      'einops',
                      'numpy'],
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
