from setuptools import setup


setup(
    name="scikit-weka-examples",
    description="Examples for the scikit-weka library.",
    url="https://github.com/fracpete/scikit-weka-examples",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    license='GNU General Public License version 3.0 (GPLv3)',
    package_dir={
        '': 'src'
    },
    packages=[
        "scikitwekaexamples",
    ],
    version="0.0.1",
    author='Peter "fracpete" Reutemann',
    author_email='scikit-weka@fracpete.org',
    install_requires=[
        "scikit-weka>=0.0.1",
    ],
)
