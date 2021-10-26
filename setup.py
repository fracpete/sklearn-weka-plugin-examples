from setuptools import setup


setup(
    name="sklearn-weka-plugin-examples",
    description="Examples for the sklearn-weka-plugin library.",
    url="https://github.com/fracpete/sklearn-weka-plugin-examples",
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
        "sklwekaexamples",
    ],
    version="0.0.2",
    author='Peter "fracpete" Reutemann',
    author_email='sklweka@fracpete.org',
    install_requires=[
        "sklweka>=0.0.3",
    ],
)
