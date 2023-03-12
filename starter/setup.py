import setuptools

setuptools.setup(
    name="salary_prediction",
    version="0.1.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    description="Modeling code to train/test salary prediction model.  Also includes API to make predictions.",
    author="Srikanth Kotha",
)
