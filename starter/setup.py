import setuptools

setuptools.setup(
    name="salary_prediction",
    version="0.1.0",
    packages= ['src', 'tests'],
    include_package_data=True,
    description="Modeling code to train/test salary prediction model.  Also includes API to make predictions.",
    author="Srikanth Kotha",
)
