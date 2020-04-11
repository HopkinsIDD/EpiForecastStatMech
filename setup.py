import setuptools
# Note: tensorflow==2.2.0rc1 is currently recommended (20200410)
setuptools.setup(
    name="EpiForecastStatMech",
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "pandas", "matplotlib", "seaborn", "tensorflow", "tensorflow_probability", "jax", "jaxlib", "glmnet_py"]
)
