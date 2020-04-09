import setuptools
setuptools.setup(
    name="EpiForecastStatMech",
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "pandas", "matplotlib", "seaborn", "tensorflow", "tensorflow_probability", "jax", "jaxlib", "glmnet_py"]
)
