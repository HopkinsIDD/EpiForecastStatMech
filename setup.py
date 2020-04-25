import setuptools

setuptools.setup(
    name="EpiForecastStatMech",
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "absl-py",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tensorflow",
        "tensorflow_probability",
        "jax",
        "jaxlib",
        "flax",
        "glmnet_py",
        "xarray",
    ],
)
