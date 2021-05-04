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
        "sklearn",
        "tensorflow",
        "tensorflow_probability",
        "jax",
        "jaxlib",
        "flax",
        "dm-tree",
        "glmnet_py",
        "xarray",
    ],
)
