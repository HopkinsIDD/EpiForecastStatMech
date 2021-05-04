import setuptools

setuptools.setup(
    name="EpiForecastStatMech",
    version="0.2",
    packages=setuptools.find_packages(),
    install_requires=[
        "absl-py",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "sklearn",
        # "tensorflow",  # revert back to these when tfp 0.13 is released.
        # "tensorflow_probability",
        "tf-nightly",
        "tfp-nightly",
        "jax",
        "jaxlib",
        "flax",
        "dm-tree",
        "glmnet_py",
        "xarray",
    ],
)
