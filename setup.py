"""Setup file for the soft-information repository."""

import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="soft_information",
    description="Repository for a quantum applications project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.11",
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
)
