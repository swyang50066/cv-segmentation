from setuptools import setup, find_packages

packages = find_packages(include=["model", "model.*"])
requirements = [
    "python>=3.6.0",
    "numpy>=1.18.0",
    "scipy>=1.1.0",
    "scikit-learn>=0.20.0",
    "scikit-build>=0.12.0",
    "opencv-python>=3.4.0",
    "PyMaxFlow>=1.2.0"
]

VERSION = {}  # type: ignore
with open("model/__version__.py", "r") as vfile:
    exec(vfile.read(), VERSION)

setup(
    name="segmentation",
    version=VERSION["version"],
    description="Algorithmic Methods of Model-Based Medical Image Segmentation Using Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/swyang50066/segmentation",
    author="Yang Seung-Won, StarPlanetDust",
    author_email="bigbang50066@gmail.com",
    license="MIT",
    packages=packages,     
    install_requires=requirements,
    python_requires=">=3.6.0",
    zip_safe=False,
)
