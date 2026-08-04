export UV_ENV_FILE=$(pwd)/.env
export PYTHONPATH=.

# mmcv has no prebuilt wheels and must compile; it needs torch present at build
# time and a setuptools that still ships pkg_resources (dropped in >=81).
pip install --quiet "setuptools<81" torch torchvision
pip install --quiet --no-build-isolation "mmcv>=2.0.0rc4,<2.2.0"
