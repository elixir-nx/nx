if [ -z ${MIX_APP_PATH} ]; then
  echo "Ensure MIX_APP_PATH is set correctly."
  exit 1
fi

PRIV_DIR=${MIX_APP_PATH}/priv

# Get the url for libtorch 1.9/CUDA 11.1 download
# macOS does not support CUDA through prebuilt lib

LIBTORCH_DOWNLOAD_URL=""
OS=$(uname -s)

case $OS in
  "Darwin")
      LIBTORCH_DOWNLOAD_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.9.1.zip"
      ;;
  "Linux")
      LIBTORCH_DOWNLOAD_URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.9.1.zip"
      ;;
  *)
    echo "Windows not supported yet"
    exit 1
    ;;
esac

if [ -z ${LIBTORCH_DIR} ]; then
  export LIBTORCH_DIR=${PRIV_DIR}/libtorch
fi

if [ -d "$LIBTORCH_DIR" ]; then
  echo "libtorch already installed at $LIBTORCH_DIR"
else
  echo "Downloading libtorch 1.9.1 for $OS"
  wget $LIBTORCH_DOWNLOAD_URL --no-check-certificate -O "${PRIV_DIR}/libtorch.zip"

  echo "Unpacking libtorch at ${PRIV_DIR}"
  cd ${PRIV_DIR} && unzip -qq libtorch.zip && cd -
fi

make ${PRIV_DIR}/torchx.so
echo "after torchx.so"