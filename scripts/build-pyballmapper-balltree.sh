#!/usr/bin/env bash
set -euo pipefail

# build-pyballmapper-balltree
# Clones the integrate-balltree branch, renames the package to pyballmapper-balltree, and builds it.
# Output: /tmp/balltree-pkg/dist/pyballmapper_balltree-*.whl

echo "==> Cloning pyBallMapper (integrate-balltree branch) into /tmp/balltree-pkg"
rm -rf /tmp/balltree-pkg
git clone --branch integrate-balltree --single-branch \
    https://github.com/jooyounghahn/pyBallMapper.git /tmp/balltree-pkg

echo "==> Renaming package to pyballmapper-balltree"
sed -i '' 's/^name = "pyBallMapper"/name = "pyBallMapper-balltree"/' /tmp/balltree-pkg/pyproject.toml

echo "==> Building wheel"
cd /tmp/balltree-pkg && uv build

echo "==> Done! Wheel at:"
ls /tmp/balltree-pkg/dist/pyballmapper_balltree-*.whl
