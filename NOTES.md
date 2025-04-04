# Migration of poetry to uv.
uvx migrate-to-uv

# Build cna-mixture-rs
uv tool run maturin develop

# Use as a virtual environment
uv venv

source .venv/bin/activate

uv pip install -r pyproject.toml

deactivate