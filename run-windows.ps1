# Use the Windows-native venv when the repo is shared with WSL (avoids .venv\lib64 access denied).
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"
& uv run @args
