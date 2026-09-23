# README

## Initialisation d'un projet

```
# installer mise
mkdir tps-laval
cd tps-laval
git init
mise use uv@latest
uv --version
# figer la version uv dans mise.toml (0.2.17)
uv init
```

## Clone du projet

```
git clone https://github.com/jlguenego/tps-laval
cd tps-laval
# install bun et uv
mise install
bun install
uv sync
mise run tp1
```

