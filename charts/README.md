# Graficos reais do projeto Jane Street

Este diretorio contem o codigo e os artefatos de visualizacao gerados a partir de:

- `data/raw/jane-street-real-time-market-data-forecasting/train.parquet`
- CSVs/JSONs reais em `reports/`

Nao ha dados mock nos graficos. Algumas visualizacoes usam amostras deterministicas ou agregados cacheados apenas para manter renderizacao e legibilidade viaveis; as amostras ainda sao linhas reais do parquet local.

## Gerar tudo

```bash
uv run python graficos/gerar_graficos.py
```

Saidas:

- `graficos/figuras/`: PNGs estaticos.
- `graficos/animacoes/`: GIFs gerados com `matplotlib.animation.FuncAnimation`.
- `graficos/dados_intermediarios/`: caches derivados dos dados reais.
- `graficos/manifest.json`: fontes, parametros e arquivos gerados.

Use `--refresh-cache` para forcar recomputacao dos agregados reais.

