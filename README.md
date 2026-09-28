# Engenharia de Atributos Físico-Informados para Predição de Ponto Final em Fornos Básicos a Oxigênio

Dissertação de Mestrado — Programa de Pós-Graduação em Informática (PPGI), UTFPR, câmpus Cornélio Procópio.

- **Autor:** Gabriel Vinicius Silva Vieira de Souza
- **Orientador:** Prof. Dr. Silvio Ricardo Rodrigues Sanches
- **Título em inglês:** *Physics-Informed Feature Engineering for Endpoint Prediction in Basic Oxygen Furnaces*

Este repositório contém o texto da dissertação (LaTeX, normas ABNT/UTFPR), o código que gera os dados
sintéticos e executa os experimentos, e os resultados usados nos Capítulos 4 e 5. Código e resultados
estão como na versão submetida.

## Leitura rápida

| Quero... | Onde |
|---|---|
| Ler a dissertação | [`modelo_dissertacao.pdf`](modelo_dissertacao.pdf) |
| Ver as tabelas de resultados em forma bruta | [`results/`](results/) (CSV) |
| Ver o gerador de dados e as equações físicas | [`scripts/generate_bof_data.py`](scripts/generate_bof_data.py), [`scripts/utils/physics.py`](scripts/utils/physics.py) |
| Ver os 8 atributos físico-derivados | `calc_physics_features()` em [`scripts/utils/physics.py`](scripts/utils/physics.py) |
| Reproduzir os experimentos | seção [Reprodução](#reprodução) abaixo |

## Estrutura

```
modelo_dissertacao.tex     documento principal (inclui Capitulos/capitulo1..6.tex)
modelo_dissertacao.pdf     versão compilada
Capitulos/                 capítulos 1–6
bibtex/reflatex.bib        referências (BibTeX)
styles/                    classe normas-utf-tex e estilos ABNT (template UTFPR, não editado)
Images/                    figuras; Images/results/ e Images/shap/ são geradas pelos scripts
aprovacao.pdf              folha de aprovação incluída no documento
scripts/                   pipeline experimental (Python)
  generate_bof_data.py       1. geração Monte Carlo de 10.000 corridas sintéticas (seed=42)
  feature_engineering.py     2. atributos físico-derivados + divisão 70/15/15
  train_models.py            3. treino e tuning (Optuna) das Configs A, B e C
  evaluate_compare.py        4. comparação A×C, Wilcoxon, extrapolação, figuras
  shap_analysis.py           5. SHAP e testes das hipóteses H1–H5
  analyze_circularity.py     6. informação mútua normalizada (circularidade)
  run_pipeline.sh            executa as etapas 1–6 em sequência
  utils/physics.py           constantes termodinâmicas e equações do modelo físico
  utils/metrics.py           MAE, RMSE, R², fração dentro da tolerância
data/                      dataset sintético gerado (bof_synthetic.csv) e matrizes de atributos
results/                   métricas, testes estatísticos, extrapolação, SHAP, NMI e logs do pipeline
context/bof_physics.md     faixas operacionais e equações de referência citadas nos comentários do código
```

## Compilar a dissertação

Requer uma distribuição TeX com `pdflatex` e `bibtex` (TeX Live completo recomendado).

```bash
make          # pdflatex + bibtex + pdflatex ×2, ordena a lista de siglas e abre o PDF
make clean    # remove arquivos auxiliares
```

O `Makefile` configura `TEXINPUTS`/`BIBINPUTS`/`BSTINPUTS` para encontrar os estilos em `styles/`.

## Reprodução

### Ambiente

Os resultados foram produzidos em Linux (WSL2), Python 3.13.5, com GPU NVIDIA (RTX 5060 Laptop).
As versões exatas das bibliotecas estão em [`requirements-lock.txt`](requirements-lock.txt)
(as versões mínimas originais estão em `scripts/requirements.txt`).

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
```

`train_models.py` usa aceleração por GPU no XGBoost (`device='cuda'`) e no CatBoost
(`task_type='GPU'`); sem GPU NVIDIA com CUDA, esses dois treinadores precisam ser trocados para CPU.

### Execução

Todos os comandos rodam a partir da raiz do repositório:

```bash
bash scripts/run_pipeline.sh        # etapas 1–6 (≈ 10 h 30 min no ambiente acima)
bash scripts/run_pipeline.sh 4      # retoma a partir da etapa 4 (exige models/ já treinados)
```

- **Etapas 1–2** (dados e atributos) levam segundos e são determinísticas (`seed=42`): regeneram
  `data/bof_synthetic.csv` idêntico ao versionado (10.521 amostras avaliadas, 521 rejeitadas).
- **Etapa 3** (treino de 72 modelos, 20 *trials* Optuna cada) domina o tempo total.
- **Etapas 4–6** leem os modelos de `models/`.

Os modelos treinados (`models/*.joblib`, 176 MB) não são versionados no git. Para rodar as etapas 4–6
sem retreinar, baixe o arquivo `models.zip` na página de *Releases* do repositório e extraia-o na raiz.

### Correspondência entre resultados e texto

| Arquivo | Usado em |
|---|---|
| `results/metrics.csv` | Tabelas de resultados das Configs A, B e C (Cap. 5) |
| `results/comparison_table.csv`, `results/statistical_tests.csv` | Comparação A×C e teste de Wilcoxon (Cap. 5) |
| `results/extrapolation_results.csv` | Teste de extrapolação (Cap. 5) |
| `results/mutual_info.csv` | Análise de circularidade (NMI) (Cap. 5) |
| `results/shap_importance.csv`, `results/shap_hypothesis_tests.txt` | Importância SHAP e hipóteses H1–H5 (Cap. 5) |

## Dados

Nenhum dado industrial foi utilizado. Todo o conjunto de dados é sintético, gerado por
`scripts/generate_bof_data.py` a partir de faixas operacionais e constantes físicas extraídas da
literatura citada na dissertação.
