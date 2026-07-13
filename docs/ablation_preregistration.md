# Pré-registro — Ablação de features (shear_850_200_mps + rh_mid)

**Registrado em 2026-07-13, ANTES de qualquer resultado de treino.** Este
documento fixa métrica, veredito e disciplina de leitura. Ele não muda depois
que os números existirem.

## Métrica (fixada)

ΔPR-AUC = PR-AUC(braço B: 9ch + shear_850_200_mps + rh_mid) − PR-AUC(braço A:
9ch atuais), média out-of-fold, com IC 95% via bootstrap por cluster de SID.
PR-AUC é threshold-independent — nenhum threshold será escolhido.

## Veredito (ler o IC, não o sinal do ponto central)

- **IC exclui zero, positivo** → shear/rh AGREGAM skill (incremento
  quantificado). Reportar o delta e o IC. Sem superdimensionar: "agregam X,
  IC [a, b]" — não "features cruciais".
- **IC inclui zero** → **NULL**: não distinguível de zero. Reportar como "os
  core predictors não adicionam skill detectável nesta resolução/regime" —
  reforça que o gargalo é dado, não features. NÃO reinterpretar como positivo
  fraco.
- **IC exclui zero, negativo** → investigar (não reportar cru; pode ser
  artefato).

## Disciplina (anti-racionalização)

- Ler o IC, uma vez. Não mover a régua depois de ver o número.
- Não garimpar fold/seed/threshold que favoreça um braço.
- Não re-rodar com parâmetros diferentes esperando resultado melhor. Uma
  rodada bem-dimensionada, um veredito, aceito.
- Reportar o número honesto com o IC, seja qual for — positivo esperado ou
  null informativo. Os dois são resultados válidos.

## Critério de sucesso do experimento

Veredito CLARO: IC apertado o suficiente para cair em um dos ramos acima sem
ambiguidade. O único desfecho a evitar é o inconclusivo por falta de poder —
por isso seeds ≥ 3 (captura variabilidade de inicialização).

## Parâmetros do run

**FIXADOS em 2026-07-13 ~17h30, antes de qualquer resultado de treino:**

- **k = 3 folds, 15 épocas por treino.**
- **Seeds = {42, 123, 456}**, execução FASEADA: uma seed por noite
  (42 em 2026-07-13; 123 e 456 nas noites seguintes). 6 células de treino
  por seed (3 folds × 2 braços); 18 no total.
- **Device: CPU** (CUDA indisponível nesta máquina). Custo medido na
  calibração de 2026-07-13: ~5,61 min/época → ~84,5 min/célula → ~8,5 h/seed.
- **ADT: fiel à produção** — canal ADT presente nos dois braços (modelo de
  10/12 canais de entrada), valores crus onde há cobertura (1.257 eventos),
  zeros onde não há; simétrico entre braços A e B, idêntico ao comportamento
  atual do modelo de produção (checkpoint com 10 canais, stats sem adt_mean).
- **Veredito**: ΔPR-AUC (B−A) médio entre as 3 seeds com IC 95% bootstrap
  por cluster de SID, computado via `--aggregate` sobre os 3
  `oof_predictions.csv` salvos. Lido UMA vez pelos 3 ramos acima.
  **Nenhum veredito antes das 3 seeds agregadas** — resultados por seed são
  intermediários e não serão interpretados isoladamente.

- Conjunto: dev set validado pelo census (14.101 eventos, 687 positivos,
  839 SIDs; cobertura PL completa, gate PASS de 2026-07-13).
- Folds: StratifiedGroupKFold agrupado por SID, idênticos entre braços.
- Stats de normalização: train-only por (seed, fold, braço), escopadas no dir
  do run; stats globais intocadas.
- Treino: src.training.trainer.train() real; test split jamais lido.
