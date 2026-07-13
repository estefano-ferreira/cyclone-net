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

k (folds), seeds e épocas serão fixados pelo autor com base na calibração de
tempo real de treino desta máquina (medição de 2026-07-13), ANTES do run —
e registrados aqui no momento da autorização, antes do primeiro resultado.

- Conjunto: dev set validado pelo census (14.101 eventos, 687 positivos,
  839 SIDs; cobertura PL completa, gate PASS de 2026-07-13).
- Folds: StratifiedGroupKFold agrupado por SID, idênticos entre braços.
- Stats de normalização: train-only por (seed, fold, braço), escopadas no dir
  do run; stats globais intocadas.
- Treino: src.training.trainer.train() real; test split jamais lido.
