# Caderno de hipóteses (hypothesis registry)

Agenda de pesquisa viva do CycloneNet. Toda hipótese do projeto — testada ou
não — vive aqui, no espírito do pré-registro.

**Regras do caderno:**
- Registrar a hipótese E o teste ANTES de rodar (pré-registro).
- Registrar o veredito HONESTO depois, seja positivo, null ou refutado.
- Uma hipótese sem veredito = não-testada (não é "provável", é
  "desconhecida").
- Não remover hipóteses refutadas — elas são conhecimento (sabemos que são
  falsas). Manter o histórico.
- Datar tudo. Versionar em git (cada mudança é um commit rastreável).
- Hipótese sem motivação física clara: registrar essa ausência — é um sinal
  de fragilidade da hipótese.
- Entradas anteriores à criação deste caderno (2026-07-14) estão marcadas
  como **registro retroativo** — o teste veio antes do registro aqui.
  A partir de agora, a ordem é: registrar → testar.

Numeração própria deste caderno (HR-nn). Atenção: `analysis/ri_precursors.py`
usa internamente "H1–H4" — a correspondência está nas notas de HR-05/HR-06.

---

### HR-01: O FuelMap localiza a fonte de energia física real da tempestade
- **Registrada em:** 2026-07-14 (registro retroativo; testes de jun–jul/2026)
- **Pergunta:** a região destacada pelo FuelMap corresponde à fonte de
  energia oceânica real (reservatório de calor) que sustenta a
  intensificação, superando o baseline trivial do centro da tempestade?
- **Motivação física:** o combustível do ciclone é o fluxo de entalpia
  ar-mar, alimentado pelo calor oceânico subsuperficial (TCHP) — literatura
  operacional consolidada (AOML/SHIPS-RII).
- **Teste desenhado:** comparação de 3 vias (centro / física-pura /
  FuelMap) vs pico de TCHP; teste dinâmico; controle de prior.
- **Status:** TESTADA
- **Veredito:** **REFUTADA** (3 ângulos convergentes): validação TCHP
  (n=226) sem diferença significativa vs baseline do centro (p=0,30);
  teste dinâmico negativo; controle de prior mostrou que o colapso espacial
  era aritmética do prior, não estrutura aprendida.
- **Veredito em:** 2026-07 (consolidado)
- **Notas:** ressalva anterior já registrada no roadmap: a ablação causal
  (`src/evaluation/causal_ablation.py`) prova apenas dependência causal
  INTERNA do modelo ao FuelMap, não correspondência com a fonte física.
  Consequência inviolável: FuelMap = "hypothesis maps", nunca fonte de
  energia comprovada.

---

### HR-02: ADT como canal de entrada melhora a previsão de RI
- **Registrada em:** 2026-07-14 (registro retroativo; teste de jul/2026)
- **Pergunta:** adicionar ADT (proxy de superfície do reservatório
  subsuperficial; rho=0,30 vs TCHP, replicado em 2022/2023) como canal de
  entrada melhora o skill de RI?
- **Motivação física:** TCHP é preditor operacional de RI (SHIPS-RII); ADT
  é sua assinatura altimétrica disponível em área ampla.
- **Teste desenhado:** ablação with/without ADT, mesmo seed e protocolo.
- **Status:** TESTADA
- **Veredito:** **NULL/marginal** — exp_adt: with 0,906 vs without 0,914
  (não ajudou; direção levemente negativa). Ressalva: cobertura de ADT
  parcial (canal neutro fora de 2022–2023) e positivos escassos no
  subconjunto coberto → teste subdimensionado; NULL aqui não é refutação
  definitiva do sinal físico (o proxy ADT↔TCHP segue válido).
- **Veredito em:** 2026-07
- **Notas:** re-testável com cobertura ampla de SLA/ADT multi-ano.

---

### HR-03: O gargalo de desempenho é a amostra, não a arquitetura
- **Registrada em:** 2026-07-14 (registro retroativo; teste de jul/2026)
- **Pergunta:** o PR-AUC baixo vem do n de positivos (não da capacidade do
  modelo)?
- **Motivação física/estatística:** ~35 positivos originais tornam qualquer
  AUC dominado por ruído; física não entra — é poder estatístico.
- **Teste desenhado:** intervenção direta — expandir o dataset (1980–2023)
  e observar a resposta do skill mantendo a arquitetura.
- **Status:** TESTADA
- **Veredito:** **POSITIVO** — expansão de 9→115 positivos no test
  (802 no total) confirmou que o gargalo era amostra.
- **Veredito em:** 2026-07-12 (estágio 2 do dataset completo)
- **Notas:** base para o dataset atual (16.780 eventos / 802 positivos).

---

### HR-04: Shear e RH-mid agregam skill preditivo sobre o baseline de canais
- **Registrada em:** 2026-07-13 (pré-registro REAL, antes de qualquer
  resultado: `docs/ablation_preregistration.md`, commit `eaa8ae8`)
- **Pergunta:** adicionar shear_850_200_mps e rh_mid aos 9 canais atuais
  melhora PR-AUC de RI?
- **Motivação física:** shear profundo suprime RI; umidade média sustenta a
  convecção — preditores clássicos (SHIPS).
- **Teste desenhado:** ablação A (9 canais) vs B (+shear/RH), k=3 folds,
  15 épocas, seeds {42,123,456} faseadas; veredito ÚNICO via ΔPR-AUC médio
  entre seeds com IC 95% bootstrap por cluster de SID (`--aggregate`),
  pelos 3 ramos pré-registrados. IC lido UMA vez.
- **Status:** TESTANDO
- **Veredito:** — (seed 42 completa em 2026-07-14, Δ OOF +0,033
  INTERMEDIÁRIO sem valor de veredito; seeds 123/456 pendentes)
- **Veredito em:** —
- **Notas:** progresso operacional em `docs/ablation_progress.md`.

---

### HR-05: Shear mais baixo PRECEDE o onset de RI (precursor, t-24h)
- **Registrada em:** 2026-07-12 (pré-registrado em
  `analysis/ri_precursors.py` como "H2"; re-teste pós-backfill pré-declarado
  com pares congelados)
- **Pergunta:** no pareamento onset-vs-controle (mesma banda de
  intensidade), o nível de shear a t-24h é menor nos onsets de RI?
- **Motivação física:** shear profundo desorganiza o núcleo quente e
  suprime RI.
- **Teste desenhado:** pares casados congelados, null de permutação
  sign-flip, primária = nível a t-24h, Bonferroni ×4 (família fixa H1–H4
  do script).
- **Status:** TESTADA
- **Veredito:** **POSITIVO** — n=394/394 pares (100% cobertura PL),
  Δ pareado = −1,03 m/s, Cliff's δ=−0,13, p(Bonf ×4)=1,2e-3, direção
  física correta. Efeito pequeno porém robusto: precursor consistente, não
  preditor forte.
- **Veredito em:** 2026-07-13 (commit `970a419`)
- **Notas:** versão pré-backfill era n=5 (não testável) — o veredito válido
  é o pós-backfill sobre os MESMOS pares congelados (zero re-matching).

---

### HR-06: Umidade média mais alta PRECEDE o onset de RI (precursor, t-24h)
- **Registrada em:** 2026-07-12 (pré-registrado em
  `analysis/ri_precursors.py` como "H4")
- **Pergunta:** idem HR-05, para rh_mid a t-24h (maior nos onsets?).
- **Motivação física:** ar seco em níveis médios suprime a convecção
  profunda necessária à RI.
- **Teste desenhado:** idem HR-05 (mesma família Bonferroni ×4).
- **Status:** TESTADA
- **Veredito:** **POSITIVO** — n=394/394, Δ pareado = +2,59% RH,
  Cliff's δ=+0,12, p(Bonf ×4)=8,8e-3, direção física correta. Mesma
  leitura de HR-05: pequeno e robusto.
- **Veredito em:** 2026-07-13 (commit `970a419`)
- **Notas:** as outras duas primárias da família (queda de pressão "H1",
  p=4,0e-4; SST "H3", δ=+0,53, p=4,0e-4) também POSITIVAS no mesmo teste —
  registradas aqui como contexto da família, sem entrada própria.

---

### HR-07: Existe anomalia espacial de RI não explicada pelas condições conhecidas ("anomalia C")
- **Registrada em:** 2026-07-14
- **Pergunta:** existe região onde RI ocorre mais do que SST/TCHP/shear/RH
  explicam (resíduo espacial positivo após descontar as condições
  conhecidas)?
- **Motivação física:** (rascunho, a confirmar pelo autor) candidatos a
  driver não-medido: calor subsuperficial não capturado pela superfície
  (eddies), dinâmica de bacia (ondas, MJO), ou artefato de amostragem do
  IBTrACS. Se nenhum candidato sobreviver, a "anomalia" é provavelmente
  variância não modelada — a fragilidade está registrada.
- **Teste desenhado:** a desenhar — esboço pré-declarado: taxa de RI por
  célula espacial (não contagem bruta), resíduo vs modelo de condições,
  null espacial por permutação. Desenhar ANTES de olhar qualquer mapa.
- **Status:** NÃO-TESTADA
- **Veredito:** —
- **Veredito em:** —
- **Notas:** risco alto de garimpo espacial; o null e a correção de
  múltiplas comparações precisam estar fixados antes do primeiro plot.

---

_Para adicionar uma nova hipótese: copiar o bloco-modelo acima, numerar
HR-nn sequencial, preencher Pergunta/Motivação/Teste ANTES de rodar
qualquer análise, e commitar o registro antes do resultado._
