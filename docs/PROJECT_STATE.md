# PROJECT_STATE — estado e retomada

**LEIA ESTE ARQUIVO PRIMEIRO ao iniciar uma sessão — ele diz onde paramos e
o próximo passo.**

Regras de manutenção deste arquivo:
- ATUALIZAR ao fim de cada sessão/marco: mover o "próximo passo" para
  frente, atualizar o estado do experimento, mover pendências concluídas
  para "marcos".
- Um arquivo de estado DESATUALIZADO engana — se algo mudou e não foi
  refletido aqui, corrigir antes de confiar.
- No INÍCIO de cada sessão: LER este arquivo primeiro para se localizar.

_Última atualização: 2026-07-14 ~08h15._

## 1. RETOMADA IMEDIATA (o que fazer AGORA)

Rodar a **seed 123** da ablação (noite 2):

```
./venv/Scripts/python.exe analysis/feature_ablation_cnn.py --folds 3 --epochs 15 --seeds 123 --execute
```

- Método DETACHED obrigatório (fora da árvore do terminal — `Start-Process`
  com logs redirecionados, ou Task Scheduler). Processos filhos da sessão do
  terminal já foram mortos 2× nesta máquina.
- ~11 h de parede em CPU; máquina não pode suspender (já configurado).
- Condição: rodar à NOITE (protocolo faseado: uma seed por noite; CPU
  dedicada — não competir com uso da máquina).
- **Já existe disparo automático agendado:** tarefa do Task Scheduler
  `CycloneNet-Ablation-Night2-Seed123`, 14/07 19h30, com pre-checks
  (aborta se houver treino ativo ou seed 42 incompleta). Se a máquina
  estiver desligada às 19h30, o gatilho NÃO re-dispara → lançar manualmente
  com o comando acima. Launcher/logs: `C:\Users\Estéfano\cyclone-net-ops\`.
- Ao terminar (6/6 células + `seed123/oof_predictions.csv` + run
  `summary.json`): commitar (o .gitignore já captura só OOF+summary) e
  agendar a noite 3.

## 2. ESTADO DO EXPERIMENTO EM ANDAMENTO (ablação de features CNN)

Protocolo: `docs/ablation_preregistration.md` (fixado em `eaa8ae8`, antes de
qualquer resultado). Detalhe operacional: `docs/ablation_progress.md`.

| Seed | Status |
|---|---|
| 42 | **COMPLETA** (run `20260713T232126Z`, commit `c608f19`; Δ PR-AUC OOF +0,033 — intermediário, SEM veredito) |
| 123 | **PENDENTE** — agendada p/ 14/07 19h30 |
| 456 | **PENDENTE** — noite 3 (15/07), agendar igual |

**REGRA EM DESTAQUE: NÃO rodar `--aggregate` com menos de 3 seeds. Nenhuma
conclusão antes do IC agregado — uma seed é ruído de inicialização. O IC é
lido UMA vez; sem garimpo, sem re-run.** Agregação final:

```
./venv/Scripts/python.exe analysis/feature_ablation_cnn.py --aggregate outputs/results/feature_ablation_cnn
```

## 3. REGRAS PERMANENTES DO PROJETO (invioláveis)

- Nenhum commit/PR leva atribuição de IA.
- `secret_guard` CLEAN antes de todo commit; nunca commitar `config.yaml`,
  `run_snapshot.json`, `.cdsapirc`, `.netrc`.
- Paths sempre relativos (`rel_to_root`) — paths absolutos com o username
  acentuado QUEBRAM leitura netCDF nesta máquina (não é só higiene, é
  funcional).
- Não editar código que um processo ativo (treino/backfill) está usando.
- Verificar antes de descartar/sobrescrever (gate de completude,
  manifest+dados juntos).
- Splits por SID hash-determinísticos + `frozen_splits.json`; test set
  congelado, nunca lido em desenvolvimento.
- Honestidade epistêmica: FuelMap = hypothesis maps; nada de inflar
  resultados.

## 4. FILA DE PENDÊNCIAS (por prioridade)

1. **EM ANDAMENTO — Ablação faseada:** seed 123 (hoje 19h30) → seed 456
   (15/07) → `--aggregate` → veredito pelos 3 ramos pré-registrados.
2. **BLOQUEADA (pelo veredito) — Pós-ablação:** aplicar o resultado à V3,
   Forma A (modelo na plataforma).
3. **TODO — PR #9:** aberto; merge é do usuário (commits recentes já
   pushados na branch `feature/tchp`).
4. **TODO — higiene/docs (detalhes a confirmar com o usuário; herdados de
   sessão anterior, sem registro em disco):** dv24 no ERRATA (arquivo
   ERRATA ainda não existe em docs/), link no README, parágrafos da V3.
5. **TODO (segurança, pendente do usuário):** rotacionar chave CDS e senha
   Copernicus (vazaram no histórico git; redação na origem já implementada).
6. **CONTÍNUO — caderno de hipóteses:** `docs/hypothesis_registry.md`
   (em inglês, padrão dos docs científicos do repo) é a agenda de pesquisa
   viva (H1..H7). Registrar hipótese+teste ANTES de rodar; veredito honesto
   depois. Próxima não-testada: H7 ("anomaly hypothesis" — desenhar o teste
   e fixar o null antes de olhar qualquer mapa).

## 5. MARCOS JÁ CONCLUÍDOS (não refazer)

- Backfill PL 20/20 (1980–2019; 21.662 eventos, zero falhas), proveniência
  completa (`884ac36`).
- Auditoria core-integrity 5/5 (pós-backfill fecha itens 3 e 5, `ff223cd`).
- Census PL: gate PASS (14.101 eventos dev, cobertura 100%).
- Plataforma no ar com painel ambiental + basin (`08ad031`).
- Higiene de paths relativos nos manifestos (`ee7dc7c`).
- Pré-registro da ablação fixado antes de qualquer resultado (`eaa8ae8`).
- Re-teste pós-backfill dos precursores RI em pares congelados: H1–H4
  significativas sob Bonferroni ×4, H2/H4 com 394/394 pares (`970a419`).
- Ablação noite 1 / seed 42 completa (`c608f19`).
- Dataset 1980–2023: 16.780 eventos válidos / 802 positivos RI / 992
  tempestades; splits sem leakage; benchmark congelado intacto.

## 6. EXECUÇÃO DE PROCESSOS LONGOS (anti-interrupção externa — OBRIGATÓRIO)

Contexto: processos longos rodando como filhos da árvore do terminal foram
mortos 2× nesta máquina (13/07 ~01h00 e ~20h19). Suspeito mais forte da 2ª:
self-update do Google Updater + sessão do RestartManager (roda a cada ~3h);
1ª inconclusiva (janela só tinha Windows Update/Defender). O padrão comum:
**só morrem processos da árvore da sessão do terminal**. Um run detached
atravessou 11 h e várias sessões do RestartManager sem ser tocado.

Regras para QUALQUER processo > ~15 min (treino, backfill, download):

1. **NUNCA rodar como filho do terminal/sessão** (nem como "background task"
   da sessão — foi exatamente o que morreu 2×).
2. **Sempre DETACHED**, com stdout/stderr em arquivo e `-u` (sem buffer):
   ```powershell
   Start-Process -FilePath ".\venv\Scripts\python.exe" `
     -ArgumentList "-u","<script>","<args...>" `
     -WorkingDirectory "<raiz do repo>" `
     -RedirectStandardOutput "C:\Users\Estéfano\cyclone-net-ops\<nome>.log" `
     -RedirectStandardError  "C:\Users\Estéfano\cyclone-net-ops\<nome>.err.log" `
     -WindowStyle Hidden -PassThru
   ```
   Ou via **Task Scheduler** (nasce fora de qualquer árvore de terminal):
   launcher `.ps1` com pre-checks em `C:\Users\Estéfano\cyclone-net-ops\`,
   gatilho com `-ExecutionTimeLimit` folgado (ex.: 16 h). Atenção: gatilho
   "Once" perdido (máquina desligada) NÃO re-dispara.
3. **Pre-checks antes de disparar** (o launcher da noite 2 é o modelo):
   etapa anterior completa? máquina livre (sem outro treino/backfill)?
   Se falhar, abortar e logar o motivo — nunca disparar em cima.
4. **Logs e progresso fora do Temp do Windows** (Temp pode ser limpo):
   usar `C:\Users\Estéfano\cyclone-net-ops\` para logs operacionais;
   artefatos científicos ficam em `outputs/` como sempre.
5. **Acompanhar por artefatos em disco, não pelo terminal**: timestamps de
   checkpoints, `ablation_eval.json` por célula, OOF/summary no fim. O
   processo não depende de ninguém observando.
6. **Máquina:** não suspender (já configurado); atualizações automáticas
   (Google Updater/Windows Update) podem rodar à noite — o detached é
   imune, mas evitar instalar/atualizar software durante um treino.
7. **Retomada:** todo fluxo longo deve ser retomável (skip-if-exists,
   manifesto por janela, OOF por célula) — se morrer, re-rodar continua,
   não recomeça. Antes de re-rodar, checar o que ficou completo em disco.

## 7. NÚMEROS-CHAVE DE REFERÊNCIA

- Modelo produção: PR-AUC 0,251 [IC 0,179–0,331], ROC-AUC 0,796.
- Dataset: 1980–2023, 16.780 eventos, 802 positivos RI (dev PL-gated:
  14.101 eventos / 687 positivos / 839 tempestades).
- Custo de treino nesta máquina (CPU): ~110 min/célula (15 épocas) →
  ~11 h por seed da ablação (6 células).
- Seed 42 (intermediário): OOF agrupado A=0,162 / B=0,195 (PR-AUC);
  ROC A=0,786 / B=0,825.
