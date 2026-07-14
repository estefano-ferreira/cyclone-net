# Ablação faseada de features (CNN) — estado e retomada

Protocolo pré-registrado em `docs/ablation_preregistration.md` (commit
`eaa8ae8`, fixado ANTES de qualquer resultado): k=3 folds, 15 épocas,
seeds {42, 123, 456} executadas UMA POR NOITE, CPU, braço A (9 canais
atuais) vs braço B (+shear_850_200_mps, +rh_mid). Veredito ÚNICO após
agregar as 3 seeds — resultados por seed são intermediários e não devem
ser interpretados isoladamente.

## Estado (2026-07-14)

| Seed | Noite | Status | Artefatos |
|---|---|---|---|
| 42 | 1 (13→14/07) | **COMPLETA** — commitada e pushada (`c608f19`) | `outputs/results/feature_ablation_cnn/20260713T232126Z/` (seed42/oof_predictions.csv + summary.json no git) |
| 123 | 2 | **PENDENTE** — ainda não rodou | — |
| 456 | 3 | **PENDENTE** | — |

Registro intermediário da seed 42 (SEM veredito): Δ PR-AUC OOF agrupado
(B−A) = +0,033; por fold +0,051 / +0,043 / +0,019. Custo real observado:
~110 min/célula → ~11 h de parede por seed (calibração original previa
8,5 h).

## Para retomar (com a máquina ligada)

1. **Seed 123** (noite 2):
   ```
   ./venv/Scripts/python.exe analysis/feature_ablation_cnn.py --folds 3 --epochs 15 --seeds 123 --execute
   ```
   - Rodar DETACHED (fora da árvore do terminal — ex.: `Start-Process` com
     logs redirecionados, ou Task Scheduler). Processos filhos da sessão do
     terminal foram mortos 2× nesta máquina (~01h00 e ~20h19 de 13/07;
     suspeito: RestartManager/Google Updater). Runs detached sobreviveram
     11 h sem problema.
   - ~11 h de parede; máquina não pode suspender (já configurado).
   - Já existe tarefa agendada **`CycloneNet-Ablation-Night2-Seed123`**
     (Task Scheduler, dispara 14/07 19h30 SE a máquina estiver ligada e
     logada; gatilho perdido NÃO re-dispara). Launcher com pre-checks:
     `C:\Users\Estéfano\cyclone-net-ops\ablation_night2_seed123.ps1` —
     aborta se a seed 42 estiver incompleta ou se houver treino ativo
     (portanto é seguro coexistir com disparo manual). Logs em
     `C:\Users\Estéfano\cyclone-net-ops\`.
2. **Seed 456** (noite 3): mesmo comando com `--seeds 456`, mesmo método.
3. Ao fim de cada seed: commitar `oof_predictions.csv` + `summary.json`
   do run dir (checkpoints .pt são gitignorados).

## Veredito final — só com as 3 seeds

```
./venv/Scripts/python.exe analysis/feature_ablation_cnn.py --aggregate outputs/results/feature_ablation_cnn
```

Calcula ΔPR-AUC médio entre seeds com IC 95% bootstrap por cluster de SID
e escreve `aggregate_*.json` com o veredito pelos 3 ramos pré-registrados.
**NÃO rodar o agregador com menos de 3 seeds. Ler o IC UMA vez; sem
garimpo, sem re-run.**
