#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard individual do aluno (FocusAI)
--------------------------------------
Lê users_v2.csv, tasks_v2.csv, sessions_v2.csv e gera um painel com:
- KPIs do aluno (tarefas, taxa de conclusão, duração/distrações)
- Gráficos de evolução, sessões, complexidade, temas etc.
- Salva tudo em: reports/etapa2/dashboard_user_<id>/

Agora com input direto: o script pergunta o user_id quando executado.
"""

from pathlib import Path
import re
from collections import Counter
import numpy as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- Input do usuário -----------------------
try:
    USER_ID = int(input("Digite o user_id que deseja gerar o dashboard: ").strip())
except ValueError:
    print("❌ Por favor, insira um número inteiro válido para o user_id.")
    raise SystemExit(0)

# ----------------------- Configurações -----------------------
BASE = Path(".").resolve()
USERS_CSV = BASE / "users_v2.csv"
TASKS_CSV = BASE / "tasks_v2.csv"
SESS_CSV  = BASE / "sessions_v2.csv"

OUT_DIR = BASE / "reports" / "etapa2" / f"dashboard_user_{USER_ID}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------- Função segura para ler CSV -----------------------
def _safe_read(csv: Path) -> pd.DataFrame:
    if not csv.exists():
        print(f"[WARN] Arquivo não encontrado: {csv}")
        return pd.DataFrame()
    return pd.read_csv(csv)

# ----------------------- Load -----------------------
users = _safe_read(USERS_CSV)
tasks = _safe_read(TASKS_CSV)
sessions = _safe_read(SESS_CSV)

if users.empty or tasks.empty or sessions.empty:
    print("[ERROR] CSVs não encontrados ou vazios. Execute o script no mesmo diretório dos arquivos v2.")
    raise SystemExit(0)

# ----------------------- ETL mínimo -----------------------
tasks["data_criacao_parsed"] = pd.to_datetime(tasks["data_criacao"], errors="coerce").fillna(
    pd.to_datetime(tasks["data_criacao"], errors="coerce", dayfirst=True)
)
tasks["data_limite_parsed"] = pd.to_datetime(tasks["data_limite"], errors="coerce").fillna(
    pd.to_datetime(tasks["data_limite"], errors="coerce", dayfirst=True)
)
sessions["inicio_parsed"] = pd.to_datetime(sessions["inicio"], errors="coerce")
sessions["fim_parsed"] = pd.to_datetime(sessions["fim"], errors="coerce")

def norm_status(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().capitalize()
    if x.startswith("Conclu"): return "Concluída"
    if x.startswith("Pende"):  return "Pendente"
    if x.startswith("Adi"):    return "Adiada"
    return x

tasks = tasks.drop_duplicates(subset=["task_id"]).copy()
tasks["status_norm"] = tasks["status"].apply(norm_status)
tasks["data_criacao"] = tasks["data_criacao_parsed"]
tasks["data_limite"]  = tasks["data_limite_parsed"]

sessions = sessions.dropna(subset=["inicio_parsed","fim_parsed"]).copy()
sessions = sessions[sessions["fim_parsed"] >= sessions["inicio_parsed"]]
sessions = sessions[sessions["duracao_min"].between(10,360)]
sessions = sessions[sessions["distracoes"] >= 0]

# ----------------------- Filtro por user_id -----------------------
user_row = users.loc[users["user_id"]==USER_ID]
if user_row.empty:
    print(f"[ERROR] user_id {USER_ID} não encontrado em users_v2.csv.")
    raise SystemExit(0)

tasks_u = tasks.loc[tasks["user_id"]==USER_ID].copy()
sessions_u = sessions.loc[sessions["user_id"]==USER_ID].copy()

# ----------------------- KPIs -----------------------
total_tasks = len(tasks_u)
taxa_conc = (tasks_u["status_norm"]=="Concluída").mean() if total_tasks>0 else np.nan
por_complex = tasks_u.groupby("complexidade")["task_id"].count().rename("qtd").reset_index()
conc_complex = (tasks_u.assign(is_done=tasks_u["status_norm"].eq("Concluída"))
                .groupby("complexidade")["is_done"].mean().reset_index())

dur_med = sessions_u["duracao_min"].median() if len(sessions_u)>0 else np.nan
distr_med = sessions_u["distracoes"].median() if len(sessions_u)>0 else np.nan
perfil = user_row.iloc[0].get("perfil", user_row.iloc[0].get("perfil_estudo","-"))
nivel_foco = user_row.iloc[0].get("nivel_foco","-")
curso = user_row.iloc[0].get("curso","-")
nome = user_row.iloc[0].get("nome","Aluno")

# ----------------------- Evolução diária (barras legíveis + meta; diário OU semanal) -----------------------
# 1) garantir só a data (sem hora/minuto)
tasks_u["dia"] = pd.to_datetime(tasks_u["data_criacao"], errors="coerce").dt.normalize()

# 2) agregar por dia
daily = (tasks_u.groupby("dia")
         .agg(criados=("task_id", "count"),
              concluidos=("status_norm", lambda s: (s == "Concluída").sum()))
         .sort_index())

# 3) reindexar intervalo completo para preencher dias sem registro (0)
if not daily.empty:
    full_range = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    daily.index.name = "dia"


# 4) meta diária dinâmica baseada no histórico do próprio aluno
media_concluidas_dia = daily["concluidos"].mean() if len(daily) > 0 else 0

# meta = 20% acima da média, mínimo 1 tarefa/dia
META_DIARIA = max(1, int(round(media_concluidas_dia * 1.2)))
print(f"[INFO] Meta diária calculada para {nome}: {META_DIARIA} tarefas/dia")


# 5) granularidade automática
USE_WEEKLY = len(daily) > 60
if USE_WEEKLY:
    agg = daily.resample("W-SUN").sum()
    x_vals = agg.index
    criados_vals = agg["criados"].to_numpy()
    concluidos_vals = agg["concluidos"].to_numpy()
    meta_vals = np.full(len(agg), META_DIARIA * 7)
    titulo = f"Evolução semanal de tarefas - {nome} (agregado)"
    ylabel = "Tarefas por semana"
    outfile = OUT_DIR / "evolucao_semanal.png"
    labels = []
    for end in x_vals:
        start = (end - pd.Timedelta(days=6)).date()
        labels.append(f"{start.strftime('%d/%m')}–{end.date().strftime('%d/%m')}")
else:
    agg = daily
    x_vals = agg.index
    criados_vals = agg["criados"].to_numpy()
    concluidos_vals = agg["concluidos"].to_numpy()
    meta_vals = np.full(len(agg), META_DIARIA)
    titulo = f"Evolução diária de tarefas - {nome}"
    ylabel = "Tarefas por dia"
    outfile = OUT_DIR / "evolucao_diaria.png"
    labels = [pd.Timestamp(d).strftime("%d/%m") for d in x_vals]

# 6) salvar agregado
agg.reset_index().rename(columns={"index": "dia"}).to_csv(OUT_DIR / "evolucao_agrupada.csv", index=False)

# 7) gráfico de barras (sem formatters de data do Matplotlib)
fig, ax = plt.subplots(figsize=(10, 5))
idx = np.arange(len(x_vals))
width = 0.4

ax.bar(idx - width/2, criados_vals, width, label="Criadas")
ax.bar(idx + width/2, concluidos_vals, width, label="Concluídas")
ax.plot(idx, meta_vals, linestyle="--", label=("Meta semanal" if USE_WEEKLY else "Meta diária"))

# reduzir rótulos para não poluir
max_labels = 10
step = max(1, len(labels) // max_labels)
shown_labels = [labels[i] if i % step == 0 else "" for i in range(len(labels))]

ax.set_xticks(idx)
ax.set_xticklabels(shown_labels, rotation=45, ha="right")
ax.set_title(titulo)
ax.set_xlabel("Semana" if USE_WEEKLY else "Data")
ax.set_ylabel(ylabel)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig(outfile)
plt.close(fig)

# ----------------------- Sessões por hora -----------------------
sessions_u["hora"] = sessions_u["inicio_parsed"].dt.hour
hora_counts = sessions_u["hora"].value_counts().sort_index()

# ----------------------- Top termos (matérias/temas) -----------------------
def tokenize(s):
    if pd.isna(s): return []
    s = re.sub(r"[^A-Za-zÀ-ÿ0-9 ]+", " ", str(s))
    return [t.lower() for t in s.split() if len(t) > 3]

freq = Counter()
for desc in tasks_u["descricao"].tolist():
    for tok in tokenize(desc):
        freq[tok] += 1
top_terms = pd.DataFrame(freq.most_common(15), columns=["termo","freq"])

# ----------------------- Gráfico: Acompanhamento de tarefas concluídas (acumulado) -----------------------
tarefas_concluidas = (tasks_u.loc[tasks_u["status_norm"] == "Concluída"]
                      .groupby("dia")["task_id"].count()
                      .rename("concluidas_no_dia")
                      .reset_index()
                      .sort_values("dia"))
tarefas_concluidas["total_concluidas"] = tarefas_concluidas["concluidas_no_dia"].cumsum()
tarefas_concluidas.to_csv(OUT_DIR / "tarefas_concluidas_por_dia.csv", index=False)

plt.figure(figsize=(9,4))
plt.plot(tarefas_concluidas["dia"], tarefas_concluidas["total_concluidas"], marker="o", linestyle="-", label="Concluídas (acumulado)")
plt.title(f"Acompanhamento de tarefas concluídas - {nome}")
plt.xlabel("Data")
plt.ylabel("Total concluídas")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "acompanhamento_tarefas_concluidas.png")
plt.close()

# ----------------------- Demais gráficos -----------------------
plt.figure(figsize=(8,4))
hora_counts.plot(kind="bar")
plt.title(f"Sessões por hora do dia - {nome}")
plt.xlabel("Hora")
plt.ylabel("Sessões")
plt.tight_layout()
plt.savefig(OUT_DIR / "sessoes_por_hora.png")
plt.close()

plt.figure(figsize=(6,4))
plt.boxplot(sessions_u["duracao_min"].dropna().values, vert=True, labels=["Duração (min)"])
plt.title(f"Duração das sessões - {nome}")
plt.tight_layout()
plt.savefig(OUT_DIR / "boxplot_duracao.png")
plt.close()

plt.figure(figsize=(6,4))
plt.bar(por_complex["complexidade"], por_complex["qtd"])
plt.title(f"Tarefas por complexidade - {nome}")
plt.xlabel("Complexidade")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig(OUT_DIR / "tarefas_por_complexidade.png")
plt.close()

plt.figure(figsize=(6,4))
plt.bar(conc_complex["complexidade"], conc_complex["is_done"])
plt.title(f"Taxa de conclusão por complexidade - {nome}")
plt.xlabel("Complexidade")
plt.ylabel("Taxa de conclusão")
plt.tight_layout()
plt.savefig(OUT_DIR / "taxa_conclusao_por_complexidade.png")
plt.close()

plt.figure(figsize=(7,5))
plt.barh(top_terms["termo"].iloc[::-1], top_terms["freq"].iloc[::-1])
plt.title(f"Temas mais estudados - {nome}")
plt.xlabel("Frequência")
plt.tight_layout()
plt.savefig(OUT_DIR / "top_termos.png")
plt.close()

print(f"[OK] Dashboard gerado para o user_id={USER_ID}. Arquivos salvos em {OUT_DIR}")
