"""
FocusAI
--------------------------------
Script que reproduz o pipeline de limpeza mínima e as análises exploratórias
usadas na Etapa 2, com base nos CSVs v2 users_v2.csv, tasks_v2.csv, sessions_v2.csv
"""
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


print("[INFO] Carregando CSVs...")
users = pd.read_csv("users_v2.csv")
tasks = pd.read_csv("tasks_v2.csv")
sessions = pd.read_csv("sessions_v2.csv")

# Datas
iso_cr = pd.to_datetime(tasks["data_criacao"], errors="coerce")
alt_cr = pd.to_datetime(tasks["data_criacao"], errors="coerce", dayfirst=True)
tasks["data_criacao_parsed"] = iso_cr.fillna(alt_cr)

iso_dl = pd.to_datetime(tasks["data_limite"], errors="coerce")
alt_dl = pd.to_datetime(tasks["data_limite"], errors="coerce", dayfirst=True)
tasks["data_limite_parsed"] = iso_dl.fillna(alt_dl)

sessions["inicio_parsed"] = pd.to_datetime(sessions["inicio"], errors="coerce")
sessions["fim_parsed"] = pd.to_datetime(sessions["fim"], errors="coerce")

# Qualidade
dq = {
    "tasks_null_desc": tasks["descricao"].isna().sum(),
    "tasks_null_prio": tasks["prioridade"].isna().sum(),
    "tasks_null_status": tasks["status"].isna().sum(),
    "tasks_bad_date_criacao": tasks["data_criacao_parsed"].isna().sum(),
    "tasks_bad_date_limite": tasks["data_limite_parsed"].isna().sum(),
    "sessions_bad_inicio": sessions["inicio_parsed"].isna().sum(),
    "sessions_bad_fim": sessions["fim_parsed"].isna().sum(),
    "sessions_end_before_start": (sessions["fim_parsed"] < sessions["inicio_parsed"]).sum(),
    "sessions_dur_outlier_low": (sessions["duracao_min"] < 10).sum(),
    "sessions_dur_outlier_high": (sessions["duracao_min"] > 360).sum(),
    "sessions_negative_distracoes": (sessions["distracoes"] < 0).sum(),
    "sessions_fk_break": (~sessions["task_id"].isin(tasks["task_id"])).sum()
}
print("\n=== 1) Data Quality Snapshot ===")
print(pd.DataFrame([dq]).to_string(index=False))

# Normalizações
map_prio = {"alta":"Alta","Alta":"Alta","ALTA":"Alta","alta ":"Alta",
            "Média":"Média","media":"Média","MEDIA":"Média",
            "Baixa":"Baixa","baixa":"Baixa","BAIXA":"Baixa"}

def normalize_prio(x):
    if pd.isna(x): return np.nan
    x = str(x).strip()
    if x.lower().startswith("high") or "hight" in x.lower():
        return "Alta"
    if x in ["—","-",""]:
        return np.nan
    return map_prio.get(x, x.capitalize())

def normalize_status(x):
    if pd.isna(x): return np.nan
    x = str(x).strip().capitalize()
    if x.startswith("Conclu"): return "Concluída"
    if x.startswith("Pende"): return "Pendente"
    if x.startswith("Adi"):   return "Adiada"
    return x

tasks_clean = tasks.drop_duplicates(subset=["task_id"]).copy()
tasks_clean["prioridade_norm"] = tasks_clean["prioridade"].apply(normalize_prio)
tasks_clean["status_norm"] = tasks_clean["status"].apply(normalize_status)
tasks_clean["data_criacao"] = tasks["data_criacao_parsed"]
tasks_clean["data_limite"]  = tasks["data_limite_parsed"]

sessions_clean = sessions.dropna(subset=["inicio_parsed","fim_parsed"]).copy()
sessions_clean = sessions_clean[sessions_clean["fim_parsed"] >= sessions_clean["inicio_parsed"]]
sessions_clean = sessions_clean[sessions_clean["duracao_min"].between(10,360)]
sessions_clean = sessions_clean[sessions_clean["distracoes"] >= 0]
sessions_clean = sessions_clean.merge(
    tasks_clean[["task_id","complexidade"]],
    on="task_id", how="inner"
)

users_enriched = users.rename(columns={"perfil":"perfil_estudo"})
tasks_enriched = tasks_clean.merge(users_enriched[["user_id","curso","nivel_foco","perfil_estudo"]], on="user_id", how="left")
sessions_enriched = sessions_clean.merge(users_enriched[["user_id","curso","nivel_foco","perfil_estudo"]], on="user_id", how="left")

# KPIs
prod_user = (
    tasks_enriched.assign(is_done=tasks_enriched["status_norm"].eq("Concluída"))
    .groupby(["user_id","perfil_estudo","curso"])["is_done"]
    .agg(["count","mean"]).reset_index()
    .rename(columns={"count":"qtd_tarefas","mean":"taxa_conclusao"})
)

comp_counts = tasks_enriched.groupby("complexidade")["task_id"].count().reset_index().sort_values("task_id", ascending=False)

tasks_enriched["dow_idx"] = tasks_enriched["data_criacao"].dt.dayofweek
dow_map = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
tasks_enriched["dow"] = tasks_enriched["dow_idx"].map(dow_map).fillna("Unknown")
dow_counts = tasks_enriched.groupby("dow")["task_id"].count().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","Unknown"]
).fillna(0).reset_index()

sessions_enriched["hora"] = sessions_enriched["inicio_parsed"].dt.hour
hora_counts = sessions_enriched["hora"].value_counts().sort_index()

dur_por_perfil = sessions_enriched.groupby("perfil_estudo")["duracao_min"].agg(["count","median","mean","std"]).reset_index()

done_by_complex = tasks_enriched.assign(is_done=tasks_enriched["status_norm"].eq("Concluída")).groupby("complexidade")["is_done"].mean().reset_index()

prod_curso = tasks_enriched.assign(is_done=tasks_enriched["status_norm"].eq("Concluída")).groupby("curso")["is_done"].agg(["count","mean"]).reset_index().rename(columns={"count":"tarefas_total","mean":"taxa_conclusao"})

def tokenize(s):
    if pd.isna(s): return []
    s = re.sub(r"[^A-Za-zÀ-ÿ0-9 ]+", " ", str(s))
    return [t.lower() for t in s.split() if len(t) > 3]

freq = Counter()
for desc in tasks_enriched["descricao"].tolist():
    for tok in tokenize(desc):
        freq[tok] += 1
top_terms = pd.DataFrame(freq.most_common(30), columns=["termo","freq"])

agg_user = sessions_enriched.groupby("user_id").agg(
    dur_med=("duracao_min","mean"),
    distr_med=("distracoes","mean")
).reset_index().merge(
    tasks_enriched.assign(done=tasks_enriched["status_norm"].eq("Concluída")).groupby("user_id")["done"].mean().reset_index(),
    on="user_id", how="left"
).rename(columns={"done":"taxa_conclusao"})

summary_df = pd.DataFrame([{
    "tarefas_total_pos_dedup": int(len(tasks_enriched)),
    "sessions_total_limpa": int(len(sessions_enriched)),
    "periodo": f'{tasks_enriched["data_criacao"].min().date()} a {tasks_enriched["data_criacao"].max().date()}'
}])

# Prints rápidos
print("\n=== 2) Produtividade por aluno (amostra) ===")
print(prod_user.head(10).to_string(index=False))
print("\n=== 3) Distribuição de tarefas por complexidade ===")
print(comp_counts.to_string(index=False))
print("\n=== 4) Tarefas por dia da semana ===")
print(dow_counts.to_string(index=False))
print("\n=== 5) Sessões por hora (contagem) ===")
print(hora_counts.to_string())
print("\n=== 6) Duração por perfil ===")
print(dur_por_perfil.to_string(index=False))
print("\n=== 7) Taxa de conclusão por complexidade ===")
print(done_by_complex.to_string(index=False))
print("\n=== 8) Produtividade por curso ===")
print(prod_curso.to_string(index=False))
print("\n=== 9) Top termos (NLP simples) ===")
print(top_terms.to_string(index=False))
print("\n=== 10) Agregado por usuário ===")
print(agg_user.to_string(index=False))
print("\n=== 11) Resumo do dataset ===")
print(summary_df.to_string(index=False))

# Gráficos
print("\n[INFO] Gerando gráficos em reports/etapa2 ...")

plt.figure(figsize=(8,4))
plt.bar(dow_counts["dow"], dow_counts["task_id"])
plt.title("Tarefas por dia da semana")
plt.xticks(rotation=45)
plt.ylabel("Quantidade")
plt.xlabel("Dia da semana")
plt.tight_layout()
plt.savefig("tarefas_por_dia_da_semana.png")
plt.close()

plt.figure(figsize=(8,4))
hora_counts.plot(kind="bar")
plt.title("Sessões por hora do dia")
plt.xlabel("Hora")
plt.ylabel("Sessões")
plt.tight_layout()
plt.savefig("sessoes_por_hora.png")
plt.close()

plt.figure(figsize=(8,4))
sessions_enriched.boxplot(column="duracao_min", by="perfil_estudo")
plt.title("Duração de sessões por perfil")
plt.suptitle("")
plt.xlabel("Perfil de estudo")
plt.ylabel("Duração (min)")
plt.tight_layout()
plt.savefig("duracao_por_perfil.png")
plt.close()

plt.figure(figsize=(8,6))
plt.barh(top_terms["termo"].iloc[::-1], top_terms["freq"].iloc[::-1])
plt.title("Top termos nas descrições (30)")
plt.xlabel("Frequência")
plt.tight_layout()
plt.savefig("top_termos_nlp.png")
plt.close()

plt.figure(figsize=(6,5))
plt.scatter(agg_user["dur_med"], agg_user["taxa_conclusao"])
plt.title("Duração média vs. taxa de conclusão (por usuário)")
plt.xlabel("Duração média (min)")
plt.ylabel("Taxa de conclusão")
plt.tight_layout()
plt.savefig("duracao_media_vs_taxa_conclusao.png")
plt.close()

print("[OK] Concluído.")


# ===== Recomendações por usuário (horário e método) =====
def _choose_hour(group):
    by_h = group.groupby("hora").agg(
        sessoes=("session_id","count"),
        concluidas=("status_norm", lambda s: (s=="Concluída").sum())
    )
    by_h["taxa"] = by_h["concluidas"] / by_h["sessoes"]
    by_h = by_h.sort_values(["taxa","sessoes"], ascending=[False, False])
    return int(by_h.index[0]) if not by_h.empty else 9

def _choose_method(durations, distractions):
    med = durations.median() if len(durations) else 50
    distr_med = distractions.median() if len(distractions) else 1
    if (med <= 40) or (distr_med >= 2):
        return ("Pomodoro", 30)
    if (med >= 90) and (distr_med <= 1):
        return ("Intensivo", 90)
    return ("Moderado", 50)

def _window_2h(h):
    return f"{h:02d}:00–{(h+2)%24:02d}:00"

# sessions_enriched precisa conter: user_id, session_id, inicio_parsed, duracao_min, distracoes
# tasks_enriched precisa conter: task_id, status_norm
_sess = sessions_enriched.merge(tasks_enriched[["task_id","status_norm"]], on="task_id", how="inner")
_sess["hora"] = _sess["inicio_parsed"].dt.hour

recs = []
for uid, g in _sess.groupby("user_id"):
    h = _choose_hour(g)
    method, minutes = _choose_method(g["duracao_min"], g["distracoes"])
    recs.append({"user_id": uid, "hora_recomendada": h, "janela_recomendada": _window_2h(h),
                 "metodo_recomendado": method, "bloco_minutos": minutes})

recs_df = pd.DataFrame(recs).merge(users_enriched[["user_id","nome","perfil_estudo","curso","nivel_foco"]], on="user_id", how="left")
recs_df.to_csv("recommendations.csv", index=False)
print(recs_df)
print("[OK] Recomendações salvas em", "recommendations.csv")

# ======================= NOVOS INDICADORES (idade, padrões, consistência) =======================
print("\n[INFO] Gerando NOVOS INDICADORES...")

# --- 0) Enriquecimentos auxiliares ---
# idade no nível de tarefas e sessões
tasks_age = tasks_enriched.merge(users_enriched[["user_id","idade"]], on="user_id", how="left")
sessions_age = sessions_enriched.merge(users_enriched[["user_id","idade"]], on="user_id", how="left")

# dia/semana, fim de semana
sessions_age["dia"] = sessions_age["inicio_parsed"].dt.date
sessions_age["dow_idx"] = sessions_age["inicio_parsed"].dt.dayofweek
sessions_age["is_weekend"] = sessions_age["dow_idx"] >= 5

# faixas etárias
bins_idade = [0, 20, 24, 29, 34, 200]
labels_idade = ["<=20", "21–24", "25–29", "30–34", "35+"]
users_enriched["faixa_idade"] = pd.cut(users_enriched["idade"], bins=bins_idade, labels=labels_idade, right=True)
tasks_age = tasks_age.merge(users_enriched[["user_id","faixa_idade"]], on="user_id", how="left")
sessions_age = sessions_age.merge(users_enriched[["user_id","faixa_idade"]], on="user_id", how="left")

# --- 1) Sessões por idade (contagem) ---
sess_por_idade = sessions_age.groupby("idade")["session_id"].count().reset_index().rename(columns={"session_id":"sessoes"})
print("\n=== 12) Sessões por idade (contagem) ===")
print(sess_por_idade.head(15).to_string(index=False))

plt.figure(figsize=(8,4))
plt.plot(sess_por_idade["idade"], sess_por_idade["sessoes"], marker="o")
plt.title("Sessões por idade")
plt.xlabel("Idade")
plt.ylabel("Nº de sessões")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("sessoes_por_idade.png")
plt.close()

# --- 2) Duração de sessão por idade (média/mediana) ---
dur_por_idade = sessions_age.groupby("idade")["duracao_min"].agg(["count","mean","median"]).reset_index()
print("\n=== 13) Duração de sessão por idade (média/mediana) ===")
print(dur_por_idade.head(15).to_string(index=False))

plt.figure(figsize=(8,4))
plt.plot(dur_por_idade["idade"], dur_por_idade["mean"], marker="o", label="Média")
plt.plot(dur_por_idade["idade"], dur_por_idade["median"], marker="o", linestyle="--", label="Mediana")
plt.title("Duração de sessão por idade")
plt.xlabel("Idade")
plt.ylabel("Duração (min)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("duracao_sessao_por_idade.png")
plt.close()

# --- 3) Indicadores por faixa etária (sessões, duração, conclusão) ---
# taxa de conclusão por faixa etária
taxa_conc_idade = (tasks_age.assign(done=tasks_age["status_norm"].eq("Concluída"))
                   .groupby("faixa_idade")["done"].mean().reset_index()
                   .rename(columns={"done":"taxa_conclusao"}))
# sessões e duração por faixa etária
sess_idade_stats = (sessions_age.groupby("faixa_idade")["session_id"].count().reset_index()
                    .rename(columns={"session_id":"sessoes"}))
dur_idade_stats = sessions_age.groupby("faixa_idade")["duracao_min"].agg(["mean","median"]).reset_index()

idade_merge = sess_idade_stats.merge(dur_idade_stats, on="faixa_idade", how="left").merge(taxa_conc_idade, on="faixa_idade", how="left")
print("\n=== 14) Painel por faixa etária (sessões, duração, taxa de conclusão) ===")
print(idade_merge.to_string(index=False))

plt.figure(figsize=(8,4))
plt.bar(idade_merge["faixa_idade"].astype(str), idade_merge["sessoes"])
plt.title("Sessões por faixa etária")
plt.xlabel("Faixa etária")
plt.ylabel("Nº de sessões")
plt.tight_layout()
plt.savefig("sessoes_por_faixa_etaria.png")
plt.close()

plt.figure(figsize=(8,4))
x = np.arange(len(idade_merge))
w = 0.35
plt.bar(x - w/2, idade_merge["mean"], width=w, label="Média")
plt.bar(x + w/2, idade_merge["median"], width=w, label="Mediana")
plt.xticks(x, idade_merge["faixa_idade"].astype(str))
plt.title("Duração de sessão por faixa etária")
plt.ylabel("Minutos")
plt.legend()
plt.tight_layout()
plt.savefig("duracao_por_faixa_etaria.png")
plt.close()

plt.figure(figsize=(6,4))
plt.bar(idade_merge["faixa_idade"].astype(str), idade_merge["taxa_conclusao"])
plt.title("Taxa de conclusão por faixa etária")
plt.xlabel("Faixa etária")
plt.ylabel("Taxa de conclusão")
plt.tight_layout()
plt.savefig("taxa_conclusao_por_faixa_etaria.png")
plt.close()

# --- 4) Dias úteis vs fim de semana (por perfil) ---
weekend_profile = (sessions_enriched.assign(is_weekend=sessions_enriched["inicio_parsed"].dt.dayofweek >= 5)
                   .groupby(["perfil_estudo","is_weekend"])["session_id"].count()
                   .reset_index()
                   .pivot(index="perfil_estudo", columns="is_weekend", values="session_id")
                   .fillna(0).rename(columns={False:"Dias úteis", True:"Fins de semana"}))
print("\n=== 15) Sessões por perfil: dias úteis vs fins de semana ===")
print(weekend_profile.to_string())

plt.figure(figsize=(8,4))
weekend_profile.plot(kind="bar")
plt.title("Sessões por perfil — úteis vs fim de semana")
plt.ylabel("Nº de sessões")
plt.xlabel("Perfil")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sessoes_perfil_uteis_vs_fds.png")
plt.close()

# --- 5) Consistência: maior streak de dias com ≥1 sessão por usuário ---
def longest_streak(dates):
    if not len(dates): return 0
    ds = sorted(set(pd.to_datetime(dates)))
    best = cur = 1
    for i in range(1, len(ds)):
        if (ds[i] - ds[i-1]).days == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best

streaks = (sessions_enriched.assign(dia=sessions_enriched["inicio_parsed"].dt.date)
           .groupby("user_id")["dia"].apply(longest_streak)
           .reset_index().rename(columns={"dia":"maior_streak"}))
streaks = streaks.merge(users_enriched[["user_id","nome","perfil_estudo","idade"]], on="user_id", how="left").sort_values("maior_streak", ascending=False)

print("\n=== 16) Top 10 maior sequência de dias com estudo ===")
print(streaks.head(10).to_string(index=False))
streaks.to_csv("streaks_por_usuario.csv", index=False)

# --- 6) Eficiência: (tarefas concluídas) por (horas estudadas) ---
hrs_por_user = (sessions_enriched.groupby("user_id")["duracao_min"].sum() / 60.0).reset_index().rename(columns={"duracao_min":"horas_estudadas"})
conc_por_user = (tasks_enriched.assign(done=tasks_enriched["status_norm"].eq("Concluída"))
                 .groupby("user_id")["done"].sum().reset_index().rename(columns={"done":"tarefas_concluidas"}))
eff = hrs_por_user.merge(conc_por_user, on="user_id", how="outer").fillna(0.0)
eff["eficiencia_tarefa_por_hora"] = eff.apply(lambda r: (r["tarefas_concluidas"] / r["horas_estudadas"]) if r["horas_estudadas"] > 0 else np.nan, axis=1)
eff = eff.merge(users_enriched[["user_id","nome","perfil_estudo","curso","idade"]], on="user_id", how="left")
print("\n=== 17) Eficiência (tarefas concluídas por hora estudada) — top 10 ===")
print(eff.sort_values("eficiencia_tarefa_por_hora", ascending=False).head(10).to_string(index=False))
eff.to_csv("eficiencia_por_usuario.csv", index=False)

plt.figure(figsize=(7,5))
plt.scatter(eff["horas_estudadas"], eff["tarefas_concluidas"])
plt.title("Tarefas concluídas vs horas estudadas (por usuário)")
plt.xlabel("Horas estudadas")
plt.ylabel("Tarefas concluídas")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("tarefas_vs_horas.png")
plt.close()

# --- 7) Coorte semanal simples: tarefas concluídas por semana (global e por perfil) ---
tasks_week = (
    tasks_enriched
    .assign(
        is_done=tasks_enriched["status_norm"].eq("Concluída"),
        semana=tasks_enriched["data_criacao"].apply(
            lambda d: d.to_period("W").start_time if pd.notnull(d) else pd.NaT
        )
    )
    .dropna(subset=["semana"])
    .groupby("semana")["is_done"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(9,4))
plt.plot(tasks_week["semana"], tasks_week["is_done"], marker="o")
plt.title("Concluídas por semana (global)")
plt.xlabel("Semana")
plt.ylabel("Tarefas concluídas")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("concluidas_por_semana_global.png")
plt.close()

tasks_week_profile = (
    tasks_enriched
    .assign(
        is_done=tasks_enriched["status_norm"].eq("Concluída"),
        semana=tasks_enriched["data_criacao"].apply(
            lambda d: d.to_period("W").start_time if pd.notnull(d) else pd.NaT
        )
    )
    .dropna(subset=["semana"])
    .groupby(["semana","perfil_estudo"])["is_done"]
    .sum()
    .reset_index()
)
# Pivot para gráfico múltiplo (linhas por perfil)
pivot_w = tasks_week_profile.pivot(index="semana", columns="perfil_estudo", values="is_done").fillna(0)

plt.figure(figsize=(10,5))
for col in pivot_w.columns:
    plt.plot(pivot_w.index, pivot_w[col], marker="o", label=col)
plt.title("Concluídas por semana (por perfil de estudo)")
plt.xlabel("Semana")
plt.ylabel("Tarefas concluídas")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("concluidas_por_semana_por_perfil.png")
plt.close()


# --- 8) Heatmap hora × dia da semana (nº de sessões) ---
heat = sessions_enriched.copy()
heat["dow"] = heat["inicio_parsed"].dt.day_name()
heat["hora"] = heat["inicio_parsed"].dt.hour
order_dow = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
heat["dow"] = pd.Categorical(heat["dow"], categories=order_dow, ordered=True)
mat = (heat.groupby(["dow","hora"])["session_id"].count().reset_index()
       .pivot(index="dow", columns="hora", values="session_id").fillna(0))

plt.figure(figsize=(12,4.5))
plt.imshow(mat.values, aspect="auto", interpolation="nearest")
plt.colorbar(label="Nº de sessões")
plt.yticks(range(len(mat.index)), mat.index)
plt.xticks(range(24), range(24))
plt.title("Heatmap sessões (hora × dia da semana)")
plt.xlabel("Hora do dia")
plt.ylabel("Dia da semana")
plt.tight_layout()
plt.savefig("heatmap_sessoes_hora_dia.png")
plt.close()

print("[OK] Novos indicadores gerados.")


# ======================= MÉTODO ANALÍTICO (Classificação, Clusterização, Regras) =======================
print("\n[INFO] Iniciando MÉTODO ANALÍTICO (Classificação, Clusterização, Regras)...")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------- Utilidades ----------
def _priority_to_ord(x):
    if pd.isna(x): return np.nan
    d = {"Baixa":0, "Média":1, "Media":1, "Alta":2}
    return d.get(str(x).strip().capitalize(), np.nan)

def _complex_to_ord(x):
    d = {"básica":0, "basica":0, "média":1, "media":1, "difícil":2, "dificil":2}
    return d.get(str(x).strip().lower(), np.nan)

def _hour_from_sessions_for_task(sessions_df):
    # média de hora por task (se não houver sessão, fica NaN)
    tmp = sessions_df.assign(hora=sessions_df["inicio_parsed"].dt.hour).groupby("task_id")["hora"].mean().rename("hora_media_task")
    return tmp
def _add_user_history_features(df_tasks_sorted):
    """
    Para cada user, calcula features históricas SEM vazamento:
      - taxa_conc_past7 / taxa_conc_past30 (janelas temporais por data_criacao)
      - distr_med_past7 (média de distrações das sessões últimos 7 dias)
    Requer data_criacao válida (Datetime). Linhas sem data viram NaN nas features.
    """
    # garante datetime
    df = df_tasks_sorted.copy()
    df["data_criacao"] = pd.to_datetime(df["data_criacao"], errors="coerce")

    out = []
    for uid, g in df.groupby("user_id", group_keys=False):
        g = g.copy()

        # remove NaT temporariamente para o cálculo temporal
        mask_valid = g["data_criacao"].notna()
        g_valid = g.loc[mask_valid].copy()

        if len(g_valid) == 0:
            # nada a calcular; devolve com NaN nas features
            g["taxa_conc_past7"] = np.nan
            g["taxa_conc_past30"] = np.nan
            g["distr_med_past7"] = np.nan
            out.append(g)
            continue

        # index temporal + ordenação
        g_valid = g_valid.set_index("data_criacao").sort_index()

        # calcula y_done DEPOIS de setar o índice (necessário para rolling com janela temporal)
        y_done = (g_valid["status_norm"] == "Concluída").astype(float)

        # rolling temporal com janelas de tempo; shift(1) evita vazamento do próprio instante
        g_valid["taxa_conc_past7"]  = y_done.rolling(window="7D",  min_periods=1).mean().shift(1)
        g_valid["taxa_conc_past30"] = y_done.rolling(window="30D", min_periods=1).mean().shift(1)

        # volta a ser coluna
        g_valid = g_valid.reset_index()

        # reintroduz linhas sem data (ficam NaN nas features)
        g = g.merge(
            g_valid[["task_id", "taxa_conc_past7", "taxa_conc_past30"]],
            on="task_id", how="left"
        )

        out.append(g)

    df_hist = pd.concat(out, ignore_index=True)

    # distracoes médias dos últimos 7 dias (via sessions)
    sess = sessions_enriched[["user_id","inicio_parsed","distracoes"]].dropna().copy()
    sess["dia"] = sess["inicio_parsed"].dt.floor("D")

    # média de distrações por user-dia
    distr_daily = (sess.groupby(["user_id","dia"])["distracoes"]
                      .mean().reset_index())

    # para cada task, pegar média nos últimos 7 dias do usuário (exclusivo do dia atual)
    def _merge_past7_user(row):
        uid = row["user_id"]
        dt  = pd.to_datetime(row["data_criacao"], errors="coerce")
        if pd.isna(dt):
            return np.nan
        mask = (
            (distr_daily["user_id"] == uid) &
            (distr_daily["dia"] < dt.normalize()) &
            (distr_daily["dia"] >= (dt.normalize() - pd.Timedelta(days=7)))
        )
        v = distr_daily.loc[mask, "distracoes"]
        return v.mean() if len(v) > 0 else np.nan

    df_hist["distr_med_past7"] = df_hist.apply(_merge_past7_user, axis=1)
    return df_hist

# MÉTODO ANALÍTICO por Classificação
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import joblib

print("\n[INFO] Iniciando método analítico — Classificação (RandomForest)")

# preparação
tasks_ml = tasks_enriched.copy()
tasks_ml["target_done"] = (tasks_ml["status_norm"] == "Concluída").astype(int)
tasks_ml = tasks_ml.dropna(subset=["data_criacao"]).sort_values(["user_id","data_criacao"])

tasks_ml["hora"] = tasks_ml["data_criacao"].dt.hour
tasks_ml["dow"]  = tasks_ml["data_criacao"].dt.dayofweek
tasks_ml["mes"]  = tasks_ml["data_criacao"].dt.month

def _prio_ord(x):
    s = str(x).strip().lower() if pd.notna(x) else ""
    if s.startswith("alta"): return 3
    if s.startswith("méd"):  return 2
    if s.startswith("bai"):  return 1
    return np.nan

def _comp_ord(x):
    s = str(x).strip().lower() if pd.notna(x) else ""
    if s.startswith("dif"): return 3
    if s.startswith("méd"): return 2
    if s.startswith("bás"): return 1
    return np.nan

tasks_ml["prioridade_ord"] = tasks_ml["prioridade_norm"].apply(_prio_ord)
tasks_ml["complex_ord"]    = tasks_ml["complexidade"].apply(_comp_ord)

# histórico cumulativo anterior por usuário 
tasks_ml["hist_taxa_conc"] = (
    tasks_ml.groupby("user_id")["target_done"]
            .expanding()
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
)
tasks_ml["hist_taxa_conc"] = tasks_ml["hist_taxa_conc"].fillna(tasks_ml["target_done"].mean())

tasks_ml["descricao"] = tasks_ml["descricao"].astype(str)

feature_cols = [
    "descricao",
    "hora","dow","mes",
    "prioridade_ord","complex_ord",
    "perfil_estudo","curso","nivel_foco",
    "hist_taxa_conc"
]
for c in feature_cols:
    if c not in tasks_ml.columns:
        tasks_ml[c] = np.nan

X = tasks_ml[feature_cols].copy()
y = tasks_ml["target_done"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

#  pré-processamento 
text_col = "descricao"
cat_cols  = ["perfil_estudo","curso","nivel_foco"]
num_cols  = ["hora","dow","mes","prioridade_ord","complex_ord","hist_taxa_conc"]

preproc = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=1000, ngram_range=(1,2)), text_col),
        ("cat",  Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("num",  Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

rf_pipe = Pipeline([
    ("prep", preproc),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    ))
])

rf_pipe.fit(X_train, y_train)
y_pred  = rf_pipe.predict(X_test)
y_proba = rf_pipe.predict_proba(X_test)[:, 1]

print("\n=== RandomForest (teste) ===")
print(classification_report(y_test, y_pred, digits=3))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f} | F1-macro: {f1_score(y_test, y_pred, average='macro'):.3f}")

joblib.dump(rf_pipe, "rf_pipeline_focusai.pkl")
print("[OK] Pipeline salvo em rf_pipeline_focusai.pkl")

def prever_chance_conclusao(user_id, descricao, prioridade, complexidade, data_criacao):
    pipe = joblib.load("rf_pipeline_focusai.pkl")
    u = users_enriched.loc[users_enriched["user_id"] == user_id]
    if u.empty:
        raise ValueError(f"user_id {user_id} não encontrado.")

    perfil = u.iloc[0]["perfil_estudo"]
    curso  = u.iloc[0]["curso"]
    nivel  = u.iloc[0]["nivel_foco"]

    dc = pd.to_datetime(data_criacao)
    hora, dow, mes = dc.hour, dc.dayofweek, dc.month

    def prio_ord(x):
        s = str(x).lower()
        if "alta" in s: return 3
        if "méd"  in s: return 2
        if "bai"  in s: return 1
        return np.nan

    def comp_ord(x):
        s = str(x).lower()
        if "dif" in s: return 3
        if "méd" in s: return 2
        if "bás" in s: return 1
        return np.nan

    prioridade_ord = prio_ord(prioridade)
    complex_ord    = comp_ord(complexidade)

    hist_user = (
        tasks_enriched.loc[tasks_enriched["user_id"] == user_id]
        .assign(td=lambda d: (d["status_norm"]=="Concluída").astype(int))
    )
    hist_taxa = hist_user["td"].mean() if not hist_user.empty else float(tasks_ml["target_done"].mean())

    row = pd.DataFrame([{
        "descricao": descricao,
        "hora": hora, "dow": dow, "mes": mes,
        "prioridade_ord": prioridade_ord, "complex_ord": complex_ord,
        "perfil_estudo": perfil, "curso": curso, "nivel_foco": nivel,
        "hist_taxa_conc": hist_taxa
    }])[feature_cols]  

    prob = pipe.predict_proba(row)[0, 1]
    return round(prob * 100, 2)

# exemplo usando o modelo treinado
chance = prever_chance_conclusao(
    user_id=1,
    descricao="Revisar capítulo de cálculo diferencial",
    prioridade="Alta",
    complexidade="Média",
    data_criacao="2025-12-05 09:00"
)
print(f"Chance de conclusão (user 1, exemplo): {chance}%")
