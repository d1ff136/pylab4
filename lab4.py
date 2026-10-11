# ----------------------------
# Розвідувальний (EDA) та візуальний аналіз даних. Варіант 9.
# ----------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Налаштування стилю графіків
sns.set_theme(style="whitegrid")

# Щоб таблиці у консолі виводились повністю
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Варіант_9_spotify.csv")

# ----------------------------
# 1. Збір та перевірка даних
# ----------------------------

df = pd.read_csv(CSV_PATH)

# Розмірність датасету (кількість рядків і стовпчиків)
print(f"Розмірність датасету Spotify: {df.shape}")
print(f"Кількість рядків (днів): {df.shape[0]}")
print(f"Кількість стовпчиків (ознак): {df.shape[1]}")


# df.info(): типи даних і кількість непустих (non-null) значень
df.info()


# df.head(): перевірка таблиці
display(df.head()) if "display" in globals() else print(df.head())


# Дата зчитується як текст (object/str) — переводимо у тип datetime
df["Date"] = pd.to_datetime(df["Date"])
print("Період спостережень:", df["Date"].min().date(), "—", df["Date"].max().date())

# ----------------------------
# 2-3. Очищення даних. Описовий аналіз
# ----------------------------

# --- Пропущені значення: df.isnull().sum() ---
print("Кількість пропусків у стовпцях таблиці:")
print(df.isnull().sum())


# Таблиця: назва стовпця, кількість пропущених, кількість спостережень
mis_table = pd.DataFrame({
    "Назва стовпця": df.columns,
    "Пропущено": df.isnull().sum().values,
    "Спостережень": len(df),
})
mis_table["Відсоток (%)"] = (mis_table["Пропущено"] / mis_table["Спостережень"] * 100).round(2)

print("Кількість та відсоток пропусків у датасеті Spotify")
print(mis_table.to_string(index=False))

# Лише стовпчики, де є пропуски
print("\nСтовпчики з пропусками:")
print(mis_table[mis_table["Пропущено"] > 0].to_string(index=False))


# Де саме розташовані пропуски? Перша дата, з якої є дані по кожній пісні
song_cols = [c for c in df.columns if c != "Date"]
print("Перша дата, з якої є дані по пісні:")
for col in song_cols:
    first_valid = df.loc[df[col].notna(), "Date"].min().date()
    only_leading = df[col].isna().sum() == (df[col].first_valid_index() or 0)
    print(f"  {col:28s} {first_valid}   пропуски лише на початку: {only_leading}")

print(
    "\nВисновок: усі пропуски стоять на початку періоду — імовірно, це дні до"
    "\nвиходу пісні в чарт (прослуховувань ще не було). Це не помилки збору,"
    "\nтому заповнювати їх середнім чи медіаною не потрібно: це спотворило б"
    "\nстатистику. Пропуски залишаємо — pandas ігнорує NaN у describe(), mean() тощо."
)


# --- Аномалії ---
print("Пошук аномалій методом міжквартильного розмаху")
for col in song_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Стовпчик '{col}': знайдено викидів — {len(outliers)} "
          f"(межі: < {lower_bound:,.0f} або > {upper_bound:,.0f})")


# --- Описовий аналіз: df.describe() ---

print("Загальна статистика датасету Spotify (числові стовпчики):")
print(df.drop(columns="Date").describe().T)

print("\nСтатистика з урахуванням усіх стовпчиків (include='all'):")
print(df.describe(include="all").T)

# Додатково: сумарні прослуховування кожної пісні
print("\nСумарні прослуховування за період:")
print(df[song_cols].sum().sort_values(ascending=False).apply(lambda v: f"{v:,.0f}"))

# ----------------------------
# 4. Візуалізація даних
# ----------------------------

# --- Три гістограми з кривою щільності ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

hist_songs = ["Shape of You", "Despacito", "HUMBLE."]
colors = ["green", "orange", "blue"]

for ax, song, color in zip(axes, hist_songs, colors):
    data = df[song].dropna()  # NaN (дні до релізу) для гістограми не потрібні
    sns.histplot(data, kde=True, bins=30, color=color, ax=ax)
    ax.axvline(data.mean(), color="black", linestyle="--", label=f"Середнє: {data.mean():,.0f}")
    ax.axvline(data.median(), color="red", linestyle=":", label=f"Медіана: {data.median():,.0f}")
    ax.set_title(f"Розподіл щоденних прослуховувань: {song}")
    ax.set_xlabel("Прослуховувань за день")
    ax.set_ylabel("Частота")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "histograms.png"), dpi=150)
plt.show()


# --- 3D-діаграма ---
# Осі: X — Shape of You, Y — Despacito, Z — Something Just Like This.
# Колір точки — місяць. Беремо лише дні без пропусків по всіх трьох піснях.
cols3 = ["Shape of You", "Despacito", "Something Just Like This"]
df_clean = df.dropna(subset=cols3)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection="3d")

scatter = ax.scatter(
    df_clean[cols3[0]] / 1e6,
    df_clean[cols3[1]] / 1e6,
    df_clean[cols3[2]] / 1e6,
    c=df_clean["Date"].dt.month,
    cmap="viridis",
    alpha=0.7,
    s=30,
)

ax.set_xlabel(f"{cols3[0]}, млн")
ax.set_ylabel(f"{cols3[1]}, млн")
ax.set_zlabel(f"{cols3[2]}, млн")
ax.set_title("3D діаграма: щоденні прослуховування трьох пісень")

cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("Місяць")

plt.savefig(os.path.join(BASE_DIR, "scatter_3d.png"), dpi=150)
plt.show()

# ----------------------------
# 5. Виявлення зв'язків та кореляцій
# ----------------------------

correlation_matrix = df[song_cols].corr()
print("Матриця кореляції Пірсона:")
print(correlation_matrix.round(2))

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Теплова карта кореляції прослуховувань пісень")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "heatmap.png"), dpi=150)
plt.show()