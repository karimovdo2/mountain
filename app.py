import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from glob import glob
import streamlit as st
import pathlib


st.set_page_config(page_title="Горный год", layout="wide")

st.title("Горный год из Garmin")
st.markdown(
    """Визуализация походов в виде горных вершин.
    По оси **X** — дата похода, по оси **Y** — максимальная высота.
    Каждая вершина — PNG-изображение горы, масштабированное по высоте."""
)

st.sidebar.header("Данные")
csv_source = st.sidebar.radio(
    "Источник данных:",
    ("Локальный файл 'походы.csv'", "Загрузить CSV")
)

default_csv_path = "походы.csv"
uploaded_file = None
if csv_source == "Загрузить CSV":
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV с походами", type=["csv"])

mountains_dir = st.sidebar.text_input(
    "Папка с PNG-горами",
    value="mount",
    help="Внутри должно лежать 7 файлов PNG с горами."
)

width_ref_days = st.sidebar.slider(
    "Толщина средней горы (в днях)",
    min_value=5,
    max_value=40,
    value=25,
    step=1
)

dx_thr = st.sidebar.slider(
    "Минимальный разнос подписей по X (в днях)",
    min_value=2,
    max_value=20,
    value=8,
    step=1
)

dy_thr = st.sidebar.slider(
    "Минимальный разнос подписей по Y (в метрах)",
    min_value=50,
    max_value=400,
    value=150,
    step=10
)

base_offset = st.sidebar.slider(
    "Смещение подписи над вершиной (м)",
    min_value=20,
    max_value=200,
    value=80,
    step=10
)

dy_step = st.sidebar.slider(
    "Шаг подъёма подписи при конфликте (м)",
    min_value=50,
    max_value=300,
    value=120,
    step=10
)

@st.cache_data
def load_data_from_path(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df

@st.cache_data
def load_data_from_uploaded(file):
    df = pd.read_csv(file, encoding="utf-8-sig")
    return df

@st.cache_data
def prepare_dataframe(df_raw):
    df = df_raw.copy()
    if "Дата" not in df.columns:
        raise ValueError("В CSV нет столбца 'Дата'")
    if "Максимальная высота" not in df.columns:
        raise ValueError("В CSV нет столбца 'Максимальная высота'")

    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")
    df = df.dropna(subset=["Дата"])

    df["max_alt"] = (
        df["Максимальная высота"]
          .astype(str)
          .str.replace(",", "", regex=False)
          .astype(float)
    )
    df = df.sort_values("Дата")
    return df

@st.cache_resource
def load_mountain_images(mountains_dir):
    pattern = str(pathlib.Path(mountains_dir) / "*.png")
    paths = sorted(glob(pattern))[:7]
    if not paths:
        raise FileNotFoundError(f"Не найдено ни одного PNG в папке: {mountains_dir}")
    imgs = []
    for p in paths:
        img = plt.imread(p)
        rgb = img[..., :3]
        brightness = rgb.mean(axis=2)
        mask_rows = (brightness < 0.995).any(axis=1)
        if mask_rows.any():
            top = np.argmax(mask_rows)
            bottom = len(mask_rows) - 1 - np.argmax(mask_rows[::-1])
            img = img[top:bottom+1, :, :]
        imgs.append(img)
    return imgs

def month_year_formatter(x, pos=None):
    dt = mdates.num2date(x)
    if dt.month == 1:
        return dt.strftime('%b\n%Y')
    else:
        return dt.strftime('%b')

def create_mountain_figure(df, mountain_imgs,
                           width_ref_days=25,
                           dx_thr=8, dy_thr=150,
                           base_offset=80, dy_step=120):
    dates = df["Дата"].to_numpy()
    dates_num = mdates.date2num(dates)
    alts = df["max_alt"].to_numpy()

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_facecolor("white")

    alts_pos = alts[alts > 0]
    alt_ref = np.median(alts_pos) if len(alts_pos) > 0 else np.max(alts)
    k = width_ref_days / alt_ref if alt_ref > 0 else 0.01

    rng = np.random.default_rng(42)

    for i, (x_num, alt) in enumerate(zip(dates_num, alts)):
        if not np.isfinite(alt) or alt <= 0:
            continue

        img = mountain_imgs[rng.integers(0, len(mountain_imgs))]

        h_px, w_px = img.shape[:2]
        ratio = w_px / h_px

        width_days = alt * k * ratio

        x0 = x_num - width_days / 2
        x1 = x_num + width_days / 2
        y0 = 0
        y1 = alt

        ax.imshow(
            img,
            extent=[x0, x1, y0, y1],
            aspect="auto",
            interpolation="bilinear",
            zorder=i
        )

    # подписи вершин
    label_positions = []
    for x_num, alt in zip(dates_num, alts):
        if not np.isfinite(alt) or alt <= 0:
            continue

        text_x = x_num
        text_y = alt + base_offset

        conflict = True
        while conflict:
            conflict = False
            for lx, ly in label_positions:
                if abs(text_x - lx) < dx_thr and abs(text_y - ly) < dy_thr:
                    text_y += dy_step
                    conflict = True
                    break

        ax.annotate(
            f"{int(round(alt))}",
            xy=(x_num, alt),
            xytext=(text_x, text_y),
            textcoords="data",
            ha="center", va="bottom",
            fontsize=8,
            arrowprops=dict(
                arrowstyle="-",
                lw=0.6,
                color="gray",
                shrinkA=0,
                shrinkB=0
            ),
            zorder=1000
        )

        label_positions.append((text_x, text_y))

    dates_num_all = dates_num
    ax.set_xlim(dates_num_all.min() - 15, dates_num_all.max() + 15)
    ax.set_ylim(0, alts.max() * 1.1)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(FuncFormatter(month_year_formatter))

    ax.set_ylabel("Максимальная высота, м")
    ax.set_xlabel("Дата похода")

    plt.tight_layout()
    return fig

# ---------- Логика приложения ----------

df_raw = None
if csv_source == "Локальный файл 'походы.csv'":
    try:
        df_raw = load_data_from_path(default_csv_path)
    except FileNotFoundError:
        st.error("Файл 'походы.csv' не найден в текущей папке. "
                 "Либо положите его рядом с app.py, либо выберите вариант 'Загрузить CSV'.")
else:
    if uploaded_file is not None:
        df_raw = load_data_from_uploaded(uploaded_file)
    else:
        st.info("Загрузите CSV-файл с походами, чтобы построить график.")

if df_raw is not None:
    try:
        df = prepare_dataframe(df_raw)
    except ValueError as e:
        st.error(str(e))
    else:
        st.subheader("Сводка по походам")
        total_hikes = len(df)
        total_distance = None
        total_ascent = None
        if "Расстояние" in df_raw.columns:
            try:
                dist = (
                    df_raw["Расстояние"]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )
                total_distance = dist.sum()
            except Exception:
                total_distance = None
        if "Подъем" in df_raw.columns:
            try:
                climb = (
                    df_raw["Подъем"]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )
                total_ascent = climb.sum()
            except Exception:
                total_ascent = None

        col1, col2, col3 = st.columns(3)
        col1.metric("Походов всего", total_hikes)
        if total_distance is not None:
            col2.metric("Дистанция суммарно, км", f"{total_distance:.1f}")
        else:
            col2.write("Суммарная дистанция: нет корректного столбца 'Расстояние'")
        if total_ascent is not None:
            col3.metric("Набор высоты суммарно, м", f"{total_ascent:.0f}")
        else:
            col3.write("Суммарный набор: нет корректного столбца 'Подъем'")

        try:
            mountain_imgs = load_mountain_images(mountains_dir)
        except Exception as e:
            st.error(f"Ошибка при загрузке картинок гор: {e}")
        else:
            fig = create_mountain_figure(
                df,
                mountain_imgs,
                width_ref_days=width_ref_days,
                dx_thr=dx_thr,
                dy_thr=dy_thr,
                base_offset=base_offset,
                dy_step=dy_step
            )
            st.pyplot(fig)
