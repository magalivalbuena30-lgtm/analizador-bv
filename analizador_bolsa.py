"""
Analizador educativo de bolsa con señales explicadas
-----------------------------------------------------
Proyecto escolar sin fines de lucro. NO es asesoría financiera.

Instalación:
    pip install streamlit yfinance pandas numpy plotly

Ejecución:
    streamlit run analizador_bolsa.py
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------
# 1. DATOS
# ----------------------------------------------------------------------
PERIODOS = {"1d": "2y", "1h": "6mo", "5m": "1mo"}


@st.cache_data(ttl=60)  # se refresca cada 60 s => "casi en vivo"
def cargar_datos(ticker: str, intervalo: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=PERIODOS[intervalo],
        interval=intervalo,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


# ----------------------------------------------------------------------
# 2. INDICADORES
# ----------------------------------------------------------------------
def rsi(close, n=14):
    delta = close.diff()
    ganancia = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    perdida = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = ganancia / perdida
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def agregar_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    df["RSI"] = rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()
    media, desv = c.rolling(20).mean(), c.rolling(20).std()
    df["BB_UP"], df["BB_LOW"] = media + 2 * desv, media - 2 * desv
    df["ATR"] = atr(df)
    df["VOL_MED"] = df["Volume"].rolling(20).mean()
    return df.dropna()


# ----------------------------------------------------------------------
# 3. MOTOR DE SEÑALES (confluencia + explicación)
# ----------------------------------------------------------------------
def evaluar(r: pd.Series, p: pd.Series):
    """Devuelve (puntaje, [razones]) para la vela r (p = vela anterior)."""
    pts, razones = 0, []

    # Tendencia de fondo
    if r.Close > r.SMA200:
        pts += 1
        razones.append("+1 | Precio sobre SMA200: la tendencia de fondo es alcista.")
    else:
        pts -= 1
        razones.append("-1 | Precio bajo SMA200: la tendencia de fondo es bajista.")

    if r.SMA50 > r.SMA200:
        pts += 1
        razones.append("+1 | SMA50 sobre SMA200: el promedio reciente supera al de largo plazo.")
    else:
        pts -= 1
        razones.append("-1 | SMA50 bajo SMA200: el promedio reciente está por debajo del de largo plazo.")

    # RSI
    if r.RSI < 30:
        pts += 2
        razones.append(f"+2 | RSI={r.RSI:.0f} (<30): sobreventa, el precio cayó muy rápido y suele rebotar.")
    elif r.RSI > 70:
        pts -= 2
        razones.append(f"-2 | RSI={r.RSI:.0f} (>70): sobrecompra, el precio subió muy rápido y suele corregir.")
    else:
        razones.append(f"0 | RSI={r.RSI:.0f}: zona neutral, no aporta señal.")

    # MACD
    if r.MACD > r.MACD_SIG and p.MACD <= p.MACD_SIG:
        pts += 2
        razones.append("+2 | Cruce alcista del MACD: el impulso acaba de cambiar a favor de las subidas.")
    elif r.MACD < r.MACD_SIG and p.MACD >= p.MACD_SIG:
        pts -= 2
        razones.append("-2 | Cruce bajista del MACD: el impulso acaba de cambiar a favor de las bajadas.")
    elif r.MACD > r.MACD_SIG:
        pts += 1
        razones.append("+1 | MACD sobre su señal: el impulso sigue siendo alcista.")
    else:
        pts -= 1
        razones.append("-1 | MACD bajo su señal: el impulso sigue siendo bajista.")

    # Bandas de Bollinger
    if r.Close < r.BB_LOW:
        pts += 1
        razones.append("+1 | Precio bajo la banda de Bollinger inferior: inusualmente bajo para su volatilidad.")
    elif r.Close > r.BB_UP:
        pts -= 1
        razones.append("-1 | Precio sobre la banda de Bollinger superior: inusualmente alto para su volatilidad.")

    # Volumen (confirma la dirección de la vela)
    if r.Volume > 1.5 * r.VOL_MED:
        if r.Close >= r.Open:
            pts += 1
            razones.append("+1 | Volumen alto con vela alcista: hay participación real respaldando la subida.")
        else:
            pts -= 1
            razones.append("-1 | Volumen alto con vela bajista: hay participación real respaldando la caída.")

    return pts, razones


def plan_operacion(r: pd.Series, capital: float, riesgo_pct: float):
    entrada = float(r.Close)
    riesgo_unit = 1.5 * float(r.ATR)
    stop = entrada - riesgo_unit
    objetivo = entrada + 2 * riesgo_unit  # relación riesgo/beneficio 1:2
    dinero_riesgo = capital * riesgo_pct / 100
    acciones = math.floor(dinero_riesgo / riesgo_unit) if riesgo_unit > 0 else 0
    return entrada, stop, objetivo, acciones, dinero_riesgo


# ----------------------------------------------------------------------
# 4. BACKTEST SIMPLE
# ----------------------------------------------------------------------
def backtest(df: pd.DataFrame, umbral: int):
    ops, en_pos = [], False
    entrada = stop = objetivo = 0.0
    for i in range(1, len(df)):
        r, p = df.iloc[i], df.iloc[i - 1]
        pts, _ = evaluar(r, p)
        if en_pos:
            if r.Low <= stop:
                ops.append(stop / entrada - 1)
                en_pos = False
            elif r.High >= objetivo:
                ops.append(objetivo / entrada - 1)
                en_pos = False
            elif pts <= -umbral:
                ops.append(r.Close / entrada - 1)
                en_pos = False
        elif pts >= umbral:
            entrada = float(r.Close)
            stop = entrada - 1.5 * float(r.ATR)
            objetivo = entrada + 3 * float(r.ATR)
            en_pos = True
    return ops


def metricas(ops):
    if not ops:
        return None
    ops = np.array(ops)
    equity = np.cumprod(1 + ops)
    pico = np.maximum.accumulate(equity)
    return {
        "Operaciones": len(ops),
        "Ganadoras (%)": round(100 * (ops > 0).mean(), 1),
        "Retorno total (%)": round(100 * (equity[-1] - 1), 1),
        "Peor caída (%)": round(100 * ((equity / pico) - 1).min(), 1),
    }


# ----------------------------------------------------------------------
# 5. GRÁFICO
# ----------------------------------------------------------------------
def grafico(df: pd.DataFrame, ticker: str):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.17, 0.18],
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Precio"), 1, 1)
    for col, color in [("SMA50", "orange"), ("SMA200", "purple"), ("BB_UP", "gray"), ("BB_LOW", "gray")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color, width=1)), 1, 1)
    fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volumen"), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI"), 3, 1)
    fig.add_hline(y=70, line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.MACD, name="MACD"), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df.MACD_SIG, name="Señal"), 4, 1)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, title=ticker,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ----------------------------------------------------------------------
# 6. INTERFAZ
# ----------------------------------------------------------------------
st.set_page_config(page_title="Analizador educativo de bolsa", layout="wide")
st.title("📈 Analizador educativo de bolsa")
st.caption("Proyecto escolar sin fines de lucro. Material educativo, no es asesoría financiera.")

with st.sidebar:
    st.header("Parámetros")
    ticker = st.text_input("Símbolo (ej. AAPL, MSFT, SPY)", "AAPL").upper().strip()
    intervalo = st.selectbox("Intervalo de velas", list(PERIODOS), index=0)
    capital = st.number_input("Capital simulado (USD)", 100.0, 1_000_000.0, 10_000.0, step=500.0)
    riesgo_pct = st.slider("Riesgo por operación (% del capital)", 0.5, 5.0, 1.0, 0.5)
    umbral = st.slider("Puntaje mínimo para señal (confluencia)", 2, 8, 4)
    if st.button("🔄 Actualizar datos"):
        st.cache_data.clear()

try:
    datos = agregar_indicadores(cargar_datos(ticker, intervalo))
except Exception as e:
    st.error(f"No se pudieron cargar datos para '{ticker}': {e}")
    st.stop()

if len(datos) < 5:
    st.warning("Datos insuficientes para calcular los indicadores.")
    st.stop()

ultima, anterior = datos.iloc[-1], datos.iloc[-2]
puntaje, razones = evaluar(ultima, anterior)

if puntaje >= umbral:
    estado = "🟢 POSIBLE ENTRADA (compra)"
elif puntaje <= -umbral:
    estado = "🔴 POSIBLE SALIDA / EVITAR"
else:
    estado = "🟡 ESPERAR (señales insuficientes)"

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader(estado)
    st.metric("Puntaje de confluencia", puntaje)
    st.metric("Último precio", f"${ultima.Close:,.2f}")
    if puntaje >= umbral:
        e, s, o, n, d = plan_operacion(ultima, capital, riesgo_pct)
        st.markdown(
            f"**Plan simulado**\n\n"
            f"- Entrada: ${e:,.2f}\n"
            f"- Stop-loss (1.5×ATR): ${s:,.2f}\n"
            f"- Objetivo (riesgo/beneficio 1:2): ${o:,.2f}\n"
            f"- Tamaño: {n} acciones (arriesgando ${d:,.0f})"
        )
with col2:
    st.subheader("¿Por qué?")
    for t in razones:
        st.write("• " + t)

st.plotly_chart(grafico(datos.tail(250), ticker), use_container_width=True)

st.subheader("Backtest con datos históricos")
m = metricas(backtest(datos, umbral))
if m:
    st.table(pd.DataFrame([m]))
    st.caption("Resultados pasados no garantizan resultados futuros. Úsalo para analizar qué señales funcionan y cuáles no.")
else:
    st.info("Con este umbral no hubo operaciones en el periodo. Prueba un puntaje menor.")

with st.expander("📚 Glosario de indicadores"):
    st.markdown(
        "- **SMA:** promedio del precio; muestra la tendencia.\n"
        "- **RSI:** mide la velocidad del movimiento (sobrecompra/sobreventa).\n"
        "- **MACD:** compara dos promedios para medir el impulso.\n"
        "- **Bollinger:** bandas de volatilidad alrededor del promedio.\n"
        "- **ATR:** volatilidad promedio; define el stop-loss.\n"
        "- **Volumen:** cantidad negociada; confirma la fuerza del movimiento."
    )
