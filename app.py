import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Funciones para el calculo de los modos

def rsi(data, window=14):
    delta = data.diff()
    ganacia = delta.where(delta > 0, 0).rolling(window=window).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = ganacia / perdida
    return 100 - (100 / (1 + rs))

def macd(close, rapida=12, larga=26, signal=9):
    ema_rapida    = close.ewm(span=rapida,   adjust=False).mean()
    ema_larga   = close.ewm(span=larga,   adjust=False).mean()
    macd  = ema_rapida - ema_larga
    señal = macd.ewm(span=signal, adjust=False).mean()
    return macd, señal, macd - señal

def atr(alto, low, close, window=14):
    prev_cierre = close.shift(1)
    tr = pd.concat([alto - low,
                    (alto - prev_cierre).abs(),
                    (low  - prev_cierre).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def modo_normal(df):
    df = df.copy()
    #se reduce la dimensionalidad a un valor escalar
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    #RSI
    df['RSI'] = rsi(close)
    # osilador
    low_min  = low.rolling(14).min()
    high_max = high.rolling(14).max()
    df['Estocastico_K'] = 100 * ((close - low_min) / (high_max - low_min))
    df['Estocastico_D'] = df['Estocastico_K'].rolling(3).mean()
    # bandas de bollinger
    df['BB_Media'] = close.rolling(20).mean()
    bb_desv        = close.rolling(20).std()
    df['BB_Alta']  = df['BB_Media'] + 2 * bb_desv
    df['BB_low']  = df['BB_Media'] - 2 * bb_desv

    #MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd(close)
    # ATR
    df['ATR'] = atr(high, low, close)

    #Calculamos las señales de compra y venta

    df['Señal_RSI']   = (df['RSI'] < 30).astype(int) - (df['RSI'] > 70).astype(int)
    df['Señal_Estoc'] = ((df['Estocastico_K'] < 20) & (df['Estocastico_K'] > df['Estocastico_D'])).astype(int)
    df['Señal_BB']    = (close < df['BB_low']).astype(int) - (close > df['BB_Alta']).astype(int)

    macd_alc = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    macd_baj = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    df['Señal_MACD'] = macd_alc.astype(int) - macd_baj.astype(int)

    atr_med = df['ATR'].rolling(50).median()
    df['ATR_Activo'] = (df['ATR'] > atr_med).astype(int)

    suma = df['Señal_RSI'] + df['Señal_Estoc'] + df['Señal_BB'] + df['Señal_MACD']
    df['Señal'] = ((suma >= 2) & (df['ATR_Activo'] == 1)).astype(int) \
                - ((suma <= -2) & (df['ATR_Activo'] == 1)).astype(int)

    df['Volatilidad'] = close.pct_change().rolling(20).std()
    return df

def dca(df, ventana=30):
    df = df.copy()
    close = df['Close'].squeeze()
    df['DCA_30']        = close.rolling(window=ventana).mean()
    df['DCA_Acumulado'] = close.expanding().mean()
    df['DCA_Señal']     = (close < df['DCA_30'] * 0.95).astype(int)
    return df

def monte_carlo(rendimientos, num_simulaciones=1000):
    cov_matrix = rendimientos.cov()
    np.random.seed(42)
    retornos_sim     = np.random.multivariate_normal(rendimientos.mean().values,
                                                     cov_matrix.values, num_simulaciones)
    retornos_cartera = retornos_sim.mean(axis=1)
    sigma = retornos_cartera.std()
    return {'media': retornos_cartera.mean(), 'sigma': sigma,
            'sharpe': retornos_cartera.mean() / sigma if sigma != 0 else 0,
            'var_95': np.percentile(retornos_cartera, 5),
            'simulaciones': retornos_sim}

def caminata_aleatoria(close, n_pasos=60, n_trayectorias=300, semilla=42):
    np.random.seed(semilla)
    rend  = close.pct_change().dropna()
    mu, sigma = rend.mean(), rend.std()
    precio_0  = float(close.iloc[-1])
    shocks    = np.random.normal(mu, sigma, size=(n_pasos, n_trayectorias))
    tray      = precio_0 * np.cumprod(1 + shocks, axis=0)
    fechas    = pd.bdate_range(start=close.index[-1], periods=n_pasos + 1, freq='B')[1:]
    return pd.DataFrame(tray, index=fechas), mu, sigma

def señal_venta_volatil(close, df_dca, df_tray, var_95):
    precio_actual  = float(close.iloc[-1])
    dca_acum       = float(df_dca['DCA_Acumulado'].iloc[-1])
    mediana_futura = float(df_tray.median(axis=1).iloc[-1])
    cond1 = precio_actual > dca_acum * 1.15
    cond2 = mediana_futura < precio_actual * 0.95
    cond3 = var_95 < -0.03
    razones = []
    if cond1: razones.append(f"Precio ({precio_actual:.2f}) supera DCA×1.15 ({dca_acum*1.15:.2f})")
    if cond2: razones.append(f"Proyección mediana ({mediana_futura:.2f}) cae >5% del precio actual")
    if cond3: razones.append(f"VaR 95% ({var_95:.2%}) indica mucho riesgo")
    return (cond1 or cond2 or cond3), razones

def backtesting_normal(close, señales, capital_inicial=10_000):
    capital, posicion, en_mercado = capital_inicial, 0.0, False
    trades, equity = [], []
    close_vals, señal_vals, fechas = close.values, señales.values, close.index
    for i in range(1, len(close_vals)):
        precio = float(close_vals[i])
        señal  = int(señal_vals[i - 1])
        if señal == 1 and not en_mercado:
            posicion, en_mercado = capital / precio, True
            trades.append({"Fecha entrada": fechas[i], "Precio entrada": round(precio, 2),
                           "Fecha salida": None, "Precio salida": None,
                           "Ganancia $": None, "Ganancia %": None, "Resultado": None})
        elif señal == -1 and en_mercado:
            valor_salida = posicion * precio
            ganancia     = valor_salida - capital
            capital, en_mercado = valor_salida, False
            if trades:
                trades[-1].update({"Fecha salida": fechas[i], "Precio salida": round(precio, 2),
                    "Ganancia $": round(ganancia, 2), "Ganancia %": round(ganancia / (capital - ganancia) * 100, 2),
                    "Resultado": "Ganancia" if ganancia >= 0 else "Pérdida"})
        equity.append({"Fecha": fechas[i], "Estrategia": (posicion * precio) if en_mercado else capital})
    if en_mercado and trades:
        pc = float(close_vals[-1])
        g  = posicion * pc - capital
        trades[-1].update({"Fecha salida": fechas[-1], "Precio salida": round(pc, 2),
            "Ganancia $": round(g, 2), "Ganancia %": round(g / capital * 100, 2), "Resultado": "⏳ Abierta"})
    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity).set_index("Fecha")
    unidades_bh = capital_inicial / float(close_vals[1])
    df_equity["Buy and Hold"] = unidades_bh * close_vals[1:]
    capital_final = posicion * float(close_vals[-1]) if en_mercado else capital
    return df_trades, df_equity, capital_final

def ganancias_dca(close, df_dca, capital_por_compra=1_000):
    señales_idx     = df_dca[df_dca["DCA_Señal"] == 1].index
    close_r         = close.reindex(df_dca.index)
    total_inv, total_uni, compras = 0.0, 0.0, []
    for fecha in señales_idx:
        precio   = float(close_r.loc[fecha])
        uni      = capital_por_compra / precio
        total_inv += capital_por_compra
        total_uni += uni
        compras.append({"Fecha": fecha, "Precio compra": round(precio, 2),
                        "Capital invertido": capital_por_compra,
                        "Unidades adquiridas": round(uni, 6)})
    precio_actual  = float(close.iloc[-1])
    valor_actual   = total_uni * precio_actual
    gan_total      = valor_actual - total_inv
    gan_pct        = (gan_total / total_inv * 100) if total_inv > 0 else 0
    precio_inicio  = float(close.iloc[0])
    uni_bh         = total_inv / precio_inicio if precio_inicio > 0 else 0
    gan_bh         = uni_bh * precio_actual - total_inv
    resumen = {"total_invertido": total_inv, "valor_actual": valor_actual,
               "ganancia_$": gan_total, "ganancia_%": gan_pct,
               "ganancia_bh_$": gan_bh,
               "ganancia_bh_%": (gan_bh / total_inv * 100) if total_inv > 0 else 0,
               "num_compras": len(compras)}
    return pd.DataFrame(compras), resumen

def render_backtesting_normal(ticker, capital_inicial):
    """Recalcula y muestra el backtesting con el capital dado."""
    datos_ind = st.session_state.datos_indicadores
    if ticker not in datos_ind or f"{ticker}_close" not in datos_ind:
        return
    df_ind = datos_ind[ticker]
    close  = datos_ind[f"{ticker}_close"]

    df_trades, df_equity, capital_final = backtesting_normal(
        close, df_ind["Señal"], capital_inicial=capital_inicial)

    ganancia_total   = capital_final - capital_inicial
    ganancia_pct_bt  = ganancia_total / capital_inicial * 100
    bh_final         = df_equity["Buy and Hold"].iloc[-1] if not df_equity.empty else capital_inicial
    bh_ganancia      = bh_final - capital_inicial
    bh_pct           = bh_ganancia / capital_inicial * 100
    trades_gan       = len(df_trades[df_trades["Resultado"] == "✅ Ganancia"]) if not df_trades.empty else 0
    trades_tot       = len(df_trades[df_trades["Resultado"].notna()]) if not df_trades.empty else 0
    win_rate         = (trades_gan / trades_tot * 100) if trades_tot > 0 else 0

    cb1, cb2, cb3, cb4 = st.columns(4)
    cb1.metric("Capital Final", f"${capital_final:,.2f}", delta=f"{ganancia_pct_bt:+.1f}%")
    cb2.metric("Ganancia del Capital Inicial", f"${ganancia_total:+,.2f}")
    cb3.metric("Buy and Hold", f"${bh_final:,.2f}", delta=f"{bh_pct:+.1f}%")
    cb4.metric("Deferencia a nuestra estrategia", f"${ganancia_total - bh_ganancia:+,.2f}")

def render_ganancias_dca(ticker, capital_por_compra):
    """Recalcula y muestra las ganancias DCA con el capital dado."""
    datos_ind = st.session_state.datos_indicadores
    if f"{ticker}_DCA" not in datos_ind or f"{ticker}_close" not in datos_ind:
        return
    df_dca = datos_ind[f"{ticker}_DCA"]
    close  = datos_ind[f"{ticker}_close"]

    df_compras, resumen = ganancias_dca(close, df_dca, capital_por_compra=capital_por_compra)

    cd1, cd2, cd3 = st.columns(3)
    cd1.metric("Total invertido (DCA)",  f"${resumen['total_invertido']:,.2f}")
    cd2.metric("Valor actual (DCA)",     f"${resumen['valor_actual']:,.2f}",
               delta=f"{resumen['ganancia_%']:+.1f}%")
    cd3.metric("Ganancia DCA ($)",       f"${resumen['ganancia_$']:+,.2f}")

st.set_page_config(page_title="Aplicacion interactiva análisis financiero", layout="wide")

for key, val in [('tickers', []), ('modo', 'Normal'), ('datos_cargados', False),
                 ('num_tickers', 3), ('datos_indicadores', {}),
                 ('capitales_bt', {}), ('capitales_dca', {})]:
    if key not in st.session_state:
        st.session_state[key] = val

st.title("Análisis Financiero para evaluar activos financieros")
st.markdown("---")

# carga de tickers
with st.expander("Carga de los activos", expanded=True):
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        num_tickers = st.number_input("¿Cuántos tickers quieres analizar?",
                                      min_value=1, max_value=20,
                                      value=st.session_state.num_tickers,
                                      step=1, key="num_tickers_input")
        st.session_state.num_tickers = num_tickers
    with col2:
        if st.button("Reiniciar activos", use_container_width=True):
            st.session_state.tickers          = []
            st.session_state.datos_cargados   = False
            st.session_state.datos_indicadores = {}
            st.session_state.capitales_bt     = {}
            st.session_state.capitales_dca    = {}
            st.rerun()
    st.subheader("Ingresa los tickers de los activos a análizar")
    while len(st.session_state.tickers) < num_tickers:
        st.session_state.tickers.append("")
    while len(st.session_state.tickers) > num_tickers:
        st.session_state.tickers.pop()

    cols = st.columns(3)
    for i in range(num_tickers):
        with cols[i % 3]:
            val = st.text_input(f"Ticker {i+1}", value=st.session_state.tickers[i],
                                key=f"ticker_input_{i}", placeholder="Ej: AAPL, BIMBOA.MX, NAFTRACISHRS.MX", help = "Revisa si el ticker esta escrito correctamente")
            st.session_state.tickers[i] = val.strip().upper() if val and val.strip() else ""

    tickers_validos = [t for t in st.session_state.tickers if t]
    if tickers_validos:
        st.success(f"Tickers cargados: {', '.join(tickers_validos)}")
    else:
        st.warning("No se ha ingresado ningún ticker")
# Seleccion de la estrategia para analisis
st.markdown("---")
st.subheader("Seleacciona la estrategia")

modo = st.radio("Según tu criterio, como consideras que se esta comportado el mercado de activos:",
                ["Normal (Estable)", "Atípico (Volátil / Crisis)"],
                horizontal=True,
                index=0 if st.session_state.modo == "Normal" else 1)
st.session_state.modo = "Normal" if "Normal" in modo else "Atípico"

col1, col2 = st.columns(2)
with col1:
    periodo   = st.selectbox("Período de análisis:", ["1mo","3mo","6mo","1y","2y","5y","max"], index=3, help= "mo = mes, y = año, max = máximo historico")
with col2:
    intervalo = st.selectbox("Intervalo de datos:", ["1d","1h","15m","5m"], index=0, help = " d = dia, h = hora, m = minutos")

tickers_actuales = [t for t in st.session_state.tickers if t]

if tickers_actuales:
    st.markdown("---")
    if st.session_state.modo == "Normal":
        st.subheader("Capital inicial invertido por activo ")
        cols_cap = st.columns(min(len(tickers_actuales), 4))
        for i, t in enumerate(tickers_actuales):
            with cols_cap[i % 4]:
                val_guardado = st.session_state.capitales_bt.get(t, 10_000)
                nuevo_val = st.number_input(f"Capital — {t} ",
                                            min_value=100, max_value=1_000_000,
                                            value=val_guardado, step=500,
                                            key=f"cap_bt_{t}")
                st.session_state.capitales_bt[t] = nuevo_val
    else:
        st.subheader("Capital por DCA (Promedio de Costo en Dólares)")
        cols_cap = st.columns(min(len(tickers_actuales), 4))
        for i, t in enumerate(tickers_actuales):
            with cols_cap[i % 4]:
                val_guardado = st.session_state.capitales_dca.get(t, 1_000)
                nuevo_val = st.number_input(f"Capital DCA — {t} ",
                                            min_value=50, max_value=100_000,
                                            value=val_guardado, step=100,
                                            key=f"cap_dca_{t}")
                st.session_state.capitales_dca[t] = nuevo_val

if st.button("Análizar", type="primary", use_container_width=True):
    tickers_validos = [t for t in st.session_state.tickers if t]
    if not tickers_validos:
        st.error("Debes ingresar al menos un ticker antes de ejecutar el análisis")
        st.stop()

    with st.spinner(f"Descargando los datos para los tickers"):
        datos = {}
        for ticker in tickers_validos:
            try:
                df = yf.download(ticker, period=periodo, interval=intervalo,
                                 progress=False, auto_adjust=True)
                if not df.empty:
                    datos[ticker] = df
                else:
                    st.warning(f"No se han encontraron datos para {ticker}")
            except Exception as e:
                st.error(f"Hay un error al descargar {ticker}: {e} :(")

    if not datos:
        st.error("No se pudo descargar ningún dato :( lo sentimos")
        st.stop()

    # Guardar todo en session_state para que no se pierda
    st.session_state.datos_cargados    = True
    st.session_state.datos_indicadores = {}

    if st.session_state.modo == "Normal":
        for ticker, df_raw in datos.items():
            df_ind = modo_normal(df_raw)
            close  = df_raw['Close'].squeeze()
            st.session_state.datos_indicadores[ticker]            = df_ind
            st.session_state.datos_indicadores[f"{ticker}_close"] = close
    else:
        precios      = {t: datos[t]['Close'].squeeze() for t in datos}
        df_precios   = pd.DataFrame(precios).dropna()
        rendimientos = df_precios.pct_change().dropna()
        sim          = monte_carlo(rendimientos)
        st.session_state.datos_indicadores['simulacion']    = sim
        st.session_state.datos_indicadores['rendimientos']  = rendimientos
        st.session_state.datos_indicadores['correlaciones'] = rendimientos.corr()
        for ticker, df_raw in datos.items():
            close  = df_raw['Close'].squeeze()
            df_dca = dca(df_raw)
            df_tray, _, _ = caminata_aleatoria(close)
            st.session_state.datos_indicadores[f"{ticker}_close"]       = close
            st.session_state.datos_indicadores[f"{ticker}_DCA"]         = df_dca
            st.session_state.datos_indicadores[f"{ticker}_trayectorias"] = df_tray


# visualizacíon de los datos
if st.session_state.datos_cargados and st.session_state.datos_indicadores:
    st.markdown("---")
    datos_ind = st.session_state.datos_indicadores

    # Escenario normal
    if st.session_state.modo == "Normal":
        st.subheader("Análisis Técnico")

        tickers_disp = [k for k in datos_ind if not k.endswith("_close")
                        and not k.endswith("_DCA") and not k.endswith("_trayectorias")
                        and k not in ("simulacion","rendimientos","correlaciones")]

        for ticker in tickers_disp:
            if f"{ticker}_close" not in datos_ind:
                continue
            df_ind = datos_ind[ticker]
            close  = datos_ind[f"{ticker}_close"]

            st.markdown(f"### {ticker}")

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                vertical_spacing=0.04, row_heights=[0.40,0.20,0.20,0.20],
                                subplot_titles=(f"{ticker} — Precio y Bollinger",
                                                "RSI / Estocástico", "MACD", "ATR"))

            fig.add_trace(go.Scatter(x=df_ind.index, y=close,
                                     name='Precio', line=dict(color='royalblue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['BB_Media'],
                                     name='BB Media', line=dict(color='orange', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['BB_Alta'],
                                     name='BB Superior', line=dict(color='gray', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['BB_low'],
                                     name='BB Inferior', line=dict(color='gray', dash='dot'),
                                     fill='tonexty', fillcolor='rgba(128,128,128,0.08)'), row=1, col=1)

            compras = df_ind[df_ind['Señal'] ==  1]
            ventas  = df_ind[df_ind['Señal'] == -1]
            if not compras.empty:
                fig.add_trace(go.Scatter(x=compras.index, y=close[compras.index], mode='markers',
                                         name='Señal Compra',
                                         marker=dict(symbol='triangle-up', size=12, color='lime')), row=1, col=1)
            if not ventas.empty:
                fig.add_trace(go.Scatter(x=ventas.index, y=close[ventas.index], mode='markers',
                                         name='Señal Venta',
                                         marker=dict(symbol='triangle-down', size=12, color='red')), row=1, col=1)

            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['RSI'],
                                     name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Estocastico_K'],
                                     name='Estocastico %K', line=dict(color='teal', dash='dash')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Estocastico_D'],
                                     name='Estocastico %D', line=dict(color='coral', dash='dot')), row=2, col=1)
            for lvl, clr in [(70,'red'),(30,'green'),(80,'red'),(20,'green')]:
                fig.add_hline(y=lvl, line_dash="dash", line_color=clr, opacity=0.4, row=2, col=1)

            colors_hist = ['green' if v >= 0 else 'red' for v in df_ind['MACD_Hist']]
            fig.add_trace(go.Bar(x=df_ind.index, y=df_ind['MACD_Hist'],
                                 name='MACD de los datos', marker_color=colors_hist, opacity=0.6), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MACD'],
                                     name='MACD', line=dict(color='dodgerblue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MACD_Signal'],
                                     name='Señal del MACD', line=dict(color='tomato', dash='dash')), row=3, col=1)
            fig.add_hline(y=0, line_color='gray', line_dash='dot', row=3, col=1)

            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['ATR'],
                                     name='ATR', line=dict(color='goldenrod'),
                                     fill='tozeroy', fillcolor='rgba(218,165,32,0.12)'), row=4, col=1)

            fig.update_layout(height=1000, showlegend=True, legend=dict(orientation='h', y=-0.05))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            señal_val = int(df_ind['Señal'].iloc[-1])
            señal_txt = "COMPRAR" if señal_val == 1 else "VENDER" if señal_val == -1 else "NEUTRAL (mantener)"
            c1.metric("RSI",            f"{df_ind['RSI'].iloc[-1]:.1f}")
            c2.metric("Estocástico %K", f"{df_ind['Estocastico_K'].iloc[-1]:.1f}")
            c3.metric("MACD",           f"{df_ind['MACD'].iloc[-1]:.3f}")
            c4.metric("ATR",            f"{df_ind['ATR'].iloc[-1]:.2f}")
            c5.metric("Señal", señal_txt)

            with st.expander(f"Tabla de indicadores — {ticker}"):
                cols_m = ['RSI','Estocastico_K','Estocastico_D','BB_Alta','BB_Media','BB_low',
                          'MACD','MACD_Signal','MACD_Hist','ATR','Volatilidad','Señal']
                st.dataframe(df_ind[cols_m].tail(15), use_container_width=True)

            # Backtesting del capital guardado en el  session_state
            st.markdown("Backtesting")
            capital_bt = st.session_state.capitales_bt.get(ticker, 10_000)
            render_backtesting_normal(ticker, capital_bt)

    # Escenario de crisis
    else:
        st.subheader("Análisis con la crisis")

        # Correlación
        if 'correlaciones' in datos_ind:
            st.markdown("Matriz de Correlación")
            corr = datos_ind['correlaciones']
            fig_corr = go.Figure(go.Heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index),
                                            colorscale='RdBu', zmin=-1, zmax=1,
                                            text=np.round(corr.values, 2),
                                            texttemplate='%{text}', textfont={"size": 12}))
            fig_corr.update_layout(title="Correlación entre activos", height=450)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Por ticker
        tickers_disp = [k.replace("_close","") for k in datos_ind if k.endswith("_close")]
        sim          = datos_ind.get('simulacion', {})

        for ticker in tickers_disp:
            close  = datos_ind[f"{ticker}_close"]
            df_dca = datos_ind[f"{ticker}_DCA"]
            df_tray = datos_ind[f"{ticker}_trayectorias"]

            st.markdown(f"---\n### {ticker}")

            vender, razones_venta = señal_venta_volatil(close, df_dca, df_tray,
                                                         sim.get('var_95', 0))
            if vender:
                st.error(f"Señal de venta para **{ticker}**")
                for r in razones_venta:
                    st.warning(f"• {r}")
            else:
                st.success(f"Sin señal de venta para **{ticker}** (mantener)")

            # Gráfico caminata aleatoria y DCA
            percentiles  = [10, 25, 50, 75, 90]
            perc_vals    = np.percentile(df_tray.values, percentiles, axis=1)
            colores_perc = ['#d73027','#fc8d59','#91bfdb','#4575b4','#313695']

            fig_walk = go.Figure()
            for p, pv, col in zip(percentiles, perc_vals, colores_perc):
                fig_walk.add_trace(go.Scatter(x=df_tray.index, y=pv, mode='lines',
                                              name=f'P{p}', line=dict(color=col, width=1.5, dash='dot'), opacity=0.7))
            fig_walk.add_trace(go.Scatter(
                x=list(df_tray.index) + list(df_tray.index[::-1]),
                y=list(perc_vals[3]) + list(perc_vals[1][::-1]),
                fill='toself', fillcolor='rgba(145,191,219,0.20)',
                line=dict(color='rgba(255,255,255,0)'), name='Rango P25-P75'))
            fig_walk.add_trace(go.Scatter(x=close.index, y=close,
                                          name='Precio histórico', line=dict(color='royalblue', width=2)))
            fig_walk.add_trace(go.Scatter(x=df_dca.index, y=df_dca['DCA_Acumulado'],
                                          name='DCA Acumulado', line=dict(color='green', dash='dash')))
            fig_walk.add_trace(go.Scatter(x=df_dca.index, y=df_dca['DCA_30'],
                                          name='DCA 30d', line=dict(color='orange', dash='dot')))

            señales_compra = df_dca[df_dca['DCA_Señal'] == 1]
            if not señales_compra.empty:
                fig_walk.add_trace(go.Scatter(x=señales_compra.index, y=close[señales_compra.index],
                                              mode='markers', name='Señal Compra DCA',
                                              marker=dict(symbol='triangle-up', size=14, color='lime')))

            uf_str = str(close.index[-1].date()) if hasattr(close.index[-1], 'date') else str(close.index[-1])
            fig_walk.add_shape(type='line', x0=uf_str, x1=uf_str, y0=0, y1=1,
                               xref='x', yref='paper', line=dict(color='white', dash='dash', width=1.5), opacity=0.5)
            fig_walk.add_annotation(x=uf_str, y=1, xref='x', yref='paper',
                                    text='Hoy', showarrow=False,
                                    font=dict(color='white', size=11), xanchor='left', yanchor='bottom')

            fig_walk.update_layout(
                title=f"{ticker} — Histórico y Proyección ",
                height=500, showlegend=True, legend=dict(orientation='h', y=-0.15),
                xaxis_title="Fecha", yaxis_title="Precio")
            st.plotly_chart(fig_walk, use_container_width=True)

            c1, c3 = st.columns(2)
            precio_actual = float(close.iloc[-1])
            dca_acum      = float(df_dca['DCA_Acumulado'].iloc[-1])
            mediana_60d   = float(np.percentile(df_tray.values, 50, axis=1)[-1])
            c1.metric("Precio Actual", f"${precio_actual:.2f}")
            c3.metric("Proyección Mediana (60 dias)", f"${mediana_60d:.2f}",
                      delta=f"{((mediana_60d/precio_actual)-1)*100:.1f}%")

            # Ganancias con el DCA con  el capital guardado en  el session_state
            st.markdown("Ganancias de la Estrategia")
            capital_dca = st.session_state.capitales_dca.get(ticker, 1_000)
            render_ganancias_dca(ticker, capital_dca)

        # Monte Carlo
        if sim:
            st.markdown("Simulación Monte Carlo")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rendimiento Esperado", f"{sim['media']:.2%}")
            c2.metric("Volatilidad",          f"{sim['sigma']:.2%}")
            c3.metric("Sharpe Ratio",         f"{sim['sharpe']:.2f}")
            c4.metric("VaR 95%",              f"{sim['var_95']:.2%}")

            retornos_cartera = sim['simulaciones'].mean(axis=1)
            fig_hist = go.Figure(go.Histogram(x=retornos_cartera, nbinsx=60,
                                              marker_color='steelblue', opacity=0.75))
            for xval, color, label, xanchor in [
                (sim['var_95'], 'red',  f"VaR 95%: {sim['var_95']:.2%}", 'right'),
                (sim['media'],  'lime', f"Media: {sim['media']:.2%}",     'left'),
            ]:
                fig_hist.add_shape(type='line', x0=xval, x1=xval, y0=0, y1=1,
                                   xref='x', yref='paper', line=dict(color=color, dash='dash', width=1.5))
                fig_hist.add_annotation(x=xval, y=0.97, xref='x', yref='paper',
                                        text=label, showarrow=False,
                                        font=dict(color=color, size=11), xanchor=xanchor, yanchor='top')
            fig_hist.update_layout(title="Distrubucion de los rendimientos simuladoc con Monte Carlo",
                                   xaxis_title="Retorno diario", yaxis_title="Frecuencia", height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
