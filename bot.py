import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import numpy as np
import time
import requests
from datetime import datetime
import dateutil.parser
import firebase_admin
from firebase_admin import credentials, messaging, firestore, storage
import warnings
import feedparser

warnings.filterwarnings("ignore", message="YF.download() has changed argument auto_adjust")

def safe_compare(value1, value2, comparison_type=">"):
    if value1 is None or value2 is None:
        return False
    if comparison_type == ">":
        return value1 > value2
    elif comparison_type == "<":
        return value1 < value2
    elif comparison_type == ">=":
        return value1 >= value2
    elif comparison_type == "<=":
        return value1 <= value2
    elif comparison_type == "==":
        return value1 == value2
    else:
        return False

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'pesanapp-12b90.appspot.com'
    })

db = firestore.client()
bucket = storage.bucket()

CRYPTO_PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT']
FOREX_PAIRS = {'XAU/USD': 'GC=F', 'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X', 'USD/JPY': 'JPY=X'}

MIN_WHALE_VALUE = 100000
MIN_CONFLUENCE_SCORE = 6 
MIN_SCORE_DIFFERENCE = 2 

last_processed_trade_id = 0

def advanced_volume_analysis(df):
    if len(df) < 20:
        return {
            'volume_spike': False,
            'volume_trend': "NEUTRAL",
            'volume_confirmation': False,
            'volume_ratio': 1.0
        }

    df['Volume_MA'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA']

    if pd.isna(df['Volume_Ratio'].iloc[-1]) or df['Volume_Ratio'].iloc[-1] == float('inf'):
        volume_ratio = 1.0
    else:
        volume_ratio = df['Volume_Ratio'].iloc[-1]

    volume_spike = volume_ratio > 1.5  

    if len(df) >= 2:
        volume_trend = "BULLISH" if df['volume'].iloc[-1] > df['volume'].iloc[-2] else "BEARISH"
        price_up = df['close'].iloc[-1] > df['close'].iloc[-2]
        volume_confirmation = (price_up and volume_trend == "BULLISH") or (not price_up and volume_trend == "BEARISH")
    else:
        volume_trend = "NEUTRAL"
        volume_confirmation = False
    return {
        'volume_spike': volume_spike,
        'volume_trend': volume_trend,
        'volume_confirmation': volume_confirmation,
        'volume_ratio': volume_ratio
    }
#market rigme
def detect_market_regime(df):
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0.01
    current_range = df['high'].tail(20).max() - df['low'].tail(20).min()
    avg_range = df['ATR'].tail(20).mean() * 20 if 'ATR' in df.columns else current_range

    if avg_range == 0: 
        return "NORMAL_VOLATILITY"
      
    range_ratio = current_range / avg_range

    if range_ratio > 1.5:
        return "HIGH_VOLATILITY"
    elif range_ratio < 0.7:
        return "LOW_VOLATILITY"
    else:
        return "NORMAL_VOLATILITY
      
# menghitung momentum
def calculate_momentum(df, periods=[5, 10, 20]):
    momentum_scores = {}

    for period in periods:
        if len(df) > period:
            momentum = (df['close'].iloc[-1] / df['close'].iloc[-period] - 1) * 100
            momentum_scores[f'MOM_{period}'] = momentum
        else:
            momentum_scores[f'MOM_{period}'] = 0
          
    if momentum_scores:
        avg_momentum = sum(momentum_scores.values()) / len(momentum_scores)
        momentum_bias = "BULLISH" if avg_momentum > 0.1 else "BEARISH" if avg_momentum < -0.1 else "NEUTRAL"
    else:
        momentum_bias = "NEUTRAL"
      
    return momentum_scores, momentum_bias

def optimize_tp_levels(entry, sl, market_regime):
    risk = abs(entry - sl)

    if market_regime == "HIGH_VOLATILITY":
        return {
            'tp1': entry + (risk * 2.0) if entry > sl else entry - (risk * 2.0),
            'tp2': entry + (risk * 3.5) if entry > sl else entry - (risk * 3.5),
            'tp3': entry + (risk * 5.0) if entry > sl else entry - (risk * 5.0)

        }

    elif market_regime == "LOW_VOLATILITY":
        return {
            'tp1': entry + (risk * 1.2) if entry > sl else entry - (risk * 1.2),
            'tp2': entry + (risk * 2.0) if entry > sl else entry - (risk * 2.0),
            'tp3': entry + (risk * 3.0) if entry > sl else entry - (risk * 3.0)
        }
    else:
        return {
            'tp1': entry + (risk * 1.5) if entry > sl else entry - (risk * 1.5),
            'tp2': entry + (risk * 2.5) if entry > sl else entry - (risk * 2.5),
            'tp3': entry + (risk * 4.0) if entry > sl else entry - (risk * 4.0)
        }

def should_enter_trade(bullish_score, bearish_score, market_regime, volume_analysis):
    if bullish_score >= 7 and bearish_score >= 7:
        print(f"    ⚠️ Filtered: Market too conflicting (Bull: {bullish_score}, Bear: {bearish_score})")
        return False
      
    if market_regime == "LOW_VOLATILITY" and volume_analysis['volume_ratio'] < 0.8:
        print(f"    ⚠️ Filtered: Low volatility + low volume ({volume_analysis['volume_ratio']:.2f}x)")
        return False
      
    score_diff = abs(bullish_score - bearish_score)
    if score_diff < 2:  
        print(f"    ⚠️ Filtered: Score difference too small ({score_diff})")
        return False

    if volume_analysis['volume_ratio'] < 1.0:
        print(f"    ⚠️ Filtered: Volume too low ({volume_analysis['volume_ratio']:.2f}x)")
        return False
      
    print(f"    ✅ Signal QUALIFIED (Bull: {bullish_score}, Bear: {bearish_score}, Diff: {score_diff})")
    return True

def detect_swing_points(df, window=5):
    df['swing_high'] = df['high'].rolling(window=window, center=True).max() == df['high']
    df['swing_low'] = df['low'].rolling(window=window, center=True).min() == df['low']
    return df

def detect_bos_choch(df):
    swings = df[df['swing_high'] | df['swing_low']].tail(5)
    if len(swings) < 3:
        return None, None

    last_high = swings[swings['swing_high']]['high'].iloc[-1] if any(swings['swing_high']) else None
    last_low = swings[swings['swing_low']]['low'].iloc[-1] if any(swings['swing_low']) else None
    current_price = df['close'].iloc[-1]

    bos = None
    choch = None

    if last_high and current_price > last_high:
        bos = "BULLISH"
      
    if last_low and current_price < last_low:
        bos = "BEARISH"
      
    highs = swings[swings['swing_high']]['high'].values
    lows = swings[swings['swing_low']]['low'].values

    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] < highs[-2] and lows[-1] > lows[-2]: 
            choch = "BEARISH"

        elif highs[-1] > highs[-2] and lows[-1] < lows[-2]: 
            choch = "BULLISH"
          
    return bos, choch
  
def detect_trend(df):
    swings = df[df['swing_high'] | df['swing_low']].tail(4)
    if len(swings) < 4:
        return "SIDEWAYS"

    highs = swings[swings['swing_high']]['high'].values
    lows = swings[swings['swing_low']]['low'].values

    if len(highs) >= 2 and len(lows) >= 2:
        higher_highs = highs[-1] > highs[-2] if len(highs) >= 2 else False
        higher_lows = lows[-1] > lows[-2] if len(lows) >= 2 else False
        lower_highs = highs[-1] < highs[-2] if len(highs) >= 2 else False
        lower_lows = lows[-1] < lows[-2] if len(lows) >= 2 else False

        if higher_highs and higher_lows:
            return "UPTREND"
        elif lower_highs and lower_lows:
            return "DOWNTREND"

    return "SIDEWAYS"

# SUPPORT & RESISTANCE - EXISTING
def find_support_resistance(df, num_clusters=3):
    swings = df[df['swing_high'] | df['swing_low']].tail(20)
    levels = pd.concat([swings['high'], swings['low']]).values

    levels = np.sort(levels)
    clusters = []

    for level in levels:
        if not clusters or abs(level - clusters[-1]) / clusters[-1] > 0.02:  # 2% threshold
            clusters.append(level)
          
    clusters = sorted(clusters)[-num_clusters:]
    current = df['close'].iloc[-1]
    support = [l for l in clusters if l < current]
    resistance = [l for l in clusters if l > current]

    return support[-1] if support else None, resistance[0] if resistance else None

# FAIR VALUE GAP (FVG) 
def detect_fair_value_gap(df):
    fvg_zones = []

    for i in range(2, len(df)):
        prev2 = df.iloc[i-2] 
        prev1 = df.iloc[i-1]  
        current = df.iloc[i]  

        if prev1['low'] > current['high']:
            fvg_zones.append({
                'type': 'BULLISH',
                'top': prev1['low'],
                'bottom': current['high'],
                'index': i
            })

        elif prev1['high'] < current['low']:
            fvg_zones.append({
                'type': 'BEARISH',
                'top': current['low'],
                'bottom': prev1['high'],
                'index': i
            })

    if fvg_zones:
        return fvg_zones[-1]
    return None

# MULTI-TIMEFRAME ANALYSIS 
def get_higher_timeframe_bias(exchange, pair, is_crypto=True):
    try:
        if is_crypto:
            bars_1h = exchange.fetch_ohlcv(pair, '1h', limit=100)
            if not bars_1h:
                return 'NEUTRAL'

            df_1h = pd.DataFrame(bars_1h, columns=['time', 'open', 'high', 'low', 'close', 'volume'])

            if df_1h.empty or df_1h['close'].isna().all():
                return 'NEUTRAL'

            bars_4h = exchange.fetch_ohlcv(pair, '4h', limit=50)
            if not bars_4h:
                return 'NEUTRAL'

            df_4h = pd.DataFrame(bars_4h, columns=['time', 'open', 'high', 'low', 'close', 'volume'])

            if df_4h.empty or df_4h['close'].isna().all():
                return 'NEUTRAL'
        else:
            return 'NEUTRAL'

        try:
            df_1h['EMA_50'] = ta.ema(df_1h['close'], length=50)
            df_1h['EMA_200'] = ta.ema(df_1h['close'], length=200)

            if df_1h['EMA_200'].iloc[-1] is None or df_1h['close'].iloc[-1] is None:
                bias_1h = 'NEUTRAL'
            else:
                bias_1h = 'BULLISH' if df_1h['close'].iloc[-1] > df_1h['EMA_200'].iloc[-1] else 'BEARISH'
        except:
            bias_1h = 'NEUTRAL'

        try:
            df_4h['EMA_50'] = ta.ema(df_4h['close'], length=50)
            df_4h['EMA_200'] = ta.ema(df_4h['close'], length=200)

            if df_4h['EMA_200'].iloc[-1] is None or df_4h['close'].iloc[-1] is None:
                bias_4h = 'NEUTRAL'
            else:
                bias_4h = 'BULLISH' if df_4h['close'].iloc[-1] > df_4h['EMA_200'].iloc[-1] else 'BEARISH'
        except:
            bias_4h = 'NEUTRAL'

        if bias_1h == bias_4h and bias_1h != 'NEUTRAL':
            return bias_1h
          
        return 'NEUTRAL'

    except Exception as e:
        print(f"MTF Error: {e}")
        return 'NEUTRAL'
      
# FIBONACCI RETRACEMENT 

def calculate_fibonacci(df, lookback=50):
    recent = df.tail(lookback)
    high = recent['high'].max()
    low = recent['low'].min()

    diff = high - low
    fib_levels = {
        '0.236': high - (diff * 0.236),
        '0.382': high - (diff * 0.382),
        '0.500': high - (diff * 0.500),
        '0.618': high - (diff * 0.618),
        '0.786': high - (diff * 0.786),
    }
    return fib_levels, high, low

# CANDLESTICK PATTERNS
def detect_candlestick_patterns(df):
    patterns = []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last['close'] - last['open'])
    prev_body = abs(prev['close'] - prev['open'])

    if prev['close'] < prev['open'] and last['close'] > last['open']:
        if last['open'] < prev['close'] and last['close'] > prev['open']:
            patterns.append("BULLISH_ENGULFING")

    if prev['close'] > prev['open'] and last['close'] < last['open']:
        if last['open'] > prev['close'] and last['close'] < prev['open']:
            patterns.append("BEARISH_ENGULFING")

    lower_shadow = last['open'] - last['low'] if last['close'] > last['open'] else last['close'] - last['low']
    if lower_shadow > body * 2 and last['close'] > last['open']:
        patterns.append("HAMMER")

    upper_shadow = last['high'] - last['open'] if last['close'] < last['open'] else last['high'] - last['close']
    if upper_shadow > body * 2 and last['close'] < last['open']:
        patterns.append("SHOOTING_STAR")

    return patterns
  
def analyze_market_ultimate(df, pair, type_asset, exchange=None):
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['EMA_200'] = ta.ema(df['close'], length=200)
    df['MA_20'] = ta.sma(df['close'], length=20)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    macd = ta.macd(df['close'])

    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']

    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df['BB_lower'] = bbands.iloc[:, 0]  # Lower
        df['BB_mid'] = bbands.iloc[:, 1]    # Mid
        df['BB_upper'] = bbands.iloc[:, 2]  # Upper
    else:
        df['BB_mid'] = df['MA_20']
        df['BB_upper'] = df['MA_20'] + (df['ATR'] * 2)
        df['BB_lower'] = df['MA_20'] - (df['ATR'] * 2)

    df['Volume_SMA'] = df['volume'].rolling(20).mean()

    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    if stoch is not None and not stoch.empty:
        df['STOCH_K'] = stoch.iloc[:, 0]  # %K
        df['STOCH_D'] = stoch.iloc[:, 1]  # %D
    else:
        df['STOCH_K'] = 50
        df['STOCH_D'] = 50

    df = detect_swing_points(df)
    bos, choch = detect_bos_choch(df)
    trend = detect_trend(df)
    support, resistance = find_support_resistance(df)
    fib_levels, swing_high, swing_low = calculate_fibonacci(df)
    patterns = detect_candlestick_patterns(df)
    fvg = detect_fair_value_gap(df)

    volume_analysis = advanced_volume_analysis(df)
    market_regime = detect_market_regime(df)
    momentum_scores, momentum_bias = calculate_momentum(df)

    mtf_bias = 'NEUTRAL'
    if type_asset == 'CRYPTO' and exchange:
        mtf_bias = get_higher_timeframe_bias(exchange, pair, is_crypto=True)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last['close']

    bullish_score = 0
    bullish_reasons = []

    if trend == "UPTREND" or bos == "BULLISH":
        bullish_score += 1
        bullish_reasons.append(f"✅ {trend if trend == 'UPTREND' else 'BOS Bullish'}")

    if safe_compare(price, last['EMA_200'], ">"):
        bullish_score += 1
        bullish_reasons.append("✅ Above EMA 200")

    if safe_compare(last['RSI'], 35, "<") or (safe_compare(prev['RSI'], 30, "<") and safe_compare(last['RSI'], 30, ">")):
        bullish_score += 1
        bullish_reasons.append(f"✅ RSI Oversold ({last['RSI']:.1f})")

    if safe_compare(last['MACD'], last['MACD_signal'], ">") and safe_compare(last['MACD_hist'], 0, ">"):
        bullish_score += 1
        bullish_reasons.append("✅ MACD Bullish")

    if last['volume'] > last['Volume_SMA'] * 1.2:
        bullish_score += 1
        bullish_reasons.append("✅ Volume Spike")

    in_fib_zone = False
    for level_name, level_value in fib_levels.items():
        if level_name in ['0.618', '0.786'] and abs(price - level_value) / price < 0.01:
            bullish_score += 1
            bullish_reasons.append(f"✅ Fib {level_name} Support")
            in_fib_zone = True
            break

    if support and abs(price - support) / price < 0.015:
        bullish_score += 1
        bullish_reasons.append("✅ Near Support Zone")

    if safe_compare(price, last['BB_lower'], "<"):
        bullish_score += 1
        bullish_reasons.append("✅ BB Oversold")
      
    bullish_patterns = [p for p in patterns if 'BULLISH' in p or p == 'HAMMER']
    if bullish_patterns:
        bullish_score += 1
        bullish_reasons.append(f"✅ {bullish_patterns[0]}")

    if choch == "BULLISH":
        bullish_score += 1
        bullish_reasons.append("✅ CHoCH Reversal")

    if safe_compare(last['STOCH_K'], 20, "<") and safe_compare(last['STOCH_K'], prev['STOCH_K'], ">"):
        bullish_score += 1
        bullish_reasons.append(f"✅ Stoch Oversold ({last['STOCH_K']:.1f})")

    if fvg and fvg['type'] == 'BULLISH' and fvg['bottom'] <= price <= fvg['top']:
        bullish_score += 1
        bullish_reasons.append("✅ Bullish FVG Zone")

    if volume_analysis['volume_confirmation'] and volume_analysis['volume_ratio'] > 1.5:
        bullish_score += 1
        bullish_reasons.append("✅ Volume Confirmation")

    if momentum_bias == "BULLISH":
        bullish_score += 1
        bullish_reasons.append("✅ Momentum Bullish")

    if market_regime in ["HIGH_VOLATILITY", "NORMAL_VOLATILITY"]:
        bullish_score += 1
        bullish_reasons.append(f"✅ Regime: {market_regime}")

    if mtf_bias == 'BULLISH':
        bullish_score += 1
        bullish_reasons.append("✅ MTF Bullish (1H+4H)")

    bearish_score = 0
    bearish_reasons = []

    if trend == "DOWNTREND" or bos == "BEARISH":
        bearish_score += 1
        bearish_reasons.append(f"✅ {trend if trend == 'DOWNTREND' else 'BOS Bearish'}")

    if safe_compare(price, last['EMA_200'], "<"):
        bearish_score += 1
        bearish_reasons.append("✅ Below EMA 200")

    if safe_compare(last['RSI'], 65, ">") or (safe_compare(prev['RSI'], 70, ">") and safe_compare(last['RSI'], 70, "<")):
        bearish_score += 1
        bearish_reasons.append(f"✅ RSI Overbought ({last['RSI']:.1f})")

    if safe_compare(last['MACD'], last['MACD_signal'], "<") and safe_compare(last['MACD_hist'], 0, "<"):
        bearish_score += 1
        bearish_reasons.append("✅ MACD Bearish")

    if last['volume'] > last['Volume_SMA'] * 1.2:
        bearish_score += 1
        bearish_reasons.append("✅ Volume Spike")

    if not in_fib_zone:
        for level_name, level_value in fib_levels.items():
            if level_name in ['0.236', '0.382'] and abs(price - level_value) / price < 0.01:
                bearish_score += 1
                bearish_reasons.append(f"✅ Fib {level_name} Resistance")
                break
              
    if resistance and abs(price - resistance) / price < 0.015:
        bearish_score += 1
        bearish_reasons.append("✅ Near Resistance Zone")

    if safe_compare(price, last['BB_upper'], ">"):
        bearish_score += 1
        bearish_reasons.append("✅ BB Overbought")

    bearish_patterns = [p for p in patterns if 'BEARISH' in p or p == 'SHOOTING_STAR']
    if bearish_patterns:
        bearish_score += 1
        bearish_reasons.append(f"✅ {bearish_patterns[0]}")

    if choch == "BEARISH":
        bearish_score += 1
        bearish_reasons.append("✅ CHoCH Reversal")

    if safe_compare(last['STOCH_K'], 80, ">") and safe_compare(last['STOCH_K'], prev['STOCH_K'], "<"):
        bearish_score += 1
        bearish_reasons.append(f"✅ Stoch Overbought ({last['STOCH_K']:.1f})")

    if fvg and fvg['type'] == 'BEARISH' and fvg['bottom'] <= price <= fvg['top']:
        bearish_score += 1
        bearish_reasons.append("✅ Bearish FVG Zone")

    if volume_analysis['volume_confirmation'] and volume_analysis['volume_ratio'] > 1.5:
        bearish_score += 1
        bearish_reasons.append("✅ Volume Confirmation")

    if momentum_bias == "BEARISH":
        bearish_score += 1
        bearish_reasons.append("✅ Momentum Bearish")

    if market_regime in ["HIGH_VOLATILITY", "NORMAL_VOLATILITY"]:
        bearish_score += 1
        bearish_reasons.append(f"✅ Regime: {market_regime}")

    if mtf_bias == 'BEARISH':
        bearish_score += 1
        bearish_reasons.append("✅ MTF Bearish (1H+4H)")

    if bullish_score >= MIN_CONFLUENCE_SCORE:
        if not should_enter_trade(bullish_score, bearish_score, market_regime, volume_analysis):
            print(f"⏸️  Signal {pair} BULLISH filtered out (conflicting/weak)")
            return
          
        atr = last['ATR']
        entry = price

        if support:
            sl = support - (atr * 0.5)
        else:
            sl = price - (atr * 1.5)

        optimized_tp = optimize_tp_levels(entry, sl, market_regime)

        bullish_reasons.append(f"📊 Confluence: {bullish_score}/15")
        bullish_reasons.append(f"🎯 Market Regime: {market_regime}")
        bullish_reasons.append(f"📈 Volume Ratio: {volume_analysis['volume_ratio']:.2f}x")

        manage_signal(type_asset, pair, "LONG", entry,
                    optimized_tp['tp1'], optimized_tp['tp2'], optimized_tp['tp3'],
                    sl, "\n".join(bullish_reasons), df)

    elif bearish_score >= MIN_CONFLUENCE_SCORE:
        if not should_enter_trade(bullish_score, bearish_score, market_regime, volume_analysis):
            print(f"⏸️  Signal {pair} BEARISH filtered out (conflicting/weak)")
            return

        atr = last['ATR']
        entry = price

        if resistance:
            sl = resistance + (atr * 0.5)
        else:
            sl = price + (atr * 1.5)

        optimized_tp = optimize_tp_levels(entry, sl, market_regime)

        position_size = calculate_position_size(ACCOUNT_BALANCE, entry, sl, RISK_PER_TRADE)

        bearish_reasons.append(f"📊 Confluence: {bearish_score}/15")
        bearish_reasons.append(f"💰 Position Size: {position_size:.4f}")
        bearish_reasons.append(f"🎯 Market Regime: {market_regime}")
        bearish_reasons.append(f"📈 Volume Ratio: {volume_analysis['volume_ratio']:.2f}x")

        manage_signal(type_asset, pair, "SHORT", entry,
                      optimized_tp['tp1'], optimized_tp['tp2'], optimized_tp['tp3'],
                      sl, "\n".join(bearish_reasons), df, position_size)

def manage_signal(type_asset, pair, action, entry, tp1, tp2, tp3, sl, reason, df, position_size=0):
    existing = db.collection('signals').where('pair', '==', pair).where('status', '==', 'ACTIVE').stream()

    for doc in existing:
        old_data = doc.to_dict()
        old_action = old_data.get('action')
        doc_id = doc.id

        if old_action == action:
            print(f"    ⏸️ Skipped: {pair} already has ACTIVE {action} signal.")
            return

        else:
            print(f"    🔄 REVERSAL DETECTED: {pair} flipping from {old_action} to {action}!")

            db.collection('signals').document(doc_id).update({
                'status': 'INVALID', # Tandai Invalid/Closed
                'invalid_reason': f'Market Reversal: Strong {action} detected!',
                'invalid_time': datetime.now()

            })

            try:
                msg = messaging.Message(
                    notification=messaging.Notification(
                        title=f"⚠️ CLOSE {pair} NOW!",
                        body=f"Reversal Detected! Preparing {action} signal..."
                    ),
                    topic='trading_signals',
                )
                messaging.send(msg)
            except:
                pass
              
    print(f"🎯 NEW SIGNAL: {pair} ({action}) | Score: {reason.split('Confluence: ')[1].split('/')[0]}/15")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{pair.replace('/','')}_{timestamp_str}"
    local_img = generate_chart_image(df.tail(60), pair, file_name)

    img_url = "https://via.placeholder.com/400"
    if local_img:
        img_url = upload_image(local_img, f"{file_name}.png")

    signal_data = {
        'type': type_asset,
        'pair': pair,
        'action': action,
        'entry': round(float(entry), 6),
        'tp1': round(float(tp1), 6),
        'tp2': round(float(tp2), 6),
        'tp3': round(float(tp3), 6),
        'sl': round(float(sl), 6),
        'position_size': round(float(position_size), 6),
        'reason': reason,
        'image_url': img_url,
        'timestamp': datetime.now(),
        'status': 'ACTIVE',
        'hit_price': 0.0,
        'hit_time': None,
    }

    doc_ref = db.collection('signals').document()
    doc_ref.set(signal_data)

    try:
        short_reason = reason.split('\n')[0] if '\n' in reason else reason[:50] + "..."
        msg = messaging.Message(
            notification=messaging.Notification(
                title=f"🎯 {action} SIGNAL: {pair}",
                body=f"Entry: {entry:.4f} | Score: {reason.split('Confluence: ')[1].split('/')[0]}/15"
            ),
            data={
                'type': 'new_signal',
                'pair': pair,
                'action': action,
                'screen': 'signal_page',
                'signal_id': doc_ref.id
            },
            topic='trading_signals',
        )
        response = messaging.send(msg)
        print(f"✅ Notification Sent! ID: {response}")
    except Exception as e:
        print(f"❌ Notif Failed: {e}")

def generate_chart_image(df, pair, filename):
    try:
        s = mpf.make_mpf_style(base_mpf_style='nightclouds', rc={'font.size': 8})

        apds = [
            mpf.make_addplot(df['EMA_50'], color='cyan', width=1.5),
            mpf.make_addplot(df['EMA_200'], color='orange', width=2),
            mpf.make_addplot(df['BB_upper'], color='gray', width=0.8, linestyle='--'),
            mpf.make_addplot(df['BB_lower'], color='gray', width=0.8, linestyle='--'),
        ]

        chart_path = f"{filename}.png"
        mpf.plot(df, type='candle', style=s, addplot=apds, volume=True,
                 savefig=dict(fname=chart_path, dpi=100, bbox_inches='tight'),
                 title=f"\n{pair} - Multi Indicator Analysis", num_panels=2, panel_ratios=(6,2))
        return chart_path
    except Exception as e:
        print(f"Chart Error: {e}")
        return None
      
def upload_image(local_path, remote_name):
    try:
        blob = bucket.blob(f"charts/{remote_name}")
        blob.upload_from_filename(local_path)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"Upload Error: {e}")
        return None

def calculate_performance_stats():
    print("    [STATS] Calculating performance metrics...")
    try:
        finished_signals = db.collection('signals').where(filter=firestore.FieldFilter('status', 'in', ['HIT_TP1', 'HIT_TP2', 'HIT_TP3', 'HIT_SL'])).stream()

        total_signals = 0
        wins = 0
        total_profit_pct = 0.0

        for sig_doc in finished_signals:
            data = sig_doc.to_dict()
            total_signals += 1
            status = data.get('status')
            action = data.get('action')
            entry = float(data.get('entry', 0))
            hit_price = float(data.get('hit_price', 0))

            if 'HIT_TP' in status:
                wins += 1

            if entry > 0 and hit_price > 0:
                if action == 'LONG':
                    profit = ((hit_price - entry) / entry) * 100
                else: # SHORT
                    profit = ((entry - hit_price) / entry) * 100

                total_profit_pct += profit

        win_rate = round((wins / total_signals * 100), 1) if total_signals > 0 else 0
        avg_profit = round((total_profit_pct / total_signals), 2) if total_signals > 0 else 0

        stats_data = {
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_signals': total_signals,
            'last_updated': datetime.now()
        }

        db.collection('performance').document('stats').set(stats_data)
        print(f"✅ [STATS] Updated: WR={win_rate}%, Avg={avg_profit}%, Total={total_signals}")

    except Exception as e:
        print(f"❌ Stats Calculation Error: {e}")

def update_news_ticker():
    print("    [NEWS] Fetching latest crypto news...")
    try:
        feed_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        feed = feedparser.parse(feed_url)

        headlines = []

        for entry in feed.entries[:5]:
            title = entry.title.replace("  ", " ")
            headlines.append(f"📰 {title}")

        full_ticker = "   |   ".join(headlines)

        db.collection('market_summary').document('news').set({
            'ticker': full_ticker,
            'timestamp': datetime.now(),
            'source': 'CoinDesk'
        })

        print("✅ [NEWS] Ticker updated!")
    except Exception as e:
        print(f"❌ News Error: {e}")

def update_crypto_bubbles():
    print("    [BUBBLES] Fetching Top Coins data...")
    try:
        exchange = ccxt.binance()
        tickers = exchange.fetch_tickers(['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT'])
        bubbles_data = []

        for symbol, data in tickers.items():
            name = symbol.split('/')[0]
            price = data['last']
            change = data['percentage'] # Perubahan 24h

            bubbles_data.append({
                'symbol': name,
                'price': price,
                'change': change,
                'size': abs(change) + 1 
            })

        db.collection('market_summary').document('bubbles').set({
            'coins': bubbles_data,
            'timestamp': datetime.now()
        })
        print(f"✅ [BUBBLES] Updated {len(bubbles_data)} coins!")
    except Exception as e:
        print(f"❌ Bubbles Error: {e}")

def run_signal_scanner():
    exchange = ccxt.binance()
  
    for pair in CRYPTO_PAIRS:
        try:
            bars = exchange.fetch_ohlcv(pair, '15m', limit=300)
            if not bars or len(bars) < 50:  # Validasi data cukup
                print(f"⚠️ Data tidak cukup untuk {pair}")
                continue

            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)

            if df.empty or df['close'].isna().all():
                print(f"⚠️ Data kosong untuk {pair}")
                continue

            analyze_market_ultimate(df, pair, 'CRYPTO', exchange)
        except Exception as e:
            print(f"Error {pair}: {e}")

    for pair_name, symbol in FOREX_PAIRS.items():
        try:
            df = yf.download(symbol, interval='15m', period='5d', progress=False, auto_adjust=True, multi_level_index=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            analyze_market_ultimate(df, pair_name, 'FOREX', None)
        except Exception as e:
            print(f"Error {pair_name}: {e}")
          
def check_whales():

    global last_processed_trade_id

    exchange = ccxt.binance()



    try:

        trades = exchange.fetch_trades('BTC/USDT', limit=50)

        for trade in trades:

            if int(trade['id']) <= last_processed_trade_id:

                continue



            if trade['cost'] >= MIN_WHALE_VALUE:

                print(f"🐋 WHALE: {trade['side']} ${trade['cost']:,.0f}")



                # FORMAT YANG LEBIH BAIK UNTUK FLUTTER

                whale_data = {

                    'symbol': 'BTC',

                    'amount': round(float(trade['amount']), 6),

                    'usd': round(float(trade['cost']), 2),

                    'from': 'Binance',

                    'to': f"Market {trade['side']}",

                    'timestamp': datetime.now(),

                    'hash': str(trade['id']),

                    'side': trade['side'],  # Tambah field side untuk color coding

                    'price': round(float(trade['price']), 2),  # Harga execution

                }



                db.collection('whales').document(str(trade['id'])).set(whale_data)



        if trades:

            last_processed_trade_id = int(trades[-1]['id'])

    except Exception as e:

        print(f"Whale Error: {e}")



# ==========================================

# ECONOMIC CALENDAR

# ==========================================

def check_calendar():

    print("    [EVENT] Updating Calendar...")

    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"



    try:

        data = requests.get(url).json()

        today_str = datetime.now().strftime("%Y-%m-%d")



        count = 0

        for event in data:

            title = event['title']

            country = event['country']

            date_full = event['date']

            impact = event['impact']

            forecast = event.get('forecast', '')

            actual = event.get('actual', '')



            if today_str not in date_full:

                continue

            if impact == 'Low':

                continue



            event_time = dateutil.parser.parse(date_full)

            time_str = event_time.strftime("%H:%M")



            insight = "Menunggu Rilis Data..."

            status = "UPCOMING"



            if actual != '':

                status = "RELEASED"

                try:

                    act_val = float(actual.replace('%','').replace('K','').replace('M',''))

                    fc_val = float(forecast.replace('%','').replace('K','').replace('M',''))



                    is_unemployment = 'Unemployment' in title

                    is_good_for_currency = False



                    if is_unemployment:

                        if act_val < fc_val:

                            is_good_for_currency = True

                    else:

                        if act_val > fc_val:

                            is_good_for_currency = True



                    if country == 'USD':

                        if is_good_for_currency:

                            insight = "🔥 USD MENGUAT! \n📉 Potensi: GOLD (XAU), EURUSD, BTC cenderung TURUN."

                        else:

                            insight = "🩸 USD MELEMAH! \n🚀 Potensi: GOLD (XAU), EURUSD, BTC cenderung NAIK."

                    else:

                        if is_good_for_currency:

                            insight = f"Positif untuk {country} 🔼"

                        else:

                            insight = f"Negatif untuk {country} 🔻"

                except:

                    insight = f"Rilis Data: {actual}"



            safe_title = "".join(c for c in title if c.isalnum() or c in (' ','_')).rstrip()

            if not safe_title:

                safe_title = "Event"

            doc_id = f"{today_str}-{safe_title.replace(' ','_')}"



            # ENHANCE EVENT DATA UNTUK FLUTTER

            event_data = {

                'title': title,

                'country': country,

                'impact': impact,

                'time': time_str,

                'forecast': forecast,

                'actual': actual,

                'insight': insight,

                'timestamp': event_time,

                'status': status,

                'date': today_str,  # Tambah field date untuk filtering

                'is_high_impact': impact == 'High',  # Untuk easy filtering

            }



            db.collection('events').document(doc_id).set(event_data)

            count += 1



        print(f"    [EVENT] {count} Events Updated.")

    except Exception as e:

        print(f"Calendar Error: {e}")



# ==========================================

# SIGNAL MONITORING & AUTO-UPDATE

# ==========================================

def get_current_price(pair, is_crypto=True):

    """Ambil harga real-time (Fixed for YFinance Update)"""

    try:

        if is_crypto:

            exchange = ccxt.binance()

            ticker = exchange.fetch_ticker(pair)

            return float(ticker['last']) # Pastikan float

        else:

            # Untuk Forex (manual mapping)

            forex_map = {'XAU/USD': 'GC=F', 'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X', 'USD/JPY': 'JPY=X'}

            symbol = forex_map.get(pair)

            if symbol:

                # Tambah auto_adjust=True biar warning hilang

                df = yf.download(symbol, period='1d', interval='1m', progress=False, auto_adjust=True)



                if df.empty:

                    return None



                # AMBIL HARGA TERAKHIR

                # Masalahnya disini: yfinance kadang balikin DataFrame, kadang Series

                last_price = df['Close'].iloc[-1]



                # FIX: Konversi paksa ke float scalar

                try:

                    return float(last_price.iloc[0])

                except:

                    # Kalau masih error (misal bentuknya Series), ambil item pertamanya

                    return float(last_price.iloc[0])



        return None

    except Exception as e:

        print(f"Price Error {pair}: {e}")

        return None



def monitor_active_signals():

    """

    Monitor semua sinyal ACTIVE:

    1. Cek apakah TP/SL sudah tersentuh

    2. Update status dan kirim notifikasi

    3. Deteksi reversal (invalidasi sinyal)

    4. Auto-close sinyal yang selesai

    """

    try:

        signals = db.collection('signals').where('status', 'in', ['ACTIVE', 'HIT_TP1', 'HIT_TP2']).stream()



        for sig in signals:

            data = sig.to_dict()

            pair = data['pair']

            action = data['action']

            type_asset = data['type']



            # Ambil harga sekarang

            is_crypto = (type_asset == 'CRYPTO')

            current_price = get_current_price(pair, is_crypto)



            if current_price is None:

                continue



            # Parse levels

            entry = float(data['entry'])

            tp1 = float(data['tp1'])

            tp2 = float(data['tp2'])

            tp3 = float(data['tp3'])

            sl = float(data['sl'])

            status = data['status']



            # Timestamp signal dibuat

            signal_time = data['timestamp']

            if hasattr(signal_time, 'strftime'):

                signal_time_str = signal_time.strftime("%d/%m/%Y %H:%M")

            else:

                signal_time_str = "Unknown"



            # ==========================================

            # CEK LONG POSITION

            # ==========================================

            if action == "LONG":

                # Hit Stop Loss

                if current_price <= sl:

                    update_signal_status(sig.id, 'HIT_SL', current_price, signal_time_str)

                    send_tp_sl_notification(pair, 'STOP LOSS', entry, current_price, signal_time_str)

                    print(f"❌ {pair} HIT SL: {current_price:.4f}")



                # Hit TP3 (Semua target tercapai)

                elif current_price >= tp3 and status != 'HIT_TP3':

                    update_signal_status(sig.id, 'HIT_TP3', current_price, signal_time_str)

                    send_tp_sl_notification(pair, 'TARGET 3 (ALL DONE!)', entry, current_price, signal_time_str)

                    print(f"✅✅✅ {pair} HIT TP3: {current_price:.4f}")



                # Hit TP2

                elif current_price >= tp2 and status == 'HIT_TP1':

                    update_signal_status(sig.id, 'HIT_TP2', current_price, signal_time_str)

                    send_tp_sl_notification(pair, 'TARGET 2', entry, current_price, signal_time_str)

                    print(f"✅✅ {pair} HIT TP2: {current_price:.4f}")



                # Hit TP1

                elif current_price >= tp1 and status == 'ACTIVE':

                    update_signal_status(sig.id, 'HIT_TP1', current_price, signal_time_str)

                    send_tp_sl_notification(pair, 'TARGET 1', entry, current_price, signal_time_str)

                    print(f"✅ {pair} HIT TP1: {current_price:.4f}")



                # Deteksi Reversal (Price turun terlalu dalam)

                elif status == 'ACTIVE' and current_price < entry * 0.98:  # Turun 2% dari entry

                    check_reversal(sig.id, pair, action, is_crypto, signal_time_str)



            # ==========================================

            # CEK SHORT POSITION

            # ==========================================

            elif action == "SHORT":

                # Hit Stop Loss

                if current_price >= sl:

                    update_signal_status(sig.id, 'HIT_SL', current_price, signal_time_str)

                    send_tp_sl_notification(pair, 'STOP LOSS', entry, current_price, signal_time_str)

                    print(f"❌ {pair} HIT SL: {current_price:.4f}")



                # Hit TP3

                elif current_price <= tp3 and status != 'HIT_TP3':

                    update_signal_status(sig.id, 'HIT_TP3', current_price, signal_time_str)

                    send_tp_sl_notification(pair, 'TARGET 3 (ALL DONE!)', entry, current_price, signal_time_str)

                    print(f"✅✅✅ {pair} HIT TP3: {current_price:.4f}")



                # Hit TP2

                elif current_price <= tp2 and status == 'HIT_TP1':

                    update_signal_status(sig.id, 'HIT_TP2', current_price, signal_time_str)

                    send_tp_sl_notification(pair, 'TARGET 2', entry, current_price, signal_time_str)

                    print(f"✅✅ {pair} HIT TP2: {current_price:.4f}")



                # Hit TP1

                elif current_price <= tp1 and status == 'ACTIVE':

                    update_signal_status(sig.id, 'HIT_TP1', current_price, signal_time_str)

                    send_tp_sl_notification(pair, 'TARGET 1', entry, current_price, signal_time_str)

                    print(f"✅ {pair} HIT TP1: {current_price:.4f}")



                # Deteksi Reversal

                elif status == 'ACTIVE' and current_price > entry * 1.02:  # Naik 2% dari entry

                    check_reversal(sig.id, pair, action, is_crypto, signal_time_str)



        # Auto-delete sinyal yang sudah selesai (HIT_TP3 atau HIT_SL) > 1 jam

        cleanup_old_signals()



    except Exception as e:

        print(f"Monitor Error: {e}")



def update_signal_status(doc_id, new_status, hit_price, signal_time):

    """Update status sinyal di Firestore"""

    try:

        db.collection('signals').document(doc_id).update({

            'status': new_status,

            'hit_price': str(hit_price),

            'hit_time': datetime.now(),

            'signal_created': signal_time

        })

    except Exception as e:

        print(f"Update Error: {e}")



def send_tp_sl_notification(pair, event, entry, current, signal_time):

    """Kirim notifikasi kalau TP/SL tersentuh"""

    try:

        profit_pct = ((current - entry) / entry) * 100



        if event == 'STOP LOSS':

            title = f"❌ {pair} - Stop Loss Hit"

            body = f"Entry: {entry:.4f} | Loss: {profit_pct:.2f}%\nSignal: {signal_time}"

        else:

            title = f"✅ {pair} - {event} Hit!"

            body = f"Entry: {entry:.4f} | Profit: {profit_pct:.2f}%\nSignal: {signal_time}"



        msg = messaging.Message(

            notification=messaging.Notification(title=title, body=body),

            data={'type': 'tp_sl_hit', 'pair': pair},

            topic='trading_signals',

        )

        messaging.send(msg)

    except Exception as e:

        print(f"Notif Error: {e}")



def check_reversal(doc_id, pair, original_action, is_crypto, signal_time):

    """

    Deteksi reversal: Kalau market berbalik arah drastis

    Contoh: Signal LONG, tapi market mulai DOWNTREND kuat

    """

    try:

        # Fetch data terbaru

        if is_crypto:

            exchange = ccxt.binance()

            bars = exchange.fetch_ohlcv(pair, '15m', limit=100)

            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])

        else:

            return  # Skip untuk forex



        # Hitung indikator cepat

        df['EMA_50'] = ta.ema(df['close'], length=50)

        df['EMA_200'] = ta.ema(df['close'], length=200)

        df['RSI'] = ta.rsi(df['close'], length=14)



        last = df.iloc[-1]

        price = last['close']



        # Cek apakah ada konflik kuat

        if original_action == "LONG":

            # Kalau price turun dibawah EMA 200 + RSI > 60 = Reversal

            if price < last['EMA_200'] and last['RSI'] > 60:

                invalidate_signal(doc_id, pair, "Bearish Reversal Detected", signal_time)



        elif original_action == "SHORT":

            # Kalau price naik diatas EMA 200 + RSI < 40 = Reversal

            if price > last['EMA_200'] and last['RSI'] < 40:

                invalidate_signal(doc_id, pair, "Bullish Reversal Detected", signal_time)



    except Exception as e:

        print(f"Reversal Check Error: {e}")



def invalidate_signal(doc_id, pair, reason, signal_time):

    """Invalidasi sinyal dan kirim notifikasi"""

    try:

        db.collection('signals').document(doc_id).update({

            'status': 'INVALID',

            'invalid_reason': reason,

            'invalid_time': datetime.now()

        })



        msg = messaging.Message(

            notification=messaging.Notification(

                title=f"⚠️ {pair} - Signal Invalid",

                body=f"{reason}\nScanning for new signal...\nSignal: {signal_time}"

            ),

            data={'type': 'signal_invalid'},

            topic='trading_signals',

        )

        messaging.send(msg)

        print(f"⚠️ {pair} INVALIDATED: {reason}")

    except Exception as e:

        print(f"Invalidate Error: {e}")



def cleanup_old_signals():

    """Hapus sinyal yang sudah selesai > 1 jam"""

    try:

        one_hour_ago = datetime.now() - pd.Timedelta(hours=1)



        old_signals = db.collection('signals').where('status', 'in', ['HIT_TP3', 'HIT_SL', 'INVALID']).stream()



        for sig in old_signals:

            data = sig.to_dict()

            hit_time = data.get('hit_time')



            if hit_time and hit_time < one_hour_ago:

                db.collection('signals').document(sig.id).delete()

                print(f"🗑️ Cleaned up old signal: {data['pair']}")



    except Exception as e:

        print(f"Cleanup Error: {e}")



# ==========================================

# NEW: SERVER HEALTH MONITOR

# ==========================================

def update_server_status():

    """Update server status di Firebase"""

    try:

        status_data = {

            'status': 'ONLINE',

            'last_update': datetime.now(),

            'version': 'V3.0',

            'signals_today': 0,  # Bisa dihitung dari Firestore

            'performance': 'OPTIMAL'

        }



        db.collection('server_status').document('trading_bot').set(status_data)

        print("✅ Server Status Updated")



    except Exception as e:

        print(f"❌ Server Status Error: {e}")



# ==========================================

# SMART MARKET SUMMARY

# ==========================================

def generate_market_summary():

    """Generate market summary + Fear & Greed + Market Mood"""

    try:

        exchange = ccxt.binance()



        # 1. AMBIL DATA BTC

        bars = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=2) # Cukup 2 candle terakhir

        if not bars: return



        # Hitung perubahan harga

        close_now = bars[-1][4]

        open_today = bars[-1][1]

        daily_change_pct = ((close_now - open_today) / open_today) * 100



        # 2. TENTUKAN MOOD CRYPTO (Otomatis)

        mood_crypto = "SIDEWAYS 💤"

        mood_color = "YELLOW" # Buat referensi warna di Flutter nanti



        if daily_change_pct > 1.5:

            mood_crypto = "BULLISH 🐂"

        elif daily_change_pct < -1.5:

            mood_crypto = "BEARISH 🐻"



        # 3. TENTUKAN VOLATILITAS (Otomatis)

        mood_volatility = "NORMAL 💧"

        if abs(daily_change_pct) > 4:

            mood_volatility = "EXTREME 🌊"

        elif abs(daily_change_pct) > 2.5:

            mood_volatility = "HIGH 🔥"

        elif abs(daily_change_pct) < 0.5:

            mood_volatility = "LOW ❄️"



        # 4. AMBIL DATA LAIN (Gold & F&G)

        gold_price = get_current_price('XAU/USD', False)



        # Tentukan Mood Gold/Forex (Simple Logic: Gold Naik = Dolar Lemah)

        mood_gold = "NEUTRAL ➖"

        if gold_price:

            if gold_price > 2000: mood_gold = "UPTREND 📈"

            else: mood_gold = "DOWNTREND 📉"



        # API Fear & Greed

        fng_value = 50

        fng_text = "Neutral"

        try:

            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=3).json()

            fng_value = int(r['data'][0]['value'])

            fng_text = r['data'][0]['value_classification']

        except: pass



        # 5. RACIK TEXT SUMMARY (Sama kayak tadi)

        summary = f"💰 **BTC**: ${close_now:,.0f} ({daily_change_pct:+.2f}%)\n"

        if daily_change_pct > 0: summary += "📈 Market menghijau hari ini.\n"

        else: summary += "📉 Market sedang koreksi merah.\n"



        if fng_value > 75: summary += f"⚠️ Hati-hati Extreme Greed ({fng_value}).\n"

        elif fng_value < 25: summary += f"✅ Waktunya serok di Extreme Fear ({fng_value}).\n"



        # 6. SIMPAN SEMUA KE FIREBASE

        db.collection('market_summary').document('latest').set({

            'summary': summary,

            'timestamp': datetime.now(),

            'btc_price': close_now,

            'gold_price': gold_price,

            'fng_value': fng_value,

            'fng_text': fng_text,



            # --- DATA MOOD BARU ---

            'mood_crypto': mood_crypto,

            'mood_gold': mood_gold,

            'mood_volatility': mood_volatility

        })



        print(f"✅ [MARKET] Updated: {mood_crypto} | Vol: {mood_volatility}")



    except Exception as e:

        print(f"❌ Market Error: {e}")



# ==========================================

# ENHANCE: ECONOMIC CALENDAR WITH SIMPLE INSIGHT

# ==========================================

def enhance_economic_insight(event_data):

    """Tambahkan insight sederhana untuk user"""

    title = event_data['title']

    country = event_data['country']

    impact = event_data['impact']



    # Simple rules untuk non-expert

    insight_rules = {

        'Non-Farm Payrolls': "📊 **NFP**: >200K = USD 🚀 | <100K = USD 📉",

        'CPI': "💰 **CPI**: >3% = Rate Hike Fear | <2% = Rate Cut Hope",

        'Interest Rate': "🏦 **Rates**: Hike = USD 📈 | Cut = USD 📉",

        'Unemployment': "👥 **Jobs**: Low = Economy Strong | High = Economy Weak",

        'GDP': "📈 **GDP**: Strong = Bullish | Weak = Bearish"

    }



    # Cari insight yang relevan

    for key, insight in insight_rules.items():

        if key in title:

            return f"{insight}\n\n📉 Potensi: {country} pairs volatility tinggi!"



    return "📊 Data ekonomi penting - watch for market moves!"



# ==========================================

# MAIN LOOP - UPDATE VERSION INFO

# ==========================================

print("--- 🚀 ALL SYSTEMS ONLINE (V3.0) ---")

print("🔥 FEATURES: SMC + MULTI INDICATOR + ADVANCE FILTER + SMART POSITION SIZING")



try:

    tmpex = ccxt.binance()

    trades = tmpex.fetch_trades('BTC/USDT', limit=1)

    last_processed_trade_id = int(trades[0]['id'])

except:

    pass



last_scan_signal = 0

last_scan_calendar = 0

last_status_update = 0

last_market_update = 0

last_stats_update = 0

last_monitor_update = 0



print("⚡ [STARTUP] Forcing Market & Stats Update...")

try:

    generate_market_summary()       # Paksa update Market & Fear Greed

    calculate_performance_stats()   # Paksa update Win Rate

except Exception as e:

    print(f"Startup Update Error: {e}")



while True:

    try:

        check_whales()



        # Cek setiap 10 detik biar real-time

        if time.time() - last_monitor_update > 10:

            monitor_active_signals()

            last_monitor_update = time.time()



        # SCAN SIGNAL (Tiap 60 detik)

        if time.time() - last_scan_signal > 60:

            print(f"\n[{datetime.now().strftime('%H:%M')}] 🔍 Scanning Signals (Multi-Confluence V3.0)...")

            run_signal_scanner()

            last_scan_signal = time.time()



        # CEK KALENDER (Tiap 5 menit)

        if time.time() - last_scan_calendar > 300:

            check_calendar()

            last_scan_calendar = time.time()



        # UPDATE SERVER STATUS (Tiap 30 detik)

        if time.time() - last_status_update > 30:

            update_server_status()

            last_status_update = time.time()



        # UPDATE MARKET & FEAR GREED (Tiap 60 detik)

        if time.time() - last_market_update > 60:

            print("    [MARKET] Updating Market Summary & F&G...")

            update_news_ticker()

            update_crypto_bubbles()

            generate_market_summary()

            last_market_update = time.time()



        # UPDATE STATISTIK WIN RATE (Tiap 5 menit)

        if time.time() - last_stats_update > 300:

            calculate_performance_stats()

            last_stats_update = time.time()



        time.sleep(10) # Istirahat 10 detik biar gak spam CPU



    except KeyboardInterrupt:

        print("\n👋 Bot Stopped by User")

        break

    except Exception as e:

        print(f"❌ Main Loop Error: {e}")

        time.sleep(10)
