import pandas as pd
import yfinance as yf
import requests
import os
import time

# ==============================================================================
# 🇹🇼 台股全市場技術面模組
# ==============================================================================
DYNAMIC_STOCK_NAMES = {}

def fetch_all_taiwan_market_tickers():
    """ 下載全台股市場代碼 """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_tickers = []
    
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = requests.get(url_twse, headers=headers, timeout=10)
        if res.status_code == 200:
            for item in res.json():
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()
                if code.isdigit() and len(code) == 4:
                    ticker_id = f"{code}.TW"
                    all_tickers.append(ticker_id)
                    DYNAMIC_STOCK_NAMES[ticker_id] = name
    except Exception as e:
        print(f"⚠️ 撈取全市場名單異常: {e}")

    if not all_tickers:
        backup_dict = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科"}
        for k, v in backup_dict.items():
            all_tickers.append(k)
            DYNAMIC_STOCK_NAMES[k] = v
            
    return sorted(list(set(all_tickers)))

# ==============================================================================
# 📊 策略八專用：營收月增季增雙增檢測
# ==============================================================================
def check_revenue_growth(ticker):
    """
    針對策略 1~7 篩出的候選股檢測：
    - 月增 (MoM): (當月營收 - 上月營收) / 上月營收 > 0%
    - 季增 (QoQ): (近3個月營收總和 - 前3個月營收總和) / 前3個月營收總和 > 0%
    """
    try:
        code = ticker.split('.')[0]
        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id={code}"
        res = requests.get(url, timeout=5)
        
        if res.status_code == 200:
            data = res.json().get("data", [])
            if len(data) >= 6:
                df_rev = pd.DataFrame(data).sort_values("date")
                revs = df_rev['revenue'].tolist()[-6:]
                
                # 1. 月增率計算 (MoM)
                r0, r1 = revs[-1], revs[-2]
                mom = ((r0 - r1) / r1) * 100 if r1 > 0 else 0
                
                # 2. 季增率計算 (QoQ)
                s_recent = sum(revs[-3:])
                s_prev = sum(revs[:3])
                qoq = ((s_recent - s_prev) / s_prev) * 100 if s_prev > 0 else 0
                
                if mom > 0 and qoq > 0:
                    return True, round(mom, 1), round(qoq, 1)
    except Exception:
        pass
    return False, 0.0, 0.0

# ==============================================================================
# 📈 技術面指標計算模組
# ==============================================================================
def calculate_macd(close_series, fast=12, slow=26, signal=9):
    fast_ema = close_series.ewm(span=fast, adjust=False).mean()
    slow_ema = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_kd(df_single, n=9, m1=3, m2=3):
    low_min = df_single['Low'].astype(float).rolling(window=n).min()
    high_max = df_single['High'].astype(float).rolling(window=n).max()
    close = df_single['Close'].astype(float)
    
    rsv = ((close - low_min) / (high_max - low_min)) * 100
    rsv = rsv.fillna(50)
    
    k_list, d_list = [50.0], [50.0]
    for i in range(1, len(rsv)):
        current_k = (k_list[-1] * (m1 - 1) + rsv.iloc[i]) / m1
        current_d = (d_list[-1] * (m2 - 1) + current_k) / m2
        k_list.append(current_k)
        d_list.append(current_d)
        
    return pd.Series(k_list, index=df_single.index), pd.Series(d_list, index=df_single.index)

def calculate_rsi(close_series, period=6):
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# ==============================================================================
# 🎯 核心策略檢測邏輯 (1~7)
# ==============================================================================
def check_macd_up_and_kd_gold(df_single):
    try:
        if df_single.empty or len(df_single) < 26: return False
        c = df_single['Close'].squeeze().astype(float)
        
        macd_line, signal_line, hist = calculate_macd(c)
        if len(macd_line) < 2: return False
        
        macd_up = (macd_line.iloc[-1] > macd_line.iloc[-2]) and (
            macd_line.iloc[-1] >= 0 or (hist.iloc[-1] > hist.iloc[-2])
        )
        
        k_ser, d_ser = calculate_kd(df_single)
        if len(k_ser) < 2: return False
        
        kd_gold = (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-2] <= d_ser.iloc[-2])
        return macd_up and kd_gold
    except Exception:
        return False

def check_strat1_resonance(df_30m, df_60m):
    return check_macd_up_and_kd_gold(df_30m) and check_macd_up_and_kd_gold(df_60m)

def check_strat2_resonance(df_60m, df_daily):
    return check_macd_up_and_kd_gold(df_60m) and check_macd_up_and_kd_gold(df_daily)

def check_strat3_resonance(df_daily, df_weekly):
    return check_macd_up_and_kd_gold(df_daily) and check_macd_up_and_kd_gold(df_weekly)

def check_volume_breakout(df_daily):
    try:
        if df_daily.empty or len(df_daily) < 20: return False
        c_daily = df_daily['Close'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        
        ma20 = c_daily.rolling(window=20).mean()
        close_today, close_yesterday = c_daily.iloc[-1], c_daily.iloc[-2]
        ma20_today, ma20_yesterday = ma20.iloc[-1], ma20.iloc[-2]
        
        price_break = (close_today > ma20_today) and (close_yesterday <= ma20_yesterday or (close_today - close_yesterday) / close_yesterday > 0.02)
        if not price_break: return False
        
        v_ma5 = v_daily.rolling(window=5).mean().iloc[-1]
        volume_today = v_daily.iloc[-1]
        if volume_today <= (v_ma5 * 1.5): return False
        
        k_series, d_series = calculate_kd(df_daily)
        if (k_series.iloc[-1] > d_series.iloc[-1]) and (k_series.iloc[-1] < 75):
            return True, volume_today / v_ma5 if v_ma5 > 0 else 1.0
    except Exception:
        pass
    return False

def check_extreme_drop_volume_up(df_daily):
    try:
        if df_daily.empty or len(df_daily) < 20: return False
        c_daily = df_daily['Close'].squeeze().astype(float)
        o_daily = df_daily['Open'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        
        rsi6 = calculate_rsi(c_daily, period=6).iloc[-1]
        v_ma5 = v_daily.rolling(window=5).mean().iloc[-1]
        
        if rsi6 < 20 and c_daily.iloc[-1] > o_daily.iloc[-1] and v_daily.iloc[-1] > v_ma5:
            return True
    except Exception:
        pass
    return False

def check_multi_timeframe_tangling(df_60m, df_daily, df_weekly):
    try:
        c_60m, c_daily, c_weekly = df_60m['Close'].squeeze().astype(float), df_daily['Close'].squeeze().astype(float), df_weekly['Close'].squeeze().astype(float)
        if len(c_60m) < 20 or len(c_daily) < 20 or len(c_weekly) < 20: return False
        
        m60_t = (max(c_60m.rolling(5).mean().iloc[-1], c_60m.rolling(10).mean().iloc[-1], c_60m.rolling(20).mean().iloc[-1]) - min(c_60m.rolling(5).mean().iloc[-1], c_60m.rolling(10).mean().iloc[-1], c_60m.rolling(20).mean().iloc[-1])) / c_60m.rolling(20).mean().iloc[-1]
        d_t = (max(c_daily.rolling(5).mean().iloc[-1], c_daily.rolling(10).mean().iloc[-1], c_daily.rolling(20).mean().iloc[-1]) - min(c_daily.rolling(5).mean().iloc[-1], c_daily.rolling(10).mean().iloc[-1], c_daily.rolling(20).mean().iloc[-1])) / c_daily.rolling(20).mean().iloc[-1]
        w_t = (max(c_weekly.rolling(5).mean().iloc[-1], c_weekly.rolling(10).mean().iloc[-1], c_weekly.rolling(20).mean().iloc[-1]) - min(c_weekly.rolling(5).mean().iloc[-1], c_weekly.rolling(10).mean().iloc[-1], c_weekly.rolling(20).mean().iloc[-1])) / c_weekly.rolling(20).mean().iloc[-1]
        
        if m60_t < 0.025 and d_t < 0.03 and w_t < 0.035 and c_daily.iloc[-1] > c_daily.rolling(20).mean().iloc[-1]:
            return True
    except Exception:
        pass
    return False

def check_low_position_volume_surge(df_daily):
    try:
        if df_daily.empty or len(df_daily) < 120: return False
        c_daily, o_daily = df_daily['Close'].squeeze().astype(float), df_daily['Open'].squeeze().astype(float)
        h_daily, l_daily = df_daily['High'].squeeze().astype(float), df_daily['Low'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        
        if c_daily.iloc[-1] <= o_daily.iloc[-1]: return False
        
        v_ma5 = v_daily.rolling(window=5).mean().iloc[-1]
        if v_ma5 == 0 or v_daily.iloc[-1] < (v_ma5 * 2.5): return False
        
        high_120, low_120 = h_daily.iloc[-120:].max(), l_daily.iloc[-120:].min()
        if high_120 == low_120: return False
        
        pos = (c_daily.iloc[-1] - low_120) / (high_120 - low_120)
        if pos <= 0.30:
            return True, round(pos * 100, 1), round(v_daily.iloc[-1] / v_ma5, 1)
    except Exception:
        pass
    return False

# ==============================================================================
# 💬 Telegram 發送
# ==============================================================================
def send_telegram_message(message):
    bot_token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if not bot_token or not chat_id: return
    
    bot_token = str(bot_token).strip()
    chat_id = str(chat_id).strip()
    if bot_token.lower().startswith("bot"): bot_token = bot_token[3:]

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        print(f"📢 TG 發送反饋: 狀態碼 {res.status_code}")
    except Exception as e: 
        print(f"❌ Telegram 發送異常: {e}")

# ==============================================================================
# 🚀 主程式
# ==============================================================================
if __name__ == "__main__":
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 啟動【台股 8 大策略全方位篩選報告】...")
    tech_scan_pool = fetch_all_taiwan_market_tickers()
    if not tech_scan_pool: exit()

    print(f"⏳ 步驟 1: 下載全市場日K數據 (過濾 20日均量 < 1000張)...")
    full_df_daily = yf.download(tech_scan_pool, period="1y", interval="1d", progress=False, auto_adjust=True)
    
    qualified_tickers = []
    for ticker in tech_scan_pool:
        try:
            v_daily = full_df_daily['Volume'].squeeze() if len(tech_scan_pool) == 1 else full_df_daily.xs(ticker, axis=1, level=1)['Volume'].squeeze()
            if len(v_daily) >= 20 and (v_daily.rolling(window=20).mean().iloc[-1] / 1000) >= 1000:
                qualified_tickers.append(ticker)
        except Exception:
            continue

    print(f"🎯 通過量能門檻股票共 {len(qualified_tickers)} 檔。")
    
    strat1_matches, strat2_matches, strat3_matches, strat4_matches, strat5_matches, strat6_matches, strat7_matches, strat8_matches = [], [], [], [], [], [], [], []
    tech_candidates_union = set()  # 儲存符合 1~7 策略的標的聯集

    if qualified_tickers:
        print("⏳ 步驟 2: 批次下載多週期 K 線資料 (30m, 60m, Weekly)...")
        full_df_30m = yf.download(qualified_tickers, period="1mo", interval="30m", progress=False, auto_adjust=True)
        full_df_60m = yf.download(qualified_tickers, period="1mo", interval="60m", progress=False, auto_adjust=True)
        full_df_weekly = yf.download(qualified_tickers, period="2y", interval="1wk", progress=False, auto_adjust=True)

        print("⏳ 步驟 3: 執行策略 1~7 技術面檢測...")
        for ticker in qualified_tickers:
            try:
                df_d = full_df_daily.xs(ticker, axis=1, level=1)
                df_m30 = full_df_30m.xs(ticker, axis=1, level=1)
                df_m60 = full_df_60m.xs(ticker, axis=1, level=1)
                df_w = full_df_weekly.xs(ticker, axis=1, level=1)

                if df_d.empty or df_m30.empty or df_m60.empty or df_w.empty: continue

                name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
                stock_label = f"<code>{ticker}</code>(<i>{name_zh}</i>)" if name_zh else f"<code>{ticker}</code>"

                # 策略一
                if check_strat1_resonance(df_m30, df_m60):
                    strat1_matches.append(stock_label)
                    tech_candidates_union.add(ticker)
                    
                # 策略二
                if check_strat2_resonance(df_m60, df_d):
                    strat2_matches.append(stock_label)
                    tech_candidates_union.add(ticker)

                # 策略三
                if check_strat3_resonance(df_d, df_w):
                    strat3_matches.append(stock_label)
                    tech_candidates_union.add(ticker)

                # 策略四
                v_break = check_volume_breakout(df_d)
                if v_break:
                    strat4_matches.append(f"{stock_label}[量比:{v_break[1]:.1f}倍]")
                    tech_candidates_union.add(ticker)

                # 策略五
                if check_extreme_drop_volume_up(df_d):
                    strat5_matches.append(stock_label)
                    tech_candidates_union.add(ticker)

                # 策略六
                if check_multi_timeframe_tangling(df_m60, df_d, df_w):
                    strat6_matches.append(stock_label)
                    tech_candidates_union.add(ticker)

                # 策略七
                low_vol = check_low_position_volume_surge(df_d)
                if low_vol:
                    strat7_matches.append(f"{stock_label}[位階:{low_vol[1]}%|量比:{low_vol[2]}倍]")
                    tech_candidates_union.add(ticker)

            except Exception:
                continue

    # --------------------------------------------------------------------------
    # 🔍 步驟 4: 執行【策略八】(彙整 1~7 策略出的標的，過濾營收月增 > 0% 且 季增 > 0%)
    # --------------------------------------------------------------------------
    print(f"⏳ 步驟 4: 執行【策略八】(針對 1~7 策略共 {len(tech_candidates_union)} 檔技術標的進行營收雙增檢測)...")
    for ticker in sorted(tech_candidates_union):
        is_rev_pass, mom_val, qoq_val = check_revenue_growth(ticker)
        time.sleep(0.1)  # 避免 API 觸發頻率限制

        if is_rev_pass:
            name_zh = DYNAMIC_STOCK_NAMES.get(ticker, "")
            label = f"<code>{ticker}</code>(<i>{name_zh}</i>)[月增:{mom_val}%|季增:{qoq_val}%]" if name_zh else f"<code>{ticker}</code>[月增:{mom_val}%|季增:{qoq_val}%]"
            strat8_matches.append(label)

    # 📝 建立 8 大策略完整 Telegram 報告
    tw_msg = f"🇹🇼 <b>【台股盤後 8 大策略選股報告】</b>\n"
    tw_msg += f"⚠️ <i>已自動過濾 20日均量 &lt; 1000張之低流動性股</i>\n"
    tw_msg += f"⏰ 時間: {tw_time_str}\n───────────────────\n\n"
    
    tw_msg += "📈 <b>【策略一】30分K & 60分K 共振</b>\n↳ " + (", ".join(strat1_matches) if strat1_matches else "今日無符合標的。 💤") + "\n\n"
    tw_msg += "📈 <b>【策略二】60分K & 日K 共振</b>\n↳ " + (", ".join(strat2_matches) if strat2_matches else "今日無符合標的。 💤") + "\n\n"
    tw_msg += "📈 <b>【策略三】日K & 週K 共振</b>\n↳ " + (", ".join(strat3_matches) if strat3_matches else "今日無符合標的。 💤") + "\n\n"
    tw_msg += "⚡ <b>【策略四】帶量突破</b>\n↳ " + (", ".join(strat4_matches) if strat4_matches else "今日無符合標的。 💤") + "\n\n"
    tw_msg += "🔥 <b>【策略五】恐慌止跌 (極限超賣爆量)</b>\n↳ " + (", ".join(strat5_matches) if strat5_matches else "今日無符合標的。 💤") + "\n\n"
    tw_msg += "💎 <b>【策略六】全週期同步糾結</b>\n↳ " + (", ".join(strat6_matches) if strat6_matches else "今日無符合標的。 💤") + "\n\n"
    tw_msg += "💥 <b>【策略七】低檔爆量股</b>\n↳ " + (", ".join(strat7_matches) if strat7_matches else "今日無符合標的。 💤") + "\n\n"
    tw_msg += "🏆 <b>【策略八】技術精選 × 營收雙增 (策略1~7標的中 月增&gt;0% &amp; 季增&gt;0%)</b>\n↳ " + (", ".join(strat8_matches) if strat8_matches else "無符合營收雙增之標的。 💤") + "\n"

    send_telegram_message(tw_msg)
    print("✅ 8 大策略選股報告發送完畢！")
