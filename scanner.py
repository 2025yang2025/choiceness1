import pandas as pd
import yfinance as yf
import requests
import os
import time
import html

# ==============================================================================
# 🇹🇼 台股熱門排行模組 (依當日成交量排序，擷取 Top N 熱門股)
# ==============================================================================
DYNAMIC_STOCK_NAMES = {}

def fetch_popular_taiwan_tickers(top_n=100):
    """ 從證交所抓取當日成交量熱門排行榜 (Top N) """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    popular_tickers = []
    
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        res = requests.get(url_twse, headers=headers, timeout=10)
        if res.status_code == 200:
            raw_data = res.json()
            valid_list = []
            
            for item in raw_data:
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()
                vol_str = str(item.get("TradeVolume", "0")).replace(",", "")
                
                # 僅保留普通股 (4位數代碼)
                if code.isdigit() and len(code) == 4:
                    try:
                        trade_vol = int(vol_str)
                        valid_list.append({
                            "code": code,
                            "name": name,
                            "volume": trade_vol
                        })
                    except ValueError:
                        continue
            
            # 依成交量 (TradeVolume) 由大到小排序，取前 Top N 檔熱門股
            sorted_list = sorted(valid_list, key=lambda x: x["volume"], reverse=True)[:top_n]
            
            for item in sorted_list:
                ticker_id = f"{item['code']}.TW"
                popular_tickers.append(ticker_id)
                DYNAMIC_STOCK_NAMES[ticker_id] = item["name"]
                
            print(f"🔥 成功擷取台股成交量熱門排行前 {len(popular_tickers)} 檔標的。")
            
    except Exception as e:
        print(f"⚠️ 撈取熱門排行名單異常: {e}")

    if not popular_tickers:
        backup_dict = {"2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2303.TW": "聯電", "2382.TW": "廣達"}
        for k, v in backup_dict.items():
            popular_tickers.append(k)
            DYNAMIC_STOCK_NAMES[k] = v
            
    return popular_tickers

# ==============================================================================
# 📊 技術指標計算模組
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
    
    denom = high_max - low_min
    denom = denom.replace(0, pd.NA) # 防範除以 0
    rsv = ((close - low_min) / denom) * 100
    rsv = rsv.fillna(50).astype(float)
    
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
    
    loss = loss.replace(0, pd.NA) # 防範除以 0
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# ==============================================================================
# 🎯 核心策略檢測邏輯
# ==============================================================================

def check_macd_up_and_kd_gold(df_single):
    """ 通用模組：MACD 往 0 軸向上 + KD 黃金交叉 """
    try:
        if df_single.empty or len(df_single) < 26: return False
        c = df_single['Close'].squeeze()
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
        c = c.astype(float)
        
        macd_line, signal_line, hist = calculate_macd(c)
        if len(macd_line) < 2: return False
        
        macd_up = (macd_line.iloc[-1] > macd_line.iloc[-2]) and (
            macd_line.iloc[-1] >= 0 or (hist.iloc[-1] > hist.iloc[-2])
        )
        
        k_ser, d_ser = calculate_kd(df_single)
        if len(k_ser) < 2: return False
        
        kd_gold = (k_ser.iloc[-1] > d_ser.iloc[-1]) and (k_ser.iloc[-2] <= d_ser.iloc[-2])
        
        return bool(macd_up and kd_gold)
    except Exception:
        return False

def check_volume_breakout(df_daily):
    """ 策略四：關鍵均線多頭突破 × 量能倍增 (帶量突破) """
    try:
        if df_daily.empty or len(df_daily) < 20: return False
        
        c_daily = df_daily['Close'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        
        ma20 = c_daily.rolling(window=20).mean()
        close_today = c_daily.iloc[-1]
        close_yesterday = c_daily.iloc[-2]
        ma20_today = ma20.iloc[-1]
        ma20_yesterday = ma20.iloc[-2]
        
        price_break_cond = (close_today > ma20_today) and (close_yesterday <= ma20_yesterday or (close_today - close_yesterday) / close_yesterday > 0.02)
        if not price_break_cond: return False
        
        v_ma5 = v_daily.rolling(window=5).mean().iloc[-1]
        volume_today = v_daily.iloc[-1]
        volume_cond = volume_today > (v_ma5 * 1.5)
        if not volume_cond: return False
        
        k_series, d_series = calculate_kd(df_daily)
        k_today = k_series.iloc[-1]
        d_today = d_series.iloc[-1]
        kd_cond = (k_today > d_today) and (k_today < 75)
        
        if kd_cond:
            volume_ratio = volume_today / v_ma5 if v_ma5 > 0 else 1.0
            return True, volume_ratio
    except Exception:
        pass
    return False

def check_extreme_drop_volume_up(df_daily):
    """ 策略五：短線極限超賣 × 爆量紅K (恐慌止跌) """
    try:
        if df_daily.empty or len(df_daily) < 20: return False
        c_daily = df_daily['Close'].squeeze().astype(float)
        o_daily = df_daily['Open'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        
        rsi6 = calculate_rsi(c_daily, period=6).iloc[-1]
        close_today = c_daily.iloc[-1]
        open_today = o_daily.iloc[-1]
        volume_today = v_daily.iloc[-1]
        v_ma5 = v_daily.rolling(window=5).mean().iloc[-1]
        
        if rsi6 < 20 and close_today > open_today and volume_today > v_ma5:
            return True
    except Exception:
        pass
    return False

def check_low_position_volume_surge(df_daily):
    """ 策略六（原策略七）：低檔爆量股 (半年位階 ≤ 30% × 成交量 ≥ 2.5倍5日均量 × 紅K) """
    try:
        if df_daily.empty or len(df_daily) < 120: return False
        
        c_daily = df_daily['Close'].squeeze().astype(float)
        o_daily = df_daily['Open'].squeeze().astype(float)
        h_daily = df_daily['High'].squeeze().astype(float)
        l_daily = df_daily['Low'].squeeze().astype(float)
        v_daily = df_daily['Volume'].squeeze().astype(float)
        
        close_today = c_daily.iloc[-1]
        open_today = o_daily.iloc[-1]
        
        if close_today <= open_today:
            return False
            
        v_ma5 = v_daily.rolling(window=5).mean().iloc[-1]
        volume_today = v_daily.iloc[-1]
        if v_ma5 == 0 or volume_today < (v_ma5 * 2.5):
            return False
            
        high_120 = h_daily.iloc[-120:].max()
        low_120 = l_daily.iloc[-120:].min()
        
        if high_120 == low_120:
            return False
            
        position = (close_today - low_120) / (high_120 - low_120)
        
        if position <= 0.30:
            vol_ratio = volume_today / v_ma5
            return True, round(position * 100, 1), round(vol_ratio, 1)
            
    except Exception:
        pass
    return False

# ==============================================================================
# 💬 Telegram 發送模組
# ==============================================================================
def send_telegram_message(message, max_length=3500):
    bot_token = os.environ.get("TG_BOT_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if not bot_token or not chat_id:
        print("⚠️ 未設定 TG_BOT_TOKEN 或 TG_CHAT_ID，跳過發送。")
        return
    
    bot_token = str(bot_token).strip()
    chat_id = str(chat_id).strip()
    if bot_token.lower().startswith("bot"):
        bot_token = bot_token[3:]

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    lines = message.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    if current_chunk:
        chunks.append(current_chunk)

    for idx, chunk in enumerate(chunks):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML"
        }
        try:
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200:
                print(f"📢 TG 發送成功 ({idx+1}/{len(chunks)})")
            else:
                print(f"❌ TG 發送失敗 ({idx+1}/{len(chunks)}), 狀態碼: {res.status_code}, 內文: {res.text}")
        except Exception as e:
            print(f"❌ Telegram 發送異常: {e}")
        time.sleep(0.5)

# ==============================================================================
# 🚀 主程式
# ==============================================================================
if __name__ == "__main__":
    now_tw = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Taipei')
    tw_time_str = now_tw.strftime('%Y-%m-%d %H:%M:%S')

    print("🚀 啟動【台股熱門排行 Top 100 策略篩選報告】...")
    
    popular_scan_pool = fetch_popular_taiwan_tickers(top_n=100)
    
    if not popular_scan_pool:
        print("❌ 未能取得熱門標的名單，程式結束。")
        exit()

    print(f"⏳ 批次下載熱門標的的多週期 K 線數據 (共 {len(popular_scan_pool)} 檔)...")
    full_df_daily = yf.download(popular_scan_pool, period="1y", interval="1d", progress=False, auto_adjust=True)
    full_df_60m = yf.download(popular_scan_pool, period="1mo", interval="60m", progress=False, auto_adjust=True)
    full_df_weekly = yf.download(popular_scan_pool, period="2y", interval="1wk", progress=False, auto_adjust=True)

    strat1_matches, strat2_matches, strat3_matches, strat4_matches, strat5_matches, strat6_matches = [], [], [], [], [], []

    print("⏳ 記憶體內多維策略流檢測中...")
    for ticker in popular_scan_pool:
        try:
            # 安全擷取特定 Ticker DataFrame 邏輯
            def get_ticker_df(full_df, ticker):
                if isinstance(full_df.columns, pd.MultiIndex):
                    if ticker in full_df.columns.get_level_values(1):
                        return full_df.xs(ticker, axis=1, level=1)
                    elif ticker in full_df.columns.get_level_values(0):
                        return full_df.xs(ticker, axis=1, level=0)
                return full_df

            df_d = get_ticker_df(full_df_daily, ticker)
            df_m60 = get_ticker_df(full_df_60m, ticker)
            df_w = get_ticker_df(full_df_weekly, ticker)

            if df_d.empty or df_m60.empty or df_w.empty: 
                continue

            # 擷取最新價格與名稱
            raw_code = ticker.replace(".TW", "")
            name_zh = html.escape(DYNAMIC_STOCK_NAMES.get(ticker, ""))
            
            latest_price = df_d['Close'].squeeze().iloc[-1]
            price_str = f"${latest_price:.2f}".rstrip('0').rstrip('.') # 格式化價格顯示
            
            # 格式：股票代號+中文名稱+價格
            stock_label = f"<code>{raw_code}</code>({name_zh}) {price_str}" if name_zh else f"<code>{raw_code}</code> {price_str}"

            # 策略一：60分K (MACD 往0軸向上 + KD金叉)
            if check_macd_up_and_kd_gold(df_m60):
                strat1_matches.append(stock_label)
                
            # 策略二：日K (MACD 往0軸向上 + KD金叉)
            if check_macd_up_and_kd_gold(df_d):
                strat2_matches.append(stock_label)

            # 策略三：週K (MACD 往0軸向上 + KD金叉)
            if check_macd_up_and_kd_gold(df_w):
                strat3_matches.append(stock_label)

            # 策略四：關鍵均線多頭突破 × 量能倍增 (帶量突破)
            vol_breakout_check = check_volume_breakout(df_d)
            if vol_breakout_check:
                _, vol_ratio = vol_breakout_check
                strat4_matches.append(f"{stock_label} [量比:{vol_ratio:.1f}倍]")

            # 策略五：短線極限超賣 × 爆量紅K (恐慌止跌)
            if check_extreme_drop_volume_up(df_d):
                strat5_matches.append(stock_label)

            # 策略六（原策略七）：低檔爆量股 (半年位階 ≤ 30% × 成交量 ≥ 2.5倍5日均量 × 紅K)
            low_vol_check = check_low_position_volume_surge(df_d)
            if low_vol_check:
                _, pos_val, vol_r = low_vol_check
                strat6_matches.append(f"{stock_label} [位階:{pos_val}%|量比:{vol_r}倍]")

        except Exception as e:
            print(f"⚠️ 處理個股 {ticker} 時發生異常: {e}")
            continue

    # 📝 建立發送文字報告
    tw_msg = f"🔥 <b>【台股熱門排行 Top 100 多策略精選】</b>\n⏰ 時間: {tw_time_str}\n"
    tw_msg += "───────────────────\n\n"
    
    tw_msg += "📈 <b>【策略一】60分K (MACD 往0軸向上 + KD金叉)</b>\n"
    tw_msg += "↳ " + (", ".join(strat1_matches) if strat1_matches else "熱門標的中無符合標的。 💤") + "\n\n"

    tw_msg += "📈 <b>【策略二】日K (MACD 往0軸向上 + KD金叉)</b>\n"
    tw_msg += "↳ " + (", ".join(strat2_matches) if strat2_matches else "熱門標的中無符合標的。 💤") + "\n\n"

    tw_msg += "📈 <b>【策略三】週K (MACD 往0軸向上 + KD金叉)</b>\n"
    tw_msg += "↳ " + (", ".join(strat3_matches) if strat3_matches else "熱門標的中無符合標的。 💤") + "\n\n"

    tw_msg += "⚡ <b>【策略四】關鍵均線多頭突破 × 量能倍增 (帶量突破)</b>\n"
    tw_msg += "↳ " + (", ".join(strat4_matches) if strat4_matches else "熱門標的中無符合標的。 💤") + "\n\n"

    tw_msg += "🔥 <b>【策略五】短線極限超賣 × 爆量紅K (恐慌止跌)</b>\n"
    tw_msg += "↳ " + (", ".join(strat5_matches) if strat5_matches else "熱門標的中無符合標的。 💤") + "\n\n"

    tw_msg += "💥 <b>【策略六】低檔爆量股 (半年位階 ≤ 30% × 成交量 ≥ 2.5倍5日均量 × 紅K)</b>\n"
    tw_msg += "↳ " + (", ".join(strat6_matches) if strat6_matches else "熱門標的中無符合標的。 💤") + "\n"

    send_telegram_message(tw_msg)
    print("✅ 熱門排行多策略綜合報告發送完畢！")
