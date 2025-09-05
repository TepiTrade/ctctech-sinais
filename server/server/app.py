# server/app.py — LIVE (Plataforma + Auto TF + Neural ON)
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
import numpy as np
import json, os

APP_DIR = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(APP_DIR, "weights.json")
STORE_DIR = os.path.join(APP_DIR, "store")
os.makedirs(STORE_DIR, exist_ok=True)

def load_weights():
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, "r") as f:
            return json.load(f)
    w = {
        "ema_fast": 9,
        "ema_slow": 21,
        "fib_tolerance_bps": 20,
        "ema_gap_min_bps": 5,
        "body_strength_mult": 1.15,
        "neural_active": True
    }
    save_weights(w); return w

def save_weights(w):
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(w, f, indent=2)

WEIGHTS = load_weights()
app = FastAPI(title="CTCTECH Sinais – LIVE", version="1.3.0")

# -------- utils --------
def ema(arr, period):
    arr = np.array(arr, dtype=float)
    if len(arr) < period+5:
        return [np.nan]*len(arr)
    k = 2/(period+1.0)
    ema_vals = np.zeros_like(arr)
    ema_vals[:] = np.nan
    sma = np.nanmean(arr[:period])
    ema_vals[period-1] = sma
    for i in range(period, len(arr)):
        if np.isnan(ema_vals[i-1]): ema_vals[i] = arr[i]
        else: ema_vals[i] = (arr[i] - ema_vals[i-1])*k + ema_vals[i-1]
    return ema_vals

def last_swing_high_low(highs, lows, window=30):
    h = float(np.max(highs[-window:])) if len(highs) >= 2 else np.nan
    l = float(np.min(lows[-window:])) if len(lows) >= 2 else np.nan
    return h, l

def fib_levels(high, low):
    diff = high - low
    return {"0.382": high - 0.382*diff, "0.5": high - 0.5*diff, "0.618": high - 0.618*diff}

def fib_levels_down(high, low):
    diff = high - low
    return {"0.382": low + 0.382*diff, "0.5": low + 0.5*diff, "0.618": low + 0.618*diff}

def bps(a, b): return abs((a - b) / b) * 10000.0 if b != 0 else 1e9
def strength_label(p):
    if p >= 90: return "Muito Forte"
    if p >= 80: return "Forte"
    if p >= 65: return "Média"
    return "Fraca"

def store_key(platform: str, symbol: str, timeframe: str):
    safe_plat = platform.replace(" ", "").replace("/", "").lower() or "generic"
    return os.path.join(STORE_DIR, f"{safe_plat}_{symbol.replace('/','')}_{timeframe}.json")

# -------- models --------
class Candle(BaseModel):
    t: int; o: float; h: float; l: float; c: float

class PushRequest(BaseModel):
    platform: str
    symbol: str
    timeframe: str
    candles: List[Candle]

class SignalRequest(BaseModel):
    platform: str = Field(..., description="Ex.: Casa Trade, Broker10, IQ, MT5, Binance...")
    symbol: str = Field(..., description="ex: EURUSD")
    timeframe: Optional[str] = Field("auto", description="15s,30s,1m,2m,3m,5m,15m,30m,1h,4h ou 'auto'")

# -------- core --------
def evaluate_signal_from_candles(candles: List[Dict]) -> Tuple[Dict, float]:
    closes = [x["c"] if isinstance(x, dict) else x.c for x in candles]
    highs  = [x["h"] if isinstance(x, dict) else x.h for x in candles]
    lows   = [x["l"] if isinstance(x, dict) else x.l for x in candles]
    opens  = [x["o"] if isinstance(x, dict) else x.o for x in candles]

    efast = ema(closes, WEIGHTS["ema_fast"])
    eslow = ema(closes, WEIGHTS["ema_slow"])
    last_close = float(closes[-1]); last_open  = float(opens[-1])

    bodies = [abs(c - o) for c, o in zip(closes, opens)]
    mean_body = float(np.nanmean(bodies[-30:]))
    strong_candle = abs(last_close - last_open) > (WEIGHTS["body_strength_mult"] * mean_body)

    ema_gap_bps = bps(efast[-1], eslow[-1])
    swing_h, swing_l = last_swing_high_low(highs, lows, window=30)

    operation = None; reason = []; fib_zone = None; conf = 60.0

    if not np.isnan(efast[-1]) and not np.isnan(eslow[-1]):
        if efast[-1] > eslow[-1]:
            levels = fib_levels(swing_h, swing_l)
            near_05   = bps(last_close, levels["0.5"])   <= WEIGHTS["fib_tolerance_bps"]
            near_0618 = bps(last_close, levels["0.618"]) <= WEIGHTS["fib_tolerance_bps"]
            if (near_05 or near_0618) and strong_candle and last_close > last_open:
                fib_zone = "0.5-0.618"; operation = "Compra"; reason.append("Alta (EMAs) + pullback Fibo + candle de alta")
        elif efast[-1] < eslow[-1]:
            levels = fib_levels_down(swing_h, swing_l)
            near_05   = bps(last_close, levels["0.5"])   <= WEIGHTS["fib_tolerance_bps"]
            near_0618 = bps(last_close, levels["0.618"]) <= WEIGHTS["fib_tolerance_bps"]
            if (near_05 or near_0618) and strong_candle and last_close < last_open:
                fib_zone = "0.5-0.618"; operation = "Venda"; reason.append("Baixa (EMAs) + pullback Fibo + candle de baixa")

    if ema_gap_bps >= WEIGHTS["ema_gap_min_bps"]:
        conf += 15; reason.append("EMAs separadas")
    if strong_candle:
        conf += 10; reason.append("Candle de força")
    if fib_zone:
        conf += 10; reason.append("Proximidade Fibo")

    conf = float(max(55.0, min(98.0, conf)))
    strength = strength_label(conf)
    explain = {
        "ema_fast": float(efast[-1]) if not np.isnan(efast[-1]) else None,
        "ema_slow": float(eslow[-1]) if not np.isnan(eslow[-1]) else None,
        "ema_gap_bps": float(ema_gap_bps) if ema_gap_bps != 1e9 else None,
        "fib_zone": fib_zone,
        "reason": " | ".join(reason) if reason else "Condições parciais; sem confluência total"
    }
    return {"operation": operation or "Aguardar", "confidence": round(conf, 1), "strength": strength, "explain": explain}, conf

# -------- routes --------
@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(os.path.dirname(APP_DIR), "index.html")
    return FileResponse(index_path)

@app.post("/api/push")
def api_push(req: PushRequest):
    d = req.dict()
    fp = store_key(d["platform"], d["symbol"], d["timeframe"])
    with open(fp, "w") as f:
        json.dump([c for c in d["candles"]], f)
    return {"ok": True, "stored": fp, "count": len(d["candles"])}

@app.post("/api/signal")
def api_signal(req: SignalRequest):
    platform = req.platform; symbol = req.symbol
    tf = (req.timeframe or "auto").lower()
    candidates = ["15s","30s","1m","2m","3m","5m","15m","30m","1h","4h"]

    def load_tf(tfname):
        fp = store_key(platform, symbol, tfname)
        if os.path.exists(fp):
            with open(fp, "r") as f: data = json.load(f)
            return data
        return None

    results = []
    if tf == "auto":
        for t in candidates:
            candles = load_tf(t)
            if candles and len(candles) >= 40:
                sig, conf = evaluate_signal_from_candles(candles)
                results.append((t, sig, conf))
        if not results:
            return JSONResponse({
                "platform": platform, "symbol": symbol, "timeframe": "auto",
                "operation": "Aguardar", "confidence": 60.0, "strength": "Média",
                "entry_time": "Abertura",
                "explain": {"reason": "Sem dados. Envie candles via /api/push (platform/symbol/timeframe)."}
            })
        results.sort(key=lambda x: x[2], reverse=True)
        best_tf, best_sig, _ = results[0]
        resp = {"platform": platform, "symbol": symbol, "timeframe": best_tf, **best_sig, "entry_time": "Abertura"}
        resp["explain"]["reason"] = f"[{platform}] IA sugeriu TF {best_tf} • " + resp["explain"]["reason"]
        return JSONResponse(resp)
    else:
        candles = load_tf(tf)
        if not candles or len(candles) < 40:
            return JSONResponse({
                "platform": platform, "symbol": symbol, "timeframe": tf,
                "operation": "Aguardar", "confidence": 60.0, "strength": "Média",
                "entry_time": "Abertura",
                "explain": {"reason": f"[{platform}] Sem dados ou poucos candles (<40) para {tf}"}
            })
        sig, _ = evaluate_signal_from_candles(candles)
        return JSONResponse({"platform": platform, "symbol": symbol, "timeframe": tf, **sig, "entry_time": "Abertura"})

@app.post("/api/feedback")
def api_feedback(payload: Dict):
    win = bool(payload.get("win", True))
    w = dict(WEIGHTS)
    if win:
        w["fib_tolerance_bps"] = max(8, int(w["fib_tolerance_bps"] * 0.96))
        w["ema_gap_min_bps"] = max(3, int(w["ema_gap_min_bps"] * 0.98))
    else:
        w["fib_tolerance_bps"] = min(35, int(w["fib_tolerance_bps"] * 1.08))
        w["ema_gap_min_bps"] = min(15, int(w["ema_gap_min_bps"] * 1.05))
    save_weights(w); WEIGHTS.update(w)
    return {"ok": True, "weights": w}
