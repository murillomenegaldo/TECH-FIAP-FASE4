"""
Simple monitoring script — polls /metrics and prints a live dashboard.
Usage: python monitoring/monitor.py --url http://localhost:8000
"""

import argparse
import time
import requests
from datetime import datetime


def clear():
    print("\033[H\033[J", end="")


def dashboard(url: str, interval: int):
    print(f"🔍 Monitoring {url} every {interval}s — Ctrl+C to stop\n")
    while True:
        try:
            t0 = time.time()
            resp = requests.get(f"{url}/metrics", timeout=5)
            latency = (time.time() - t0) * 1000
            data = resp.json()

            clear()
            print("=" * 55)
            print(f"  📊 LSTM API Monitor — {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 55)
            print(f"  Total requests    : {data['total_requests']}")
            print(f"  Predictions ok    : {data['successful_predictions']}")
            print(f"  Failed requests   : {data['failed_requests']}")

            rt = data["response_time_ms"]
            print(f"\n  ─── Response Time (ms) ──────────────────────")
            print(f"  Avg  : {rt['avg']:>8.2f} ms")
            print(f"  P95  : {rt['p95']:>8.2f} ms")
            print(f"  P99  : {rt['p99']:>8.2f} ms")
            print(f"  Min  : {rt['min']:>8.2f} ms")
            print(f"  Max  : {rt['max']:>8.2f} ms")

            mm = data.get("model_metrics", {})
            if mm:
                print(f"\n  ─── Model Quality ───────────────────────────")
                for k, v in mm.items():
                    print(f"  {k:<5}: {v}")

            recent = data.get("recent_predictions", [])
            if recent:
                print(f"\n  ─── Last Prediction ─────────────────────────")
                last = recent[-1]
                print(f"  Symbol : {last.get('symbol', 'N/A')}")
                if fp := last.get("first_prediction"):
                    print(f"  Date   : {fp.get('date', 'N/A')}")
                    print(f"  Price  : {fp.get('predicted_close', 'N/A')}")

            print(f"\n  Monitor latency   : {latency:.1f} ms")
            print("=" * 55)

        except requests.exceptions.ConnectionError:
            print(f"⚠️  Could not connect to {url}")
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()
    dashboard(args.url, args.interval)
