from core_engine import get_live_snapshot
import time

while True: 
    snap = get_live_snapshot()
    print(snap)
    time.sleep(5)