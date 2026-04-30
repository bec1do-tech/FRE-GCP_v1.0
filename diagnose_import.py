import traceback, sys, os

outpath = os.path.join(os.path.dirname(__file__), "diag_out.txt")
out = open(outpath, "w", encoding="utf-8")
sys.stdout = out
sys.stderr = out

try:
    import cognitive_search_agent.agent
    print("IMPORT OK")
except Exception as e:
    print(f"IMPORT FAILED: {e}")
    traceback.print_exc()
except SystemExit as e:
    print(f"SystemExit: {e}")
    traceback.print_exc()
finally:
    out.flush()
    out.close()
