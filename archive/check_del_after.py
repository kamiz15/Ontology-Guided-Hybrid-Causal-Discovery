from pathlib import Path
import runpy


SCRIPT_PATH = Path(__file__).resolve().parent / "scripts" / "check_del_after.py"
runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
