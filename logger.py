# -------------------------------
# file: src/utils/logger.py
# -------------------------------
from rich.console import Console
from rich.table import Table
console = Console()

def log_kv(title: str, kv: dict):
    table = Table(title=title)
    table.add_column("Key"); table.add_column("Value")
    for k, v in kv.items():
        table.add_row(str(k), str(v))
    console.print(table)

