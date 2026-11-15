"""Las pruebas importan los módulos de la app desde la carpeta de arriba."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.dont_write_bytecode = True
