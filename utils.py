import json
import math
import pathlib
import time
from datetime import datetime
import requests
import pandas as pd
import networkx as nx

# --- Configurações de Caminhos ---
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# URLs
AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
ROUTES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
OPENSKY_FLIGHTS_URL = "https://opensky-network.org/api/flights/all"

# Prefixo dos arquivos de snapshot
SNAPSHOT_PREFIX = "opensky_snapshot_"


def setup_directories():
    """Cria os diretórios de dados e saída se não existirem."""
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula a distância geodésica em km entre dois pontos."""
    R = 6_371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Snapshot OpenSky – Coleta Pontual e Carregamento Local
def _snapshot_filename(dt: datetime | None = None) -> pathlib.Path:
    """Gera o path de um snapshot com timestamp no nome."""
    if dt is None:
        dt = datetime.now()
    ts = dt.strftime("%Y%m%d_%H%M")
    return DATA_DIR / f"{SNAPSHOT_PREFIX}{ts}.json"


def list_snapshots() -> list[pathlib.Path]:
    """Lista todos os snapshots disponíveis em data/, ordenados do mais recente."""
    snapshots = sorted(DATA_DIR.glob(f"{SNAPSHOT_PREFIX}*.json"), reverse=True)
    return snapshots


def collect_opensky_snapshot(hours_back: int = 2) -> pathlib.Path | None:
    """
    Coleta pontual (snapshot) de voos das últimas *hours_back* horas
    via endpoint /flights/all da OpenSky Network. Por padrão, coleta as últimas 2 horas.
    """
    setup_directories()

    end_time = int(time.time())
    begin_time = end_time - (hours_back * 3600)
    snapshot_path = _snapshot_filename()

    params = {"begin": begin_time, "end": end_time}
    print(f"[INFO] Coletando snapshot OpenSky (últimas {hours_back} h)…")

    try:
        resp = requests.get(OPENSKY_FLIGHTS_URL, params=params, timeout=30)
        resp.raise_for_status()
        flights_data = resp.json()

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(flights_data, f, ensure_ascii=False)

        n = len(flights_data) if isinstance(flights_data, list) else 0
        print(f"[SUCCESS] Snapshot salvo em {snapshot_path.name}  ({n} voos)")
        return snapshot_path

    except Exception as e:
        print(f"[ERROR] Falha ao coletar snapshot: {e}")
        return None


def load_opensky_snapshot(path: pathlib.Path | None = None) -> pd.DataFrame:
    """
    Carrega um snapshot local e agrega a contagem de voos por par
    origem-destino (ICAO).
    """
    if path is None:
        candidates = list_snapshots()
        if not candidates:
            print("[WARN] Nenhum snapshot OpenSky encontrado. Usando apenas fallback.")
            return pd.DataFrame()
        path = candidates[0]
        print(f"[INFO] Carregando snapshot mais recente: {path.name}")
    else:
        if not path.exists():
            print(f"[WARN] Snapshot não encontrado em {path}. Usando apenas fallback.")
            return pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Remove voos sem aeroporto estimado de origem ou destino
    df = df.dropna(subset=["estDepartureAirport", "estArrivalAirport"])
    df = df[
        (df["estDepartureAirport"].str.len() == 4)
        & (df["estArrivalAirport"].str.len() == 4)
    ]

    if df.empty:
        print("[WARN] Snapshot não contém pares origem-destino válidos.")
        return pd.DataFrame()

    # Agrega voos por par de aeroportos
    traffic = (
        df.groupby(["estDepartureAirport", "estArrivalAirport"])
        .size()
        .reset_index(name="real_flight_count")
    )
    traffic.columns = ["src_icao", "dst_icao", "real_flight_count"]

    print(f"[INFO] Snapshot carregado: {len(traffic)} pares origem-destino únicos.")
    return traffic


def load_all_opensky_snapshots() -> pd.DataFrame:
    """
    Carrega TODOS os snapshots locais e agrega a contagem total de voos
    por par origem-destino (ICAO), somando os dados de todas as coletas.
    """
    candidates = list_snapshots()
    if not candidates:
        print("[WARN] Nenhum snapshot OpenSky encontrado. Usando apenas fallback.")
        return pd.DataFrame()

    print(f"[INFO] Carregando e agregando {len(candidates)} snapshot(s) OpenSky…")
    frames: list[pd.DataFrame] = []

    for path in candidates:
        single = load_opensky_snapshot(path)
        if not single.empty:
            frames.append(single)

    if not frames:
        print("[WARN] Nenhum snapshot contém pares origem-destino válidos.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Agrega (soma) as contagens de voos de todos os snapshots
    traffic = (
        combined.groupby(["src_icao", "dst_icao"])["real_flight_count"]
        .sum()
        .reset_index()
    )

    print(
        f"[INFO] Agregação concluída: {len(traffic)} pares origem-destino únicos "
        f"a partir de {len(candidates)} snapshot(s)."
    )
    return traffic


def estimate_traffic_fallback(routes: pd.DataFrame) -> pd.Series:
    """Proxy de tráfego baseado na contagem de rotas."""
    print("[INFO] Usando fallback para estimar tráfego (contagem de rotas).")
    traffic = routes.groupby(["SourceAirportID", "DestAirportID"]).size()
    traffic.name = "traffic_estimate"
    return traffic


def export_graph(G: nx.DiGraph):
    """Exporta o grafo para GraphML e GEXF."""
    target_dir = OUTPUT_DIR / "grafos"
    target_dir.mkdir(parents=True, exist_ok=True)

    graphml_path = target_dir / "rede_aerea_global.graphml"
    gexf_path = target_dir / "rede_aerea_global.gexf"

    nx.write_graphml(G, str(graphml_path))
    nx.write_gexf(G, str(gexf_path))
