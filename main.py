import warnings
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw

import utils

def load_airports(filepath: str | None = None) -> pd.DataFrame:
    """Carrega o dataset de aeroportos do OpenFlights."""
    cols = ["AirportID", "Name", "City", "Country", "IATA", "ICAO",
            "Latitude", "Longitude", "Altitude", "Timezone", "DST",
            "TzDatabase", "Type", "Source"]

    if filepath is None:
        filepath = utils.DATA_DIR / "airports.dat"

    if filepath.exists():
        print(f"[INFO] Carregando aeroportos de {filepath}")
        df = pd.read_csv(filepath, header=None, names=cols, na_values="\\N")
    else:
        print(f"[INFO] Baixando aeroportos de {utils.AIRPORTS_URL}")
        df = pd.read_csv(utils.AIRPORTS_URL, header=None, names=cols, na_values="\\N")
        df.to_csv(filepath, index=False, header=False)

    df = df.dropna(subset=["IATA", "ICAO"])
    df = df[(df["IATA"].str.len() == 3) & (df["ICAO"].str.len() == 4)]
    df = df[["AirportID", "Name", "IATA", "ICAO", "Latitude", "Longitude", "Country"]].copy()
    df["AirportID"] = df["AirportID"].astype(int)
    return df.reset_index(drop=True)


def load_routes(filepath: str | None = None) -> pd.DataFrame:
    """Carrega o dataset de rotas do OpenFlights."""
    cols = ["Airline", "AirlineID", "SourceAirport", "SourceAirportID",
            "DestAirport", "DestAirportID", "Codeshare", "Stops", "Equipment"]

    if filepath is None:
        filepath = utils.DATA_DIR / "routes.dat"

    if filepath.exists():
        print(f"[INFO] Carregando rotas de {filepath}")
        df = pd.read_csv(filepath, header=None, names=cols, na_values="\\N")
    else:
        print(f"[INFO] Baixando rotas de {utils.ROUTES_URL}")
        df = pd.read_csv(utils.ROUTES_URL, header=None, names=cols, na_values="\\N")
        df.to_csv(filepath, index=False, header=False)

    df["SourceAirportID"] = pd.to_numeric(df["SourceAirportID"], errors="coerce")
    df["DestAirportID"] = pd.to_numeric(df["DestAirportID"], errors="coerce")
    df = df.dropna(subset=["SourceAirportID", "DestAirportID"])
    return df.astype({"SourceAirportID": int, "DestAirportID": int}).reset_index(drop=True)


def filter_routes(routes: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Filtra rotas com aeroportos inexistentes na base limpa."""
    valid_ids = set(airports["AirportID"])
    mask = routes["SourceAirportID"].isin(valid_ids) & routes["DestAirportID"].isin(valid_ids)
    return routes[mask].reset_index(drop=True)


def enrich_routes(routes: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Adiciona distância e estimativa de tráfego às rotas.

    Estratégia de pesos:
      1. Tenta carregar o snapshot local do OpenSky (data/opensky_peak_snapshot.json).
         Se disponível, faz merge dos voos reais por par ICAO com as rotas.
      2. Caso contrário, usa o fallback (contagem de companhias por rota).
    """
    lookup = airports.set_index("AirportID")
    id_to_lat = lookup["Latitude"].to_dict()
    id_to_lon = lookup["Longitude"].to_dict()
    id_to_iata = lookup["IATA"].to_dict()
    id_to_icao = lookup["ICAO"].to_dict()

    routes = routes.copy()
    routes["SrcLat"] = routes["SourceAirportID"].map(id_to_lat)
    routes["SrcLon"] = routes["SourceAirportID"].map(id_to_lon)
    routes["DstLat"] = routes["DestAirportID"].map(id_to_lat)
    routes["DstLon"] = routes["DestAirportID"].map(id_to_lon)
    routes["SrcIATA"] = routes["SourceAirportID"].map(id_to_iata)
    routes["DstIATA"] = routes["DestAirportID"].map(id_to_iata)
    routes["SrcICAO"] = routes["SourceAirportID"].map(id_to_icao)
    routes["DstICAO"] = routes["DestAirportID"].map(id_to_icao)

    print("[INFO] Calculando distâncias geodésicas…")
    routes["distance_km"] = routes.apply(
        lambda r: utils.haversine(r["SrcLat"], r["SrcLon"], r["DstLat"], r["DstLon"]), axis=1
    )

    # --- Integração de pesos ---
    real_traffic = utils.load_all_opensky_snapshots()

    if not real_traffic.empty:
        print("[INFO] Integrando pesos do snapshot OpenSky…")
        # Merge pelo par ICAO (origem, destino)
        routes = routes.merge(
            real_traffic,
            left_on=["SrcICAO", "DstICAO"],
            right_on=["src_icao", "dst_icao"],
            how="left",
        )
        routes["real_flight_count"] = routes["real_flight_count"].fillna(0).astype(int)

        # Fallback: contagem de companhias para rotas sem cobertura OpenSky
        fallback = utils.estimate_traffic_fallback(routes)
        routes = routes.merge(
            fallback.reset_index(), on=["SourceAirportID", "DestAirportID"], how="left"
        )
        routes["traffic_estimate"] = (
            routes["real_flight_count"] + routes["traffic_estimate"].fillna(1)
        ).astype(int)

        matched = (routes["real_flight_count"] > 0).sum()
        print(f"[INFO] {matched} rotas enriquecidas com dados reais do OpenSky.")

        # Limpa colunas auxiliares do merge
        routes = routes.drop(columns=["src_icao", "dst_icao"], errors="ignore")
    else:
        traffic = utils.estimate_traffic_fallback(routes)
        routes = routes.merge(
            traffic.reset_index(), on=["SourceAirportID", "DestAirportID"], how="left"
        )
        routes["traffic_estimate"] = routes["traffic_estimate"].fillna(1).astype(int)
        routes["real_flight_count"] = 0

    return routes


# ############################################################################
#                     ETAPA 2 – CRIAÇÃO DO GRAFO
# ############################################################################

def build_graph(airports: pd.DataFrame, routes: pd.DataFrame) -> nx.DiGraph:
    """Constrói o grafo direcionado com múltiplos pesos."""
    G = nx.DiGraph()
    for _, row in airports.iterrows():
        G.add_node(row["IATA"], name=row["Name"], icao=row["ICAO"],
                   latitude=row["Latitude"], longitude=row["Longitude"], country=row["Country"])

    for _, row in routes.iterrows():
        src, dst = row["SrcIATA"], row["DstIATA"]
        if G.has_edge(src, dst):
            G[src][dst]["traffic_estimate"] += row["traffic_estimate"]
        else:
            G.add_edge(src, dst, distance_km=round(row["distance_km"], 2),
                       traffic_estimate=int(row["traffic_estimate"]))
    return G


# ############################################################################
#                     ETAPA 3 – ANÁLISE TOPOLÓGICA
# ############################################################################

def topology_report(G: nx.DiGraph):
    """Gera relatório básico da topologia."""
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    avg_degree = np.mean([d for _, d in G.degree()])

    print("\n" + "="*50)
    print("      RELATÓRIO TOPOLÓGICO DA REDE")
    print("="*50)
    print(f"Nós: {n_nodes:,} | Arestas: {n_edges:,}")
    print(f"Grau Médio: {avg_degree:.2f}")
    print(f"Componente Gigante: {len(largest_wcc):,} ({len(largest_wcc)/n_nodes*100:.1f}%)")
    print("="*50 + "\n")


def plot_degree_distribution(G: nx.DiGraph):
    """Ajusta lei de potência e gera gráfico log-log."""
    degrees = np.array([d for _, d in G.degree() if d > 0])
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

    fig, ax = plt.subplots()
    bins = np.logspace(np.log10(degrees.min()), np.log10(degrees.max()), 40)
    counts, edges = np.histogram(degrees, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.scatter(centers[counts > 0], counts[counts > 0], alpha=0.7, label="Dados")

    x_fit = np.logspace(np.log10(fit.power_law.xmin), np.log10(degrees.max()), 100)
    C = counts[counts > 0][-1] * (centers[counts > 0][-1] ** fit.power_law.alpha)
    ax.plot(x_fit, C * x_fit ** (-fit.power_law.alpha), "r--", label=f"γ = {fit.power_law.alpha:.2f}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Grau (k)"); ax.set_ylabel("P(k)")
    ax.legend()
    fig.savefig(utils.OUTPUT_DIR / "distribuicao_grau_loglog.png")
    plt.close(fig)


def centrality_analysis(G: nx.DiGraph):
    """Calcula e exibe as principais centralidades."""
    print("[INFO] Calculando centralidades…")
    dc = nx.degree_centrality(G)
    bc = nx.betweenness_centrality(G, weight="distance_km")
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000, weight="traffic_estimate")
    except:
        ec = nx.eigenvector_centrality_numpy(G, weight="traffic_estimate")

    df = pd.DataFrame({
        "IATA": list(dc.keys()),
        "Name": [G.nodes[n]["name"] for n in dc],
        "Degree": [G.degree(n) for n in dc],
        "DC": list(dc.values()), "BC": list(bc.values()), "EC": list(ec.values())
    })

    for col in ["DC", "BC", "EC"]:
        print(f"\nTop 10 {col}:")
        print(df.nlargest(10, col)[["IATA", "Name", col]])

    df.to_csv(utils.OUTPUT_DIR / "centralidades.csv", index=False)


def main():
    utils.setup_directories()
    airports = load_airports()
    routes = filter_routes(load_routes(), airports)

    # O enrich_routes carrega o snapshot local do OpenSky automaticamente.
    # Para coletar um novo snapshot, execute: python collect_snapshot.py
    routes = enrich_routes(routes, airports)

    G = build_graph(airports, routes)
    utils.export_graph(G)

    topology_report(G)
    plot_degree_distribution(G)
    centrality_analysis(G)
    print("\n✅ Fase 1 Concluída!")


if __name__ == "__main__":
    main()
