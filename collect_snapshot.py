import argparse
import utils

def main():
    parser = argparse.ArgumentParser(
        description="Coleta snapshot de voos da OpenSky Network."
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=2,
        help="Janela de tempo em horas (máx. 2 h para API gratuita). Padrão: 2.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista todos os snapshots disponíveis na pasta data/.",
    )
    args = parser.parse_args()

    # Modo listagem
    if args.list:
        snapshots = utils.list_snapshots()
        if not snapshots:
            print("Nenhum snapshot encontrado na pasta data/.")
        else:
            print(f"📁 {len(snapshots)} snapshot(s) disponível(is):\n")
            for s in snapshots:
                size_kb = s.stat().st_size / 1024
                print(f"   {s.name}  ({size_kb:.1f} KB)")
        return

    hours = min(args.hours, 2)  # API gratuita limita a 2 h
    snapshot_path = utils.collect_opensky_snapshot(hours_back=hours)

    if snapshot_path:
        df = utils.load_opensky_snapshot(path=snapshot_path)
        if not df.empty:
            print(f"\n Resumo do Snapshot ({snapshot_path.name}):")
            print(f"   Pares origem-destino únicos: {len(df)}")
            print(f"   Total de voos registrados:   {df['real_flight_count'].sum()}")
            top = df.nlargest(5, "real_flight_count")
            print(f"\n   Top 5 rotas mais movimentadas:")
            for _, row in top.iterrows():
                print(f"     {row['src_icao']} → {row['dst_icao']}  ({row['real_flight_count']} voos)")

        all_snaps = utils.list_snapshots()
        print(f"\n Total de snapshots acumulados: {len(all_snaps)}")
    else:
        print("\n❌ Não foi possível coletar o snapshot. Verifique sua conexão.")


if __name__ == "__main__":
    main()
