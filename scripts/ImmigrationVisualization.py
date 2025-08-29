import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

def load_data():
    SCRIPT_DIR = os.path.dirname(__file__)
    FILE_PATH = os.path.join(SCRIPT_DIR, '../data/2024Immigration.csv')
    
    # Read with correct header and skip the unwanted row
    df = pd.read_csv(FILE_PATH, header=0, skiprows=[1])
    
    # Drop unnecessary unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    return df

REGIONS = ['Eastern Asia', 'Central Asia', 'Western Asia', 'South-Eastern Asia', 'Southern Asia',
           'Eastern Europe', 'Northern Europe', 'Southern Europe', 'Western Europe']

def plot_region_bubbles(df,
                    regions,
                    region_col='Region, development group of destination',
                    total_col='WORLD',
                    min_flow=0):
    # Normalize region names
    df = df.copy()
    df[region_col] = df[region_col].astype(str).str.strip().str.upper()
    regions_upper = [r.upper() for r in regions]

    # Convert numeric columns
    numeric_cols = [c for c in df.columns if c != region_col]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Filter only exact matches for top-level regions
    df_filtered = df[df[region_col].isin(regions_upper)].copy()

    if df_filtered.empty:
        raise ValueError("No matching rows found. Check your region names.")

    # Keep relevant columns
    cols = [region_col, total_col] + [c for c in df.columns if c.upper() in regions_upper]
    df_filtered = df_filtered[cols]

    # Aggregate duplicates
    df_filtered = df_filtered.groupby(region_col, as_index=False).sum()

    # Determine target columns for edges
    available_cols = [c for c in df_filtered.columns if c != region_col and c.upper() in regions_upper]

    # Build graph
    G = nx.DiGraph()
    for _, row in df_filtered.iterrows():
        G.add_node(row[region_col], size=row[total_col])
    edges = []
    for _, row in df_filtered.iterrows():
        src = row[region_col]
        for tgt in available_cols:
            if src != tgt:
                weight = row[tgt]
                if weight > min_flow:
                    edges.append((src, tgt, weight))


    # Layout
    pos = nx.spring_layout(G, k=1.2, seed=42)
    node_sizes = [max(G.nodes[n]['size'] / 1e6, 1000) for n in G.nodes()]
    edge_weights = [edata['weight'] for _, _, edata in G.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1
    widths = [(w / max_weight) * 3 for w in edge_weights]

    
    # Draw
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='orange', alpha=0.85)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
    if edge_weights:
        nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=18, edge_color='gray', width=widths)

    plt.title("Immigration Stocks & Flows Between Major Regions", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
df = load_data()
plot_region_bubbles(df=df, regions=REGIONS)
