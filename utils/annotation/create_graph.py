import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Load JSON data
with open("/Users/aysanaghazadeh/experiments/generated_images/physical_sensation_com.json", "r") as f:
    image_data = json.load(f)

# Define full sensation hierarchy
SENSATION_HIERARCHY = {
    'Touch': {
        'Temperature Sensations': ['Freezing cold', 'Cool and refereshing', 'Comforting warmth', 'Intense heat'],
        'Texture Sensations': ['Softness', 'Silky smoothness', 'Stickiness', 'Roughness', 'Sharpness'],
        'Moisture/Dryness': ['Soaking wetness', 'Mistiness', 'Greasiness', 'Dryness'],
        'Movement and Body Position': ['High speed', 'Weightlessness', 'Heaviness', 'Tension', 'Vibration'],
        'Pain and relief': ['Sharp pain', 'Aching pain', 'Soothing and numbing relief']
    },
    'Smell': {
        'Fresh and clean': ['Fruit and vegtables', 'Nature', 'Clean'],
        'Rich and Food-Based': ['Drinks', 'Fruits and Vegtables', 'Foods', 'Bakery'],
        'Floral and sweets': ['Flowers', 'Fruits and vegtables'],
        'Erthy and musky': ['woody', 'Leather', 'Nature'],
        'Medicinal and Pungent': ['Medicine', 'Cleaning Products']
    },
    'Sight': ['Briliance', 'Clarity', 'Glow', 'Blur']
}

# Initialize graph
G = nx.DiGraph()

# Track levels for manual layout
levels = defaultdict(list)

# Step 1: Add full sensation hierarchy
def add_sensation_tree(G, parent, child):
    G.add_node(parent, type='top')
    levels[0].append(parent)

    if isinstance(child, dict):
        for mid, leaves in child.items():
            G.add_node(mid, type='mid')
            G.add_edge(parent, mid)
            levels[1].append(mid)
            for leaf in leaves:
                G.add_node(leaf, type='bottom')
                G.add_edge(mid, leaf)
                levels[2].append(leaf)
    elif isinstance(child, list):
        for leaf in child:
            G.add_node(leaf, type='bottom')
            G.add_edge(parent, leaf)
            levels[2].append(leaf)

for top, sub in SENSATION_HIERARCHY.items():
    add_sensation_tree(G, top, sub)

# Step 2: Add visual elements
for entry in image_data.values():
    sensation_path = [s.strip() for s in entry["sensation"]]
    visuals = [v.strip() for v in entry["visual_elements"]]
    if not sensation_path:
        continue
    bottom = sensation_path[-1]
    if not G.has_node(bottom):
        G.add_node(bottom, type='bottom')
        levels[2].append(bottom)

    for visual in visuals:
        if not G.has_node(visual):
            G.add_node(visual, type='visual')
            levels[3].append(visual)
        G.add_edge(bottom, visual)

# Step 3: Assign manual positions
pos = {}
y_gap = -2
x_gap = 2

for level, nodes in levels.items():
    for i, node in enumerate(sorted(set(nodes))):
        pos[node] = (i * x_gap, level * y_gap)

# Step 4: Define node colors
color_map = []
for node in G.nodes():
    t = G.nodes[node].get("type", "")
    if t == "top":
        color_map.append("red")
    elif t == "mid":
        color_map.append("orange")
    elif t == "bottom":
        color_map.append("gold")
    elif t == "visual":
        color_map.append("skyblue")
    else:
        color_map.append("gray")

# Step 5: Draw the graph
plt.figure(figsize=(40, 30))
nx.draw(
    G, pos,
    with_labels=True,
    node_color=color_map,
    node_size=1000,
    font_size=7,
    arrows=True,
    edge_color='gray'
)
plt.title("Sensation Hierarchy with Visual Elements (Manual Layout)", fontsize=20)
plt.axis("off")
plt.tight_layout()
plt.show()
