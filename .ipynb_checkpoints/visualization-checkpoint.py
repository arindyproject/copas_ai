import plotly.graph_objects as go
import numpy as np
from itertools import cycle

# Fungsi Pembantu (dibuat di luar agar tidak perlu didefinisikan ulang di dalam)
def hex_to_rgba(hex_color, alpha=0.5):
    """Convert hex color to rgba format"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

def calculate_neuron_positions(num_neurons, y_center=0, layer_spacing=0.6, min_spacing=0.4):
    """
    Hitung posisi neuron dengan spacing yang optimal untuk menghindari penumpukan.
    
    Args:
        num_neurons (int): Jumlah neuron dalam layer
        y_center (float): Posisi vertikal center layer
        layer_spacing (float): Jarak minimum antar layer
        min_spacing (float): Jarak minimum antar neuron dalam layer yang sama
    
    Returns:
        np.array: Posisi y untuk setiap neuron
    """
    if num_neurons == 1:
        return np.array([y_center])
    
    # Hitung total height yang dibutuhkan
    # Menggunakan min_spacing sebagai jarak dasar
    total_height = (num_neurons - 1) * min_spacing
    
    # Jika total_height terlalu kecil (misalnya, untuk satu neuron) atau jarak antar layer terlalu besar,
    # Kita dapat menyesuaikan scaling jika perlu, tapi untuk visualisasi, min_spacing sudah cukup.
    
    # Linspace akan membagi total_height secara merata
    return np.linspace(y_center + total_height/2, y_center - total_height/2, num_neurons)

def get_layer_bounds(layers):
    """
    Dapatkan batas atas dan bawah dari kumpulan layers.
    
    Args:
        layers (list): List of layers, setiap layer adalah list dari (x, y) positions
    
    Returns:
        tuple: (min_y, max_y)
    """
    all_y = []
    # layers adalah list of lists, di mana sublist berisi (x, y) tuple
    for layer in layers:
        if layer:
            all_y.extend([y for _, y in layer])
    
    if not all_y:
        return -1, 1
    
    return min(all_y), max(all_y)

# ----------------------------------------------------------------------

def plot_dynamic_multi_input_network(
    input_dims,
    hidden_dims_per_branch,
    combined_dims=[64],
    output_dim=5,
    figsize=(1600, 900),
    show_text=True
):
    """
    Visualisasi arsitektur neural network dengan jumlah input branch yang dinamis.
    Memperbaiki penentuan kolom x agar mendukung hidden layer yang tidak seimbang (panjang berbeda).
    Menampilkan label teks pada setiap neuron.
    """
    
    # Konfigurasi
    base_colors = ['#4E79A7', '#59A14F', '#E15759', '#F28E2B', '#76B7B2', '#B07AA1', '#9C755F', '#EDC948']
    color_cycle = cycle(base_colors)

    colors = {
        'combined': '#EDC948',
        'output': '#FF9DA7',
        'background': 'white',
        'text': 'black'
    }

    branch_colors = [next(color_cycle) for _ in input_dims]
    fig = go.Figure()
    neuron_positions = {}
    
    # Konstanta Jarak
    VERTICAL_SPACING = 2.5
    NEURON_SPACING = 0.6
    
    # --- Penentuan Posisi X (Kolom) ---
    # Posisi X: [Input Layer (x=0)] -> [Hidden Layers (x=1, 2, ...)] -> [Combined Layer] -> [Output Layer]
    
    max_hidden_len = max((len(h) for h in hidden_dims_per_branch), default=0)
    
    # Index kolom: 0 (Input) + max_hidden_len (Hidden) + len(combined_dims) (Combined) + 1 (Output)
    
    # Index X untuk layer spesifik:
    X_INPUT = 0
    X_HIDDEN_START = X_INPUT + 1
    
    # Jika ada hidden layer, combined layer mulai setelah hidden layer terpanjang
    # Jika tidak ada hidden layer (max_hidden_len=0), combined layer mulai setelah input
    X_COMBINED_START = X_HIDDEN_START + max_hidden_len
    
    # Index x untuk output layer
    X_OUTPUT = X_COMBINED_START + len(combined_dims) if combined_dims else X_HIDDEN_START + max_hidden_len
    
    # Membuat list posisi X yang digunakan
    x_indices = list(range(X_OUTPUT + 1))
    x_positions = {i: i for i in x_indices} # Mapping index ke posisi x

    # --- INPUT LAYER ---
    input_layer_positions = []
    # Menentukan Y center untuk setiap branch agar terpisah vertikal
    branch_y_centers = np.linspace(
        VERTICAL_SPACING * (len(input_dims) - 1) / 2,
        -VERTICAL_SPACING * (len(input_dims) - 1) / 2,
        len(input_dims)
    )

    x_idx = X_INPUT
    for idx, (input_dim, y_center) in enumerate(zip(input_dims, branch_y_centers)):
        branch_color = branch_colors[idx]
        # Menggunakan calculate_neuron_positions yang diperbaiki
        y_positions = calculate_neuron_positions(input_dim, y_center, min_spacing=NEURON_SPACING)
        layer_pos = []
        for j, y in enumerate(y_positions):
            label = f"In{idx+1}-{j+1}"
            fig.add_trace(go.Scatter(
                x=[x_positions[x_idx]], y=[y],
                mode='markers+text' if show_text else 'markers',
                marker=dict(size=28, color=branch_color, line=dict(width=2, color='black')),
                text=[label] if show_text else None,
                textposition="middle right",
                textfont=dict(size=12, color='black'),
                hovertext=f"Input {idx+1}<br>Neuron {j+1}",
                showlegend=False
            ))
            layer_pos.append((x_positions[x_idx], y))
        input_layer_positions.append(layer_pos)

    neuron_positions['input'] = input_layer_positions

    # --- HIDDEN PER BRANCH ---
    branch_positions = []
    
    # Iterasi melalui setiap branch
    for idx, (input_pos, hidden_dims) in enumerate(zip(input_layer_positions, hidden_dims_per_branch)):
        branch_color = branch_colors[idx]
        prev_layer = input_pos
        branch_layers = []
        
        # Iterasi melalui setiap hidden layer dalam branch
        for layer_idx, hdim in enumerate(hidden_dims):
            current_x_idx = X_HIDDEN_START + layer_idx
            
            # Memastikan tidak melebihi batas x_positions yang dialokasikan
            if current_x_idx not in x_positions: continue 
            
            # Center Y Layer dihitung berdasarkan layer sebelumnya
            y_center = np.mean([y for _, y in prev_layer])
            y_positions = calculate_neuron_positions(hdim, y_center, min_spacing=NEURON_SPACING)
            layer_pos = []
            
            # Gambar Neuron
            for j, y in enumerate(y_positions):
                label = f"H{idx+1}-{layer_idx+1}-{j+1}"
                fig.add_trace(go.Scatter(
                    x=[x_positions[current_x_idx]], y=[y],
                    mode='markers+text' if show_text else 'markers',
                    marker=dict(size=24, color=branch_color, line=dict(width=1.5, color='black')),
                    text=[label] if show_text else None,
                    textposition="middle right",
                    textfont=dict(size=11, color='black'),
                    hovertext=f"Branch {idx+1} Hidden {layer_idx+1}<br>Neuron {j+1}",
                    showlegend=False
                ))
                layer_pos.append((x_positions[current_x_idx], y))
                
                # Gambar Koneksi dari Layer Sebelumnya
                for x_prev, y_prev in prev_layer:
                    fig.add_trace(go.Scatter(
                        x=[x_prev, x_positions[current_x_idx]], y=[y_prev, y],
                        mode='lines',
                        line=dict(color=hex_to_rgba(branch_color, 0.3), width=0.5),
                        showlegend=False
                    ))
            
            prev_layer = layer_pos
            branch_layers.append(layer_pos)
        
        # Simpan posisi layer terakhir di branch ini
        branch_positions.append(branch_layers)

    # --- COMBINED LAYER (optional) ---
    combined_layer_output = []
    
    # Kumpulkan semua neuron terakhir dari setiap branch
    final_branch_neurons = []
    for branch_layers in branch_positions:
        if branch_layers:
            final_branch_neurons.extend(branch_layers[-1])
        # Kasus jika branch tidak punya hidden layer, ambil dari input
        elif not branch_layers and input_layer_positions[branch_positions.index(branch_layers)]:
             final_branch_neurons.extend(input_layer_positions[branch_positions.index(branch_layers)])
    
    
    # JIKA TIDAK ADA HIDDEN LAYER SAMA SEKALI
    if not final_branch_neurons:
         final_branch_neurons = [n for branch in input_layer_positions for n in branch]

    prev_layer = final_branch_neurons
    
    if combined_dims is not None and combined_dims:
        for layer_idx, cdim in enumerate(combined_dims):
            current_x_idx = X_COMBINED_START + layer_idx
            
            # Perhitungan Center Y: Berdasarkan posisi layer sebelumnya
            if prev_layer:
                min_y, max_y = get_layer_bounds([prev_layer])
                combined_center = (min_y + max_y) / 2
            else:
                combined_center = 0

            y_positions = calculate_neuron_positions(cdim, combined_center, min_spacing=NEURON_SPACING)
            comb_layer_pos = []
            
            # Gambar Neuron Combined
            for j, y in enumerate(y_positions):
                label = f"C{layer_idx+1}-{j+1}"
                fig.add_trace(go.Scatter(
                    x=[x_positions[current_x_idx]], y=[y],
                    mode='markers+text' if show_text else 'markers',
                    marker=dict(size=26, color=colors['combined'], line=dict(width=2, color='black')),
                    text=[label] if show_text else None,
                    textposition="middle right",
                    textfont=dict(size=12, color='black'),
                    hovertext=f"Combined Layer {layer_idx+1}<br>Neuron {j+1}",
                    showlegend=False
                ))
                comb_layer_pos.append((x_positions[current_x_idx], y))
                
                # Gambar Koneksi dari Layer Sebelumnya
                for x_prev, y_prev in prev_layer:
                    fig.add_trace(go.Scatter(
                        x=[x_prev, x_positions[current_x_idx]], y=[y_prev, y],
                        mode='lines',
                        line=dict(color=hex_to_rgba(colors['combined'], 0.3), width=0.5),
                        showlegend=False
                    ))
            
            prev_layer = comb_layer_pos
            combined_layer_output = comb_layer_pos # Update layer output terakhir dari combined

    # --- OUTPUT LAYER ---
    # Layer terakhir sebelum output adalah combined layer terakhir (jika ada) atau final branch neurons
    prev_layer_for_output = combined_layer_output if combined_layer_output else prev_layer
    
    # Penentuan Center Y Output Layer
    output_center = np.mean([y for _, y in prev_layer_for_output]) if prev_layer_for_output else 0
    y_positions = calculate_neuron_positions(output_dim, output_center, min_spacing=NEURON_SPACING)
    
    # Gambar Neuron Output
    for j, y in enumerate(y_positions):
        label = f"Out-{j+1}"
        fig.add_trace(go.Scatter(
            x=[x_positions[X_OUTPUT]], y=[y],
            mode='markers+text' if show_text else 'markers',
            marker=dict(size=28, color=colors['output'], line=dict(width=2, color='black')),
            text=[label] if show_text else None,
            textposition="middle right",
            textfont=dict(size=12, color='black'),
            hovertext=f"Output Neuron {j+1}",
            showlegend=False
        ))
        
        # Gambar Koneksi ke Output
        for x_prev, y_prev in prev_layer_for_output:
            fig.add_trace(go.Scatter(
                x=[x_prev, x_positions[X_OUTPUT]], y=[y_prev, y],
                mode='lines',
                line=dict(color=hex_to_rgba(colors['output'], 0.3), width=0.5),
                showlegend=False
            ))

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text="🧠 Neural Network Architecture (Multi-Branch, Dynamic)",
            x=0.5,
            font=dict(size=20, color='black', family="Arial Black")
        ),
        width=figsize[0],
        height=figsize[1],
        plot_bgcolor='white',
        paper_bgcolor='white',
        # Menampilkan X axis untuk debugging, nonaktifkan kembali jika tidak perlu
        xaxis=dict(visible=False, range=[min(x_indices) - 0.5, max(x_indices) + 1.5]), 
        yaxis=dict(visible=False),
        showlegend=False
    )

    fig.show()


# Contoh Penggunaan dengan Hidden Layer yang tidak seimbang:
# input_dims: [3, 4, 5] (3 branch input)
# hidden_dims_per_branch: 
# Branch 1: [4] (1 hidden layer)
# Branch 2: [3, 4] (2 hidden layer)
# Branch 3: [2, 4, 5] (3 hidden layer) -> Ini adalah branch terpanjang
