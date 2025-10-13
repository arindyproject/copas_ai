import plotly.graph_objects as go
import numpy as np
from itertools import cycle

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
    total_height = (num_neurons - 1) * min_spacing
    
    # Jika total_height terlalu kecil, gunakan spacing yang lebih besar
    if total_height < layer_spacing:
        min_spacing = layer_spacing / (num_neurons - 1) if num_neurons > 1 else layer_spacing
    
    total_height = (num_neurons - 1) * min_spacing
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
    for layer in layers:
        if layer:  # Jika layer tidak kosong
            all_y.extend([y for _, y in layer])
    
    if not all_y:
        return -1, 1
    
    return min(all_y), max(all_y)

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
    Menampilkan label teks pada setiap neuron.
    """
    import plotly.graph_objects as go
    import numpy as np
    from itertools import cycle

    def hex_to_rgba(hex_color, alpha=0.5):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'

    def calculate_neuron_positions(num_neurons, y_center=0, min_spacing=0.6):
        if num_neurons == 1:
            return np.array([y_center])
        total_height = (num_neurons - 1) * min_spacing
        return np.linspace(y_center + total_height/2, y_center - total_height/2, num_neurons)

    def get_layer_bounds(layers):
        all_y = [y for layer in layers for _, y in layer]
        return (min(all_y), max(all_y)) if all_y else (-1, 1)

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

    # --- Hitung total kolom ---
    total_columns = 1  # input
    total_columns += max(len(h) for h in hidden_dims_per_branch)
    if combined_dims is not None:
        total_columns += len(combined_dims) + 1
    total_columns += 1  # output

    x_positions = np.linspace(0, total_columns - 1, total_columns)

    VERTICAL_SPACING = 2.5
    NEURON_SPACING = 0.6

    # --- INPUT LAYER ---
    input_layer_positions = []
    branch_y_centers = np.linspace(
        VERTICAL_SPACING * (len(input_dims) - 1) / 2,
        -VERTICAL_SPACING * (len(input_dims) - 1) / 2,
        len(input_dims)
    )

    x_idx = 0
    for idx, (input_dim, y_center) in enumerate(zip(input_dims, branch_y_centers)):
        branch_color = branch_colors[idx]
        y_positions = calculate_neuron_positions(input_dim, y_center, NEURON_SPACING)
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
    for idx, (input_pos, hidden_dims) in enumerate(zip(input_layer_positions, hidden_dims_per_branch)):
        branch_color = branch_colors[idx]
        prev_layer = input_pos
        branch_layers = []
        for layer_idx, hdim in enumerate(hidden_dims):
            if layer_idx + 1 >= len(x_positions): break
            y_center = np.mean([y for _, y in prev_layer])
            y_positions = calculate_neuron_positions(hdim, y_center, NEURON_SPACING)
            layer_pos = []
            for j, y in enumerate(y_positions):
                label = f"H{idx+1}-{layer_idx+1}-{j+1}"
                fig.add_trace(go.Scatter(
                    x=[x_positions[layer_idx+1]], y=[y],
                    mode='markers+text' if show_text else 'markers',
                    marker=dict(size=24, color=branch_color, line=dict(width=1.5, color='black')),
                    text=[label] if show_text else None,
                    textposition="middle right",
                    textfont=dict(size=11, color='black'),
                    hovertext=f"Branch {idx+1} Hidden {layer_idx+1}<br>Neuron {j+1}",
                    showlegend=False
                ))
                layer_pos.append((x_positions[layer_idx+1], y))
                for x_prev, y_prev in prev_layer:
                    fig.add_trace(go.Scatter(
                        x=[x_prev, x_positions[layer_idx+1]], y=[y_prev, y],
                        mode='lines',
                        line=dict(color=hex_to_rgba(branch_color, 0.3), width=0.5),
                        showlegend=False
                    ))
            prev_layer = layer_pos
            branch_layers.append(layer_pos)
        branch_positions.append(branch_layers)

    # --- COMBINED LAYER (optional) ---
    if combined_dims is not None:
        combined_x_idx = len(hidden_dims_per_branch[0]) + 1
        final_branch_neurons = [n for b in branch_positions for n in (b[-1] if b else [])]
        if final_branch_neurons:
            min_y, max_y = get_layer_bounds([final_branch_neurons])
            combined_center = (min_y + max_y) / 2
        else:
            combined_center = 0

        comb_y_positions = calculate_neuron_positions(combined_dims[0], combined_center)
        comb_layer_pos = []
        for j, y in enumerate(comb_y_positions):
            label = f"C-{j+1}"
            fig.add_trace(go.Scatter(
                x=[x_positions[combined_x_idx]], y=[y],
                mode='markers+text' if show_text else 'markers',
                marker=dict(size=26, color=colors['combined'], line=dict(width=2, color='black')),
                text=[label] if show_text else None,
                textposition="middle right",
                textfont=dict(size=12, color='black'),
                hovertext=f"Combined Neuron {j+1}",
                showlegend=False
            ))
            comb_layer_pos.append((x_positions[combined_x_idx], y))
            for x_prev, y_prev in final_branch_neurons:
                fig.add_trace(go.Scatter(
                    x=[x_prev, x_positions[combined_x_idx]], y=[y_prev, y],
                    mode='lines',
                    line=dict(color=hex_to_rgba(colors['combined'], 0.3), width=0.5),
                    showlegend=False
                ))
        prev_layer = comb_layer_pos
        start_output_x = combined_x_idx + len(combined_dims)
    else:
        prev_layer = [n for b in branch_positions for n in (b[-1] if b else [])]
        start_output_x = len(hidden_dims_per_branch[0]) + 1

    # --- OUTPUT LAYER ---
    output_center = np.mean([y for _, y in prev_layer]) if prev_layer else 0
    y_positions = calculate_neuron_positions(output_dim, output_center)
    for j, y in enumerate(y_positions):
        label = f"Out-{j+1}"
        fig.add_trace(go.Scatter(
            x=[x_positions[start_output_x]], y=[y],
            mode='markers+text' if show_text else 'markers',
            marker=dict(size=28, color=colors['output'], line=dict(width=2, color='black')),
            text=[label] if show_text else None,
            textposition="middle right",
            textfont=dict(size=12, color='black'),
            hovertext=f"Output Neuron {j+1}",
            showlegend=False
        ))
        for x_prev, y_prev in prev_layer:
            fig.add_trace(go.Scatter(
                x=[x_prev, x_positions[start_output_x]], y=[y_prev, y],
                mode='lines',
                line=dict(color=hex_to_rgba(colors['output'], 0.3), width=0.5),
                showlegend=False
            ))

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text="🧠 Neural Network Architecture (with Labels)",
            x=0.5,
            font=dict(size=20, color='black', family="Arial Black")
        ),
        width=figsize[0],
        height=figsize[1],
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )

    fig.show()
