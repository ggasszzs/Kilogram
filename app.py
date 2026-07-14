import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
import os
import time

# Konfigurasi Halaman (Sidebar disembunyikan secara default)
st.set_page_config(page_title="Kilo POS System", page_icon="🛍️", layout="wide", initial_sidebar_state="collapsed")

# Inisialisasi Session State untuk Keranjang Belanja
if 'cart' not in st.session_state:
    st.session_state.cart = []

def add_to_cart(product):
    if product not in st.session_state.cart:
        st.session_state.cart.append(product)
        st.toast(f"Berhasil menambahkan {product} ke keranjang!", icon="✨")

def clear_cart():
    st.session_state.cart = []

# --- CSS "WOW" (Premium, Dinamis & Elegan) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Animasi Mengambang (Floating) */
    @keyframes floating {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    /* Animasi Cahaya (Glow) */
    @keyframes glow {
        0% { box-shadow: 0 0 15px rgba(59, 130, 246, 0.5); }
        50% { box-shadow: 0 0 30px rgba(59, 130, 246, 0.8), 0 0 10px rgba(139, 92, 246, 0.5); }
        100% { box-shadow: 0 0 15px rgba(59, 130, 246, 0.5); }
    }
    
    /* Header/Hero Section Premium */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #8b5cf6 100%);
        padding: 50px 30px;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.3);
        margin-bottom: 40px;
        color: white !important;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    /* Aksen cahaya di belakang banner */
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: spin 15s linear infinite;
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    
    .hero-title {
        font-size: 56px;
        font-weight: 800;
        margin-bottom: 10px;
        color: #ffffff !important;
        letter-spacing: -1px;
        text-shadow: 0 4px 15px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-size: 20px;
        font-weight: 300;
        color: rgba(255,255,255,0.9) !important;
        position: relative;
        z-index: 1;
    }
    
    /* Card Styles Transparan Super Premium */
    .glass-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 24px;
        padding: 35px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        margin-bottom: 30px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        backdrop-filter: blur(10px);
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    
    /* Recommendation Item Card */
    .rec-item {
        background: linear-gradient(145deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 15px;
        border-left: 6px solid #10b981;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
    }
    .rec-item:hover {
        transform: scale(1.02) translateX(5px);
        border-left: 6px solid #059669;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.2);
    }
    .rec-name {
        font-size: 20px;
        font-weight: 800;
        color: var(--text-color);
    }
    .rec-metrics {
        font-size: 14px;
        color: #888;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Cart Styling */
    .cart-item {
        font-size: 17px;
        padding: 16px 0;
        border-bottom: 1px dashed rgba(128,128,128,0.3);
        font-weight: 600;
        transition: background 0.3s ease;
    }
    .cart-item:hover {
        background: rgba(128,128,128,0.05);
        border-radius: 8px;
        padding: 16px 10px;
    }
    
    /* Kustomisasi Selectbox / Dropdown */
    .stSelectbox > div > div {
        border-radius: 16px !important;
        padding: 8px !important;
        border: 2px solid rgba(128,128,128,0.2) !important;
        background-color: var(--secondary-background-color) !important;
        font-size: 18px !important;
        transition: all 0.3s ease;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Tombol Utama (Button) */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3) !important;
    }
    # Teks Global Responsif (Opacity daripada abu-abu mati)
    .text-muted {
        color: var(--text-color);
        opacity: 0.6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_rules():
    file_path = "data/dynamic_rules.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # Fallback if csv not yet created
        df = pd.read_excel("data/Rekomendasi_CrossSell.xlsx")
    
    def clean_item(x):
        if isinstance(x, str):
            x = x.replace("frozenset({", "").replace("})", "")
            x = x.replace("(", "").replace(")", "")
            x = x.replace("'", "").replace('"', "")
            return x.strip()
        return str(x)
        
    df['antecedents'] = df['antecedents'].apply(clean_item)
    df['consequents'] = df['consequents'].apply(clean_item)
    if not df.empty:
        df = df.sort_values(by='lift', ascending=False).reset_index(drop=True)
    return df

@st.cache_data
def load_all_products():
    file_path_matrix = "data/Matriks_Biner_Transaksi.parquet"
    if os.path.exists(file_path_matrix):
        df_cols = pd.read_parquet(file_path_matrix)
    else:
        df_cols = pd.read_excel("data/Matriks_Biner_Transaksi.xlsx", nrows=0)
    return sorted(df_cols.columns.tolist())

def process_transaction_and_retrain(cart_items):
    matrix_path = "data/Matriks_Biner_Transaksi.parquet"
    rules_path = "data/dynamic_rules.csv"
    
    # 1. Load Matrix
    df_matrix = pd.read_parquet(matrix_path)
    
    # 2. Create new transaction row
    new_row = {col: False for col in df_matrix.columns}
    for item in cart_items:
        if item in new_row:
            new_row[item] = True
            
    # Append row and ensure boolean type
    df_matrix = pd.concat([df_matrix, pd.DataFrame([new_row])], ignore_index=True)
    df_matrix = df_matrix.astype(bool)
    
    # 3. Save updated Matrix
    df_matrix.to_parquet(matrix_path, index=False)
    
    # 4. Retrain AI Apriori (Low Memory Mode to prevent crashes)
    freq_items = apriori(df_matrix, min_support=0.001, use_colnames=True, low_memory=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.05)
    
    # 5. Save Rules
    rules.to_csv(rules_path, index=False)
    
    # 6. Invalidate Streamlit Cache so UI updates instantly
    load_rules.clear()

# Load Data
try:
    with st.spinner("Memuat kecerdasan buatan..."):
        df_rules = load_rules()
        all_products = load_all_products()
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

# --- SIDEBAR INTERAKTIF ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081986.png", width=60)
    st.markdown("<h3>Kilo POS</h3><p class='text-muted' style='font-size:12px; margin-top:-10px;'>Sistem Kasir Pintar AI</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        from streamlit_option_menu import option_menu
        menu_selection = option_menu(
            menu_title=None,
            options=["Kasir & Rekomendasi", "Visualisasi Data", "Database Aturan"],
            icons=['cart-plus-fill', 'pie-chart-fill', 'database-fill-gear'],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#3b82f6", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "font-weight":"600"},
                "nav-link-selected": {"background-color": "#3b82f6", "color": "white"},
            }
        )
    except ImportError:
        menu_selection = st.radio(
            "Menu Utama",
            options=["Kasir & Rekomendasi", "Visualisasi Data", "Database Aturan"],
            label_visibility="collapsed"
        )


# Mengisi ruang kosong di Sidebar
st.sidebar.markdown("### 👤 Info Profil")
st.sidebar.markdown("**Role:** Administrator")
st.sidebar.markdown("**Cabang:** Pusat")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Panduan:** Klik ikon panah di pojok kiri atas layar untuk melipat (menyembunyikan) atau memunculkan menu ini.")

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='text-muted' style='text-align:center; font-size:12px;'>© 2026 Kilo POS System<br>Powered by Apriori Machine Learning</p>", unsafe_allow_html=True)

# --- HERO BANNER ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Kilo Point of Sale</div>
    <div class="hero-subtitle">Sistem Kasir Pintar Berbasis Algoritma Apriori AI</div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# HALAMAN 1: KASIR & REKOMENDASI CROSS-SELLING
# ==============================================================================
if menu_selection == "Kasir & Rekomendasi":
    
    # Layout Utama
    col_main, col_spacer, col_sidebar = st.columns([2.5, 0.2, 1.5])
    
    with col_main:
        with st.container():
            st.markdown("<h3>🏷️ Cari & Tambah Produk</h3>", unsafe_allow_html=True)
            st.markdown("<p class='text-muted' style='font-size:14px; margin-bottom:15px;'>Ketik nama produk untuk mencari lebih cepat.</p>", unsafe_allow_html=True)
            
            selected_product = st.selectbox("", ["-- Pilih Menu Untuk Ditambahkan --"] + all_products, label_visibility="collapsed")
            
            if selected_product != "-- Pilih Menu Untuk Ditambahkan --":
                if st.button(f"➕ Tambahkan '{selected_product}' ke Keranjang", type="primary", use_container_width=True):
                    add_to_cart(selected_product)
                    st.rerun()
        
        # Area Rekomendasi Dinamis
        st.markdown("<h2 style='margin-top:30px; margin-bottom:20px;'>🔥 Rekomendasi Pintar (Cross-Selling)</h2>", unsafe_allow_html=True)
        
        if len(st.session_state.cart) > 0:
            recs = df_rules[df_rules['antecedents'].isin(st.session_state.cart)].copy()
            recs = recs[~recs['consequents'].isin(st.session_state.cart)]
            
            if len(recs) > 0:
                recs = recs.sort_values(by='lift', ascending=False).drop_duplicates(subset=['consequents'])
                
                # Tampilkan rekomendasi
                for _, row in recs.head(5).iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="rec-item">
                            <div>
                                <div class="rec-name">➕ {row['consequents']}</div>
                                <div class="rec-metrics">
                                    🌟 <span style='color:#059669; font-weight:700;'>Confidence: {row['confidence']*100:.1f}%</span> 
                                    &nbsp;|&nbsp; 
                                    🚀 <span style='color:#ef4444; font-weight:700;'>Lift: {row['lift']:.2f}</span>
                                    <br><span style='font-size:12px; opacity:0.7;'>Karena pelanggan membeli: {row['antecedents']}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Tombol asli streamlit disejajarkan menggunakan kolom kecil
                        _, btn_col = st.columns([3, 1])
                        with btn_col:
                            if st.button("➕ Tawarkan", key=f"btn_{row['consequents']}", use_container_width=True):
                                add_to_cart(row['consequents'])
                                st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("💡 Belum ada rekomendasi Cross-Selling yang cocok untuk menu di keranjang Anda.")
        else:
            # Jika kosong, tampilkan "Trending Items" untuk mengisi ruang kosong
            with st.container():
                st.markdown("""
                <div style="text-align:center; padding: 50px 30px;">
                    <h2 style="margin-bottom: 10px; font-weight:800;">Keranjang Masih Kosong</h2>
                    <p class="text-muted" style="font-size:18px; margin-bottom: 40px;">Sistem AI Apriori sedang menunggu pesanan Anda untuk memberikan rekomendasi cerdas (Cross-Selling).</p>
                    <div style="background: linear-gradient(90deg, transparent, rgba(128,128,128,0.3), transparent); height: 2px; width: 100%; margin-bottom: 35px;"></div>
                    <h4 style="font-weight:700; margin-bottom: 20px; letter-spacing:1px; color:#3b82f6;">🔥 MENU TERPOPULER HARI INI</h4>
                </div>
                """, unsafe_allow_html=True)
            
            top_trending = df_rules.sort_values(by='support', ascending=False)['antecedents'].unique()[:3]
            cols = st.columns(3)
            for i, trending in enumerate(top_trending):
                with cols[i]:
                    st.markdown(f"""
                    <div style='
                        background: linear-gradient(145deg, rgba(59, 130, 246, 0.05) 0%, rgba(59, 130, 246, 0.15) 100%); 
                        padding:25px 15px; 
                        border-radius:20px; 
                        font-weight:700; 
                        font-size:18px;
                        color:var(--text-color); 
                        border: 1px solid rgba(59, 130, 246, 0.3); 
                        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.1);
                        text-align:center;
                    '>
                        <div style="font-size:30px; margin-bottom:10px;">🏆</div>
                        {trending}
                    </div>
                    """, unsafe_allow_html=True)

    with col_sidebar:
        # Panel Keranjang Belanja
        with st.container():
            st.markdown("<h2 style='margin-top:0; border-bottom: 3px solid #3b82f6; padding-bottom: 15px;'>🛒 Keranjang</h2>", unsafe_allow_html=True)
        
        if len(st.session_state.cart) == 0:
            st.markdown("""
            <div style="text-align:center; padding: 60px 0; opacity: 0.5;">
                <div style="font-size: 70px; margin-bottom: 15px;">🛒</div>
                <p style="font-weight: 600; font-size:18px;">Belum ada pesanan</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, item in enumerate(st.session_state.cart):
                st.markdown(f"<div class='cart-item'>{i+1}. {item}</div>", unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Action Buttons
            if st.button("💳 Proses Pembayaran", type="primary", use_container_width=True):
                with st.spinner("💾 Menyimpan transaksi & Melatih ulang AI..."):
                    process_transaction_and_retrain(st.session_state.cart)
                    
                st.balloons()
                st.success("🎉 Transaksi Berhasil! AI telah mempelajari kombinasi menu ini.")
                clear_cart()
                st.rerun()
                
            if st.button("🗑️ Kosongkan Keranjang", use_container_width=True):
                clear_cart()
                st.rerun()

# ==============================================================================
# HALAMAN 2: VISUALISASI DATA
# ==============================================================================
elif menu_selection == "Visualisasi Data":
    st.title("📈 Dashboard Analitik & Performa AI")
    st.markdown("<p class='text-muted'>Analisis pola belanja pelanggan Anda secara visual.</p>", unsafe_allow_html=True)
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Menu", len(all_products))
    with m2:
        st.metric("Pola Asosiasi Kuat", len(df_rules))
    with m3:
        st.metric("Akurasi Tertinggi", f"{(df_rules['confidence'].max() * 100):.1f}%")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container():
        st.subheader("🌐 Jaringan Koneksi Menu (Network Graph)")
        st.markdown("<p class='text-muted'>Peta ini menunjukkan bagaimana menu-menu di restoran Anda saling terhubung berdasarkan data penjualan. Semakin tebal garisnya, semakin kuat rekomendasinya.</p>", unsafe_allow_html=True)
        if len(df_rules) > 0:
            # Gunakan rules yang kuat (Top 30) agar grafiknya tidak terlalu ruwet
            top_rules = df_rules.sort_values(by='lift', ascending=False).head(30)
            
            # Menggunakan DiGraph (Directed Graph) seperti di Colab
            G = nx.DiGraph()
            for _, row in top_rules.iterrows():
                # Menggunakan lift sebagai bobot
                G.add_edge(row['antecedents'], row['consequents'], weight=row['lift'])
                
            # Menggunakan tata letak melingkar (circular_layout) agar rapi seperti Colab
            pos = nx.circular_layout(G)
            
            # Normalisasi ketebalan garis (Edge)
            edge_x = []
            edge_y = []
            min_lift = min([G[u][v]['weight'] for u, v in G.edges()]) if len(G.edges()) > 0 else 1
            
            # Di Plotly, kita tidak bisa dengan mudah memberi ketebalan berbeda per garis dalam 1 trace Scatter.
            # Namun kita bisa membuat multiple traces jika ingin ketebalan berbeda, 
            # atau cukup gunakan warna/ketebalan standar yang menyesuaikan tema.
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='rgba(150, 150, 150, 0.5)'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

            node_adjacencies = []
            node_text = []
            for node, adjacencies in enumerate(G.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))
                node_text.append(f"<b>{adjacencies[0]}</b><br>Terhubung dengan {len(adjacencies[1])} menu lain")

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=[adj[0] for adj in G.adjacency()],
                textposition="bottom center",
                hoverinfo='text',
                hovertext=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    reversescale=True,
                    color=node_adjacencies,
                    size=35,
                    colorbar=dict(
                        thickness=15,
                        title='Kekuatan Hubungan'
                    ),
                    line=dict(color='white', width=2)))

            fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# HALAMAN 3: DATABASE ATURAN APRIORI
# ==============================================================================
elif menu_selection == "Database Aturan":
    st.title("🗄️ Database Algoritma (Raw Data)")
    st.markdown("<p class='text-muted'>Tabel di bawah ini menampilkan hasil komputasi <i>Machine Learning</i> Apriori dari dataset historis Anda.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    formatted_df = df_rules.copy()
    formatted_df['support'] = formatted_df['support'].apply(lambda x: f"{x:.5f}")
    formatted_df['confidence'] = formatted_df['confidence'].apply(lambda x: f"{x:.4f}")
    formatted_df['lift'] = formatted_df['lift'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(
        formatted_df, 
        use_container_width=True,
        height=600,
        column_config={
            "antecedents": st.column_config.TextColumn("Jika Pelanggan Beli", width="large"),
            "consequents": st.column_config.TextColumn("Maka Tawarkan", width="large"),
            "support": "Support Score",
            "confidence": "Confidence Score",
            "lift": "Lift Ratio"
        }
    )
