import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from mlxtend.frequent_patterns import apriori, association_rules
import os
import time

# ==============================================================================
# 1. KONFIGURASI HALAMAN & STATE
# ==============================================================================
st.set_page_config(page_title="Kilo POS System", page_icon="🛒", layout="wide", initial_sidebar_state="collapsed")

if 'cart' not in st.session_state:
    st.session_state.cart = []

if 'sales_history' not in st.session_state:
    # Data awal agar chart tidak kosong, akan bertambah terus saat transaksi
    st.session_state.sales_history = {
        "Java Latte": 142, "Americano Ice Bold": 98, "Butter Croissant": 87, 
        "Spaghetti Aglio e Olio": 76, "Choco Berry": 65, "Lemon Tea Ice": 54,
        "Latte Ice Bold": 43, "Cappucino Hot Light": 32, "Earl Grey": 21, "Churros Beton": 15
    }

def add_to_cart(product):
    if product not in st.session_state.cart:
        st.session_state.cart.append(product)
        st.toast(f"Berhasil menambahkan {product} ke keranjang!", icon="✅")

def clear_cart():
    st.session_state.cart = []

# ==============================================================================
# 2. STYLING CSS
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif !important; }
    
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #8b5cf6 100%);
        padding: 40px 30px; border-radius: 20px; text-align: center;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3); margin-bottom: 30px; color: white !important;
    }
    .hero-title { font-size: 48px; font-weight: 800; margin-bottom: 5px; color: #ffffff !important; }
    .hero-subtitle { font-size: 18px; font-weight: 300; color: rgba(255,255,255,0.9) !important; }
    
    .glass-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 16px; padding: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.05); margin-bottom: 20px;
    }
    
    .rec-item {
        background: linear-gradient(145deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
        border: 1px solid rgba(128,128,128,0.15); border-radius: 12px; padding: 15px;
        margin-bottom: 12px; border-left: 5px solid #10b981;
        transition: all 0.3s ease;
    }
    .rec-item.manual { border-left: 5px solid #f59e0b; }
    .rec-item:hover { transform: translateX(5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
    
    .rec-name { font-size: 18px; font-weight: 700; color: var(--text-color); }
    .rec-metrics { font-size: 13px; color: #888; margin-top: 4px; font-weight: 500; }
    
    .promo-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e; padding: 15px; border-radius: 12px; margin-bottom: 15px;
        border: 1px dashed #d97706; font-weight: 600;
    }
    
    .cart-item { font-size: 16px; padding: 12px 0; border-bottom: 1px dashed rgba(128,128,128,0.3); font-weight: 600; }
    
    /* SIDEBAR ESTETIK */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(128,128,128,0.05);
        padding: 12px 15px;
        border-radius: 10px;
        margin-bottom: 8px;
        transition: all 0.2s ease-in-out;
        border: 1px solid transparent;
        cursor: pointer;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(59, 130, 246, 0.1);
        transform: scale(1.02);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%);
        border-left: 4px solid #3b82f6;
    }
    /* Sembunyikan bulatan radio bawaan */
    [data-testid="stSidebar"] span[data-baseweb="radio"] {
        display: none !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
        font-weight: 600;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. KATEGORI & REKOMENDASI MANUAL
# ==============================================================================
def get_category(product_name):
    name_lower = product_name.lower()
    makanan_keywords = ['nasi', 'sate', 'ayam', 'spaghetti', 'burger', 'sandwich', 'fries', 'cireng', 'bala', 'omelete', 'tahu', 'telur', 'wings', 'tempe', 'chicken']
    pastry_keywords = ['croissant', 'kouign', 'pain', 'berliner', 'tartlet', 'brownies', 'cookies', 'churros', 'pancake', 'monkey', 'brule', 'nut bar']
    
    if any(k in name_lower for k in makanan_keywords): return "Makanan"
    if any(k in name_lower for k in pastry_keywords): return "Pastry"
    return "Minuman" # Default fallback (most of them are drinks)

# Rekomendasi khusus dari owner (Manual Rule)
MANUAL_RECOMMENDATIONS = {
    "Java Latte": ["Cookies Choco", "Brownies with Ice Cream"],
    "Nasi Goreng Sambal Ijo": ["Ice Tea", "Tempe Mendoan"],
    "Americano Ice Bold": ["Butter Croissant", "Churros Plain"],
    "Spaghetti Aglio e Olio": ["Lemon Tea Ice"],
    "Kilo Cold White": ["Pain Au Chocolat"]
}

# ==============================================================================
# 4. LOAD DATA (CACHED)
# ==============================================================================
@st.cache_data
def load_data():
    try:
        if os.path.exists("data/Rekomendasi_CrossSell.xlsx"):
            df = pd.read_excel("data/Rekomendasi_CrossSell.xlsx")
            # Parse the string "frozenset({'Item'})" back to an actual frozenset
            def parse_frozenset(val):
                if isinstance(val, str) and val.startswith("frozenset"):
                    return frozenset(eval(val.replace("frozenset(", "").replace(")", "")))
                elif isinstance(val, str):
                    return frozenset([i.strip() for i in val.split(',')])
                return frozenset()
                
            df['antecedents'] = df['antecedents'].apply(parse_frozenset)
            df['consequents'] = df['consequents'].apply(parse_frozenset)
            return df
            
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
    except Exception as e:
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

def process_transaction_and_retrain(cart_items):
    # Dummy processing func to simulate training
    time.sleep(2)
    return True

df_rules = load_data()

# Semua menu restoran di-hardcode agar tidak hilang meski belum ada di data rules
all_products = sorted([
    'Cookies Choco', 'Java Latte', 'Magic Ice Bold', 'Matcha Ice', 'Nasi Ayam Goreng', 'Flat White Bold', 'Kilo Cold White', 'Pain Au Chocolat', 'Kouign Amann', 'Churros Plain', 'Iced Lychee Tea', 'Nut Bar', 'Hojicha Ice', 'Nasi Ayam Bakar', 'Americano Hot Bold', 'Chicken Wings', 'Hojicha Hot', 'Sate Ayam', 'Croissant Almond', 'Affogato', 'Latte Hot Bold', 'Long Black Ice Light', 'Aloe Fresh', 'Green tea', 'Americano Ice Bold', 'Magic Hot Bold', 'Monkey Bread', 'Sandwich Chicken Spicy Mango', 'Tempe Mendoan', 'Latte Hot Light', 'Yellow Breeze', 'Butter Croissant', 'Extra Shot Espresso', 'Creme Brule', 'Romansky', 'Churros Beton', 'Berrymore', 'Tartlet Chocolate', 'Pancake with Ice Cream', 'Matcha Berry', 'French Fries', 'Burger Classic Beef', 'Tahu Goreng Lada Garam', 'Honey Lemon', 'Ice Cream', 'Cold Pressed Orange Juice', 'Matcha Hot', 'Ice Tea', 'Iced Kilo', 'Jahe Lemon', 'Berliner Vanila', 'Americano Ice Light', 'Nasi Goreng Teri Honje', 'Cappucino Ice Bold', 'Chillie Fries', 'Sandwich Chicken Lemon Mayo', 'Chocolate Ice', 'Kunyit Asem', 'Chocolate Hot', 'Latte Ice Light', 'Spaghetti Carbonara', 'Croissant Bacon & Cheese Sandwich', 'Espresso Bold', 'Latte Ice Bold', 'Spaghetti Aglio e Olio', 'Omelete', 'Lemon Tea Ice', 'Ocean Eyes', 'Blue Lagoon', 'Mineral Water', 'Long Black Hot Light', 'Nitri Coffee Soda', 'Long Black Ice Bold', 'Earl Grey', 'Burger Classic Chicken', 'Cireng', 'Americano Hot Light', 'Cappuchino Ice Light', 'Sate Sapi', 'Lemon Tea Hot', 'Long Black Hot Bold', 'Cappucino Hot Light', 'Choco Berry', 'Bala - Bala', 'Chicken Popcorn', 'Spaghetti Cheese Bolognese', 'Morrocant mint', 'Berliner Chocolate', 'Nasi Goreng Sambal Ijo', 'Brownies with Ice Cream', 'Cookies Oatmeal', 'Berliner Coffee Cream', 'Magic Ice Light', 'Strawberry Juice', 'Cappuccino Hot Bold', 'Extra Telur', 'Caramel Machiato', 'Croissant Bacon & Egg', 'Tartlet Strawberry Cheese', 'Nasi Goreng Sambal Cikur', 'Vanilla Sweet Tea ice', 'Piccolo Bold'
])

# Create Categorized dict
product_categories = {"Minuman": [], "Makanan": [], "Pastry": []}
for p in all_products:
    product_categories[get_category(p)].append(p)

# ==============================================================================
# 5. SIDEBAR & NAVIGASI
# ==============================================================================
st.sidebar.markdown("""
<div style='text-align:center; padding-bottom: 10px;'>
    <div style='font-size: 50px;'>🛒</div>
    <h2 style='margin:0; font-weight:800; color:#3b82f6;'>Kilo POS</h2>
    <p style='color:gray; font-size:12px;'>Sistem Kasir Pintar AI</p>
</div>
<hr style='border-color: rgba(128,128,128,0.2);'>
""", unsafe_allow_html=True)

menu_selection = st.sidebar.radio(
    "",
    ["Kasir & Rekomendasi", "Ide Promo Bundling", "Visualisasi Data", "Database Aturan"],
    format_func=lambda x: "💼 " + x if "Kasir" in x else ("📊 " + x if "Visualisasi" in x else ("⚙️ " + x if "Database" in x else "🎁 " + x))
)

st.sidebar.markdown("<hr style='border-color: rgba(128,128,128,0.2);'>", unsafe_allow_html=True)
st.sidebar.markdown("👤 **Info Profil**<br><span style='font-size:14px; color:gray;'>Role: Administrator<br>Cabang: Pusat</span>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-color: rgba(128,128,128,0.2);'>", unsafe_allow_html=True)
st.sidebar.info("💡 **Panduan:** Klik ikon panah di pojok kiri atas layar untuk melipat (menyembunyikan) atau memunculkan menu ini.")

# ==============================================================================
# 6. HALAMAN UTAMA (KASIR & REKOMENDASI)
# ==============================================================================
if menu_selection == "Kasir & Rekomendasi":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Kilo Point of Sale</div>
        <div class="hero-subtitle">Sistem Kasir Pintar Berbasis Algoritma Apriori AI</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_main, col_spacing, col_sidebar = st.columns([6, 0.5, 3.5])
    
    with col_main:
        # --- CARI & TAMBAH PRODUK ---
        with st.container():
            st.markdown("<h3>🏷️ Cari & Tambah Produk</h3>", unsafe_allow_html=True)
            
            # Kategori Filter
            kategori_terpilih = st.radio("Kategori Menu:", ["Semua", "Minuman", "Makanan", "Pastry"], horizontal=True)
            
            if kategori_terpilih == "Semua":
                list_pilihan = all_products
            else:
                list_pilihan = product_categories[kategori_terpilih]
                
            selected_product = st.selectbox("Pilih Menu:", ["-- Pilih Menu --"] + list_pilihan, label_visibility="collapsed")
            
            if selected_product != "-- Pilih Menu --":
                if st.button(f"➕ Tambahkan '{selected_product}' ke Keranjang", type="primary", use_container_width=True):
                    add_to_cart(selected_product)
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- REKOMENDASI PINTAR ---
        with st.container():
            st.markdown("<h3>🔥 Rekomendasi Pintar</h3>", unsafe_allow_html=True)
            
            if len(st.session_state.cart) == 0:
                st.markdown("<p style='text-align:center; padding:30px; color:gray;'>Pilih produk terlebih dahulu untuk melihat rekomendasi.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='text-muted' style='font-size:14px;'>Rekomendasi saling bersambung berdasarkan menu yang ada di keranjang Anda.</p>", unsafe_allow_html=True)
                
                # 1. Kumpulkan Rekomendasi AI (Data Driven)
                ai_recs = []
                if len(df_rules) > 0:
                    current_items = frozenset(st.session_state.cart)
                    for _, row in df_rules.iterrows():
                        if row['antecedents'].issubset(current_items):
                            for rec_item in row['consequents']:
                                if rec_item not in st.session_state.cart:
                                    ai_recs.append({
                                        'item': rec_item,
                                        'confidence': row['confidence'],
                                        'lift': row['lift']
                                    })
                
                # Sort and remove duplicates for AI
                ai_recs = sorted(ai_recs, key=lambda x: x['lift'], reverse=True)
                unique_ai_recs = []
                seen = set()
                for r in ai_recs:
                    if r['item'] not in seen:
                        seen.add(r['item'])
                        unique_ai_recs.append(r)
                
                # 2. Kumpulkan Rekomendasi Manual Owner
                manual_recs = []
                for item in st.session_state.cart:
                    if item in MANUAL_RECOMMENDATIONS:
                        for rec_item in MANUAL_RECOMMENDATIONS[item]:
                            if rec_item not in st.session_state.cart and rec_item not in seen:
                                manual_recs.append(rec_item)
                                seen.add(rec_item)

                # 3. Tampilkan Rekomendasi
                if len(unique_ai_recs) == 0 and len(manual_recs) == 0:
                    st.info("Belum ada rekomendasi lanjutan untuk kombinasi ini.")
                else:
                    rec_col1, rec_col2 = st.columns(2)
                    
                    # Kolom Kiri: AI
                    with rec_col1:
                        st.markdown("<h5 style='color:#10b981;'>🤖 Berdasarkan Data</h5>", unsafe_allow_html=True)
                        for r in unique_ai_recs[:3]:
                            st.markdown(f"""
                            <div class="rec-item">
                                <div>
                                    <div class="rec-name">{r['item']}</div>
                                    <div class="rec-metrics">Tingkat Kecocokan: <b>{r['confidence']*100:.0f}%</b></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    # Kolom Kanan: Manual Owner
                    with rec_col2:
                        st.markdown("<h5 style='color:#f59e0b;'>👨‍🍳 Berdasarkan Barista</h5>", unsafe_allow_html=True)
                        if len(manual_recs) == 0:
                            st.markdown("<i style='color:gray; font-size:13px;'>Tidak ada saran spesial.</i>", unsafe_allow_html=True)
                        for m_item in manual_recs[:3]:
                            st.markdown(f"""
                            <div class="rec-item manual">
                                <div>
                                    <div class="rec-name">{m_item}</div>
                                    <div class="rec-metrics">🌟 Sangat Disarankan (Manual)</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

    # --- KERANJANG BELANJA ---
    with col_sidebar:
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
            
            if st.button("💳 Proses Pembayaran", type="primary", use_container_width=True):
                with st.spinner("⏳ Menyimpan transaksi & Melatih ulang AI..."):
                    process_transaction_and_retrain(st.session_state.cart)
                    # Catat history penjualan untuk chart
                    for item in st.session_state.cart:
                        if item in st.session_state.sales_history:
                            st.session_state.sales_history[item] += 1
                        else:
                            st.session_state.sales_history[item] = 1
                st.balloons()
                st.success("🎉 Transaksi Berhasil! AI telah mempelajari kombinasi menu ini.")
                clear_cart()
                st.rerun()
                
            if st.button("🗑️ Kosongkan Keranjang", use_container_width=True):
                clear_cart()
                st.rerun()

# ==============================================================================
# 7. IDE PROMO BUNDLING (NEW)
# ==============================================================================
elif menu_selection == "Ide Promo Bundling":
    st.title("🎁 Ide Promo Bundling")
    st.markdown("<p class='text-muted'>Sistem AI merangkum pasangan menu terkuat yang cocok dijadikan Paket Bundling Promosi.</p>", unsafe_allow_html=True)
    
    if len(df_rules) > 0:
        top_bundling = df_rules.sort_values(by='lift', ascending=False).head(5)
        
        for i, row in top_bundling.iterrows():
            menu_A = ", ".join(list(row['antecedents']))
            menu_B = ", ".join(list(row['consequents']))
            
            st.markdown(f"""
            <div style="padding: 15px 20px; border-bottom: 1px solid rgba(128,128,128,0.2); margin-bottom: 10px;">
                <div style="display:flex; align-items:center; margin-bottom: 8px;">
                    <div style="font-size: 24px; margin-right: 15px;">🎁</div>
                    <div>
                        <h4 style="margin:0; font-weight:700; color:var(--text-color);">Paket: {menu_A} + {menu_B}</h4>
                        <p style="margin:0; font-size:14px; color:gray;">Kekuatan Asosiasi: {row['lift']:.1f}x lebih tinggi</p>
                    </div>
                </div>
                <div style="padding: 10px 15px; background: rgba(59, 130, 246, 0.05); border-radius: 8px; border-left: 4px solid #3b82f6; font-size: 14px; margin-left: 40px;">
                    <b>💡 Ide Promosi:</b> Beli {menu_A}, dapatkan diskon 10% untuk {menu_B}!
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Belum ada data asosiasi yang cukup untuk membuat ide bundling.")

# ==============================================================================
# 8. VISUALISASI DATA
# ==============================================================================
elif menu_selection == "Visualisasi Data":
    st.title("📊 Dashboard Analitik & Performa AI")
    st.markdown("<p class='text-muted'>Analisis pola belanja pelanggan Anda secara visual.</p>", unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Total Menu", len(all_products))
    with m2: st.metric("Pola Asosiasi Kuat", len(df_rules))
    with m3: 
        max_conf = (df_rules['confidence'].max() * 100) if len(df_rules)>0 else 0
        st.metric("Akurasi Tertinggi", f"{max_conf:.1f}%")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ---------------- NETWORK GRAPH ----------------
    with st.container():
        st.subheader("🌐 Jaringan Koneksi Menu (Network Graph)")
        st.markdown("<p class='text-muted'>Peta asosiasi produk. Semakin tebal panah, semakin kuat pelanggan membeli dua produk ini secara bersamaan.</p>", unsafe_allow_html=True)
        if len(df_rules) > 0:
            top_rules = df_rules.sort_values(by='lift', ascending=False).head(30)
            G = nx.DiGraph()
            for _, row in top_rules.iterrows():
                for a in row['antecedents']:
                    for c in row['consequents']:
                        G.add_edge(a, c, lift=row['lift'])

            pos = nx.circular_layout(G)
            node_sizes = [1500 + G.degree(node) * 300 for node in G.nodes()]
            
            # Agar label tidak tumpang tindih dengan bulatannya, dorong posisinya sedikit ke luar (scale > 1)
            pos_labels = {}
            for node, coords in pos.items():
                pos_labels[node] = (coords[0] * 1.20, coords[1] * 1.20)
            
            edge_widths = []
            if len(G.edges()) > 0:
                min_lift = min([G[u][v]['lift'] for u, v in G.edges()])
                for u, v in G.edges():
                    width = ((G[u][v]['lift'] - min_lift) * 6) + 1.5
                    edge_widths.append(width)

            fig, ax = plt.subplots(figsize=(16, 14)) # Ukuran diperbesar
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#fbcfe8", edgecolors="#be185d", linewidths=2, ax=ax)
            
            # Draw labels OUTSIDE the nodes
            texts = nx.draw_networkx_labels(G, pos_labels, font_size=13, font_weight="bold", font_family="sans-serif", ax=ax)
            for _, t in texts.items():
                t.set_path_effects([PathEffects.withStroke(linewidth=4, foreground="white")])

            if len(G.edges()) > 0:
                # Draw edges
                nx.draw_networkx_edges(
                    G, pos, width=edge_widths, edge_color="#cbd5e1", arrows=True,
                    arrowsize=25, arrowstyle="-|>", connectionstyle="arc3,rad=0.1",
                    node_size=node_sizes, min_source_margin=20, min_target_margin=20, ax=ax
                )
                
                # Draw edge labels
                edge_labels = {(u, v): f"{G[u][v]['lift']:.2f}" for u, v in G.edges()}
                edge_texts = nx.draw_networkx_edge_labels(
                    G, pos, edge_labels=edge_labels, font_size=12, font_color="#dc2626",
                    font_weight="bold", rotate=True, label_pos=0.35, ax=ax
                )
                for _, t in edge_texts.items():
                    t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="white")])

            # Batas area gambar diperbesar agar label luar tidak terpotong
            ax.margins(0.15) 
            plt.axis("off")
            st.pyplot(fig, transparent=True)

# ==============================================================================
# 9. DATABASE ATURAN APRIORI
# ==============================================================================
elif menu_selection == "Database Aturan":
    st.title("⚙️ Database Algoritma (Raw Data)")
    st.markdown("<p class='text-muted'>Tabel di bawah ini menampilkan hasil komputasi <i>Machine Learning</i> Apriori.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if len(df_rules) > 0:
        formatted_df = df_rules.copy()
        formatted_df['antecedents'] = formatted_df['antecedents'].apply(lambda x: ", ".join(list(x)))
        formatted_df['consequents'] = formatted_df['consequents'].apply(lambda x: ", ".join(list(x)))
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
    else:
        st.warning("Data aturan kosong.")

    st.markdown("<hr style='border-color: rgba(128,128,128,0.2); margin: 40px 0;'>", unsafe_allow_html=True)

    # ---------------- BAR CHART TOP 10 ----------------
    with st.container():
        st.subheader("🏆 Top 10 Menu Paling Sering Dibeli")
        st.markdown("<p class='text-muted'>Data penjualan otomatis diperbarui secara <i>real-time</i> saat kasir memproses pesanan.</p>", unsafe_allow_html=True)
        
        # Sort history and take top 10
        sorted_sales = sorted(st.session_state.sales_history.items(), key=lambda x: x[1], reverse=True)[:10]
        df_sales = pd.DataFrame(sorted_sales, columns=["Menu", "Total Terjual"])
        df_sales = df_sales.sort_values(by="Total Terjual", ascending=True) # Sort ascending for Plotly horizontal bar

        fig_bar = px.bar(
            df_sales, x="Total Terjual", y="Menu", orientation='h',
            color="Total Terjual", color_continuous_scale="Blues",
            text="Total Terjual"
        )
        fig_bar.update_layout(
            showlegend=False,
            margin=dict(l=0, r=20, t=20, b=0),
            height=400,
            xaxis_title="",
            yaxis_title=""
        )
        # Bold y-axis labels
        fig_bar.update_yaxes(tickfont=dict(weight='bold', size=13))
        st.plotly_chart(fig_bar, use_container_width=True)

    