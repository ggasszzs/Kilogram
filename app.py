import streamlit as st
import pandas as pd
import plotly.express as px
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

# --- CSS PREMIUM & ELEGAN ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 16px !important;
    }
    
    /* Background Elegan */
    .stApp {
        background: #f8fafc;
        background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    /* Teks dan Heading Global untuk mencegah crash dengan Dark Mode */
    h1, h2, h3, p, span, div {
        color: #1e293b;
    }
    
    /* Menyembunyikan header bawaan Streamlit agar lebih bersih */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Styling Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        box-shadow: 4px 0 15px rgba(0,0,0,0.03);
        border-right: 1px solid #f1f5f9;
    }
    
    /* Header/Hero Section Premium */
    .hero-banner {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 40px;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(59, 130, 246, 0.2);
        margin-bottom: 40px;
        color: white !important;
    }
    .hero-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 10px;
        color: white !important;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 18px;
        font-weight: 400;
        color: #e0e7ff !important;
    }
    
    /* Card Styles */
    .glass-card {
        background: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        margin-bottom: 25px;
    }
    .glass-card h2, .glass-card h3 {
        color: #0f172a !important;
    }
    .glass-card p {
        color: #475569 !important;
    }
    
    /* Recommendation Item Card */
    .rec-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 6px solid #10b981;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: transform 0.2s;
    }
    .rec-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.1);
    }
    .rec-name {
        font-size: 18px;
        font-weight: 800;
        color: #0f172a !important;
    }
    .rec-metrics {
        font-size: 14px;
        color: #64748b !important;
        margin-top: 5px;
    }
    
    /* Cart Styling */
    .cart-item {
        font-size: 16px;
        padding: 15px 0;
        border-bottom: 1px solid #f1f5f9;
        color: #1e293b !important;
        font-weight: 600;
    }
    
    /* Kustomisasi Selectbox */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border-radius: 12px !important;
        padding: 5px !important;
        border: 2px solid #e2e8f0 !important;
    }
    
    /* Membesarkan Teks Sidebar */
    .stRadio p {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    [data-testid="stSidebar"] h2 {
        color: #1e40af !important;
    }
    .stAlert p {
        color: #1e293b !important;
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

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("<br>", unsafe_allow_html=True)
col_img1, col_img2, col_img3 = st.sidebar.columns([1,2,1])
with col_img2:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", use_container_width=True)
st.sidebar.markdown("<h2 style='text-align:center; color:#1e3a8a;'>Kilo POS</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center; color:#64748b; margin-top:-15px; margin-bottom:30px;'>Sistem Kasir Pintar AI</p>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    ["🛍️ Kasir & Rekomendasi", "📈 Visualisasi Data", "🗄️ Database Aturan"]
)
st.sidebar.markdown("---")

# Mengisi ruang kosong di Sidebar
st.sidebar.markdown("### 👤 Info Profil")
st.sidebar.markdown("**Role:** Administrator")
st.sidebar.markdown("**Cabang:** Pusat")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Panduan:** Arahkan kursor ke pojok kiri atas layar untuk menyembunyikan atau memunculkan sidebar ini agar area kerja lebih luas.")

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center; font-size:12px; color:#94a3b8;'>© 2026 Kilo POS System<br>Powered by Apriori Machine Learning</p>", unsafe_allow_html=True)

# --- HERO BANNER ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Kilo Point of Sale</div>
    <div class="hero-subtitle">Sistem Kasir Pintar Berbasis Algoritma Apriori AI</div>
</div>
""", unsafe_allow_html=True)

# --- PAGE 1: KASIR & REKOMENDASI INTERAKTIF ---
if menu == "🛍️ Kasir & Rekomendasi":
    
    # Layout Utama
    col_main, col_spacer, col_sidebar = st.columns([2.5, 0.2, 1.5])
    
    with col_main:
        st.markdown("""
        <div class="glass-card">
            <h2 style='margin-top:0;'>🏷️ Cari & Tambah Produk</h2>
            <p style='color:#64748b; font-size:16px;'>Ketik nama produk untuk mencari lebih cepat.</p>
        """, unsafe_allow_html=True)
        
        selected_product = st.selectbox("", ["-- Pilih Menu Untuk Ditambahkan --"] + all_products, label_visibility="collapsed")
        
        if selected_product != "-- Pilih Menu Untuk Ditambahkan --":
            if st.button(f"➕ Tambahkan '{selected_product}' ke Keranjang", type="primary", use_container_width=True):
                add_to_cart(selected_product)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
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
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding: 40px;">
                <h3 style="color:#64748b !important; margin-bottom: 10px;">Keranjang Masih Kosong</h3>
                <p style="color:#94a3b8 !important; margin-bottom: 30px;">Rekomendasi otomatis (Cross-Selling) akan muncul di sini setelah Anda menambahkan menu ke keranjang.</p>
                <div style="background: #f1f5f9; height: 1px; width: 100%; margin-bottom: 25px;"></div>
                <p style="color:#334155 !important; font-weight:700; margin-bottom: 15px;">🌟 Menu Terpopuler Hari Ini:</p>
            """, unsafe_allow_html=True)
            
            top_trending = df_rules.sort_values(by='support', ascending=False)['antecedents'].unique()[:3]
            cols = st.columns(3)
            for i, trending in enumerate(top_trending):
                with cols[i]:
                    st.markdown(f"<div style='background:white; padding:15px; border-radius:12px; font-weight:600; color:#2563eb; border: 1px solid #bfdbfe; box-shadow: 0 4px 6px rgba(37, 99, 235, 0.05);'>🏆 {trending}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_sidebar:
        # Panel Keranjang Belanja
        st.markdown("""
        <div class="glass-card" style="position: sticky; top: 20px;">
            <h2 style='margin-top:0; color:#1e293b; border-bottom: 3px solid #3b82f6; padding-bottom: 15px;'>🛒 Keranjang</h2>
        """, unsafe_allow_html=True)
        
        if len(st.session_state.cart) == 0:
            st.markdown("""
            <div style="text-align:center; padding: 40px 0;">
                <div style="font-size: 50px; opacity: 0.2; margin-bottom: 10px;">🛒</div>
                <p style="color:#94a3b8 !important; font-weight: 500;">Belum ada pesanan</p>
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
                
        st.markdown("</div>", unsafe_allow_html=True)

# --- PAGE 2: VISUALISASI DATA ---
elif menu == "📈 Visualisasi Data":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.title("📈 Dashboard Analitik & Performa AI")
    st.markdown("Analisis pola belanja pelanggan Anda secara visual.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="glass-card" style="border-top: 5px solid #3b82f6; text-align:center;">
            <h3 style='color:#64748b; margin:0;'>Total Menu</h3>
            <h1 style='color:#1e293b; font-size:48px; margin:10px 0;'>{len(all_products)}</h1>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="glass-card" style="border-top: 5px solid #10b981; text-align:center;">
            <h3 style='color:#64748b; margin:0;'>Pola Asosiasi Kuat</h3>
            <h1 style='color:#1e293b; font-size:48px; margin:10px 0;'>{len(df_rules)}</h1>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="glass-card" style="border-top: 5px solid #f59e0b; text-align:center;">
            <h3 style='color:#64748b; margin:0;'>Akurasi Tertinggi</h3>
            <h1 style='color:#1e293b; font-size:48px; margin:10px 0;'>{(df_rules['confidence'].max() * 100):.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📍 Peta Kekuatan Rekomendasi")
        if len(df_rules) > 0:
            fig = px.scatter(
                df_rules, 
                x='support', 
                y='confidence', 
                size='lift',
                color='lift',
                hover_data=['antecedents', 'consequents'],
                labels={'support': 'Frekuensi (Support)', 'confidence': 'Akurasi (Confidence)', 'lift': 'Lift Ratio'},
                color_continuous_scale=px.colors.sequential.Plotly3
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🏆 Top 10 Kombinasi Terbaik")
        if len(df_rules) > 0:
            top_10 = df_rules.head(10).copy()
            top_10['Aturan'] = top_10['antecedents'] + " ➔ " + top_10['consequents']
            
            fig2 = px.bar(
                top_10,
                x='lift',
                y='Aturan',
                orientation='h',
                color='confidence',
                color_continuous_scale=px.colors.sequential.Plotly3,
                labels={'lift': 'Nilai Lift Ratio', 'Aturan': ''}
            )
            fig2.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- PAGE 3: DATABASE ATURAN ---
elif menu == "🗄️ Database Aturan":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.title("🗄️ Database Algoritma (Raw Data)")
    st.markdown("Tabel di bawah ini menampilkan hasil komputasi *Machine Learning* Apriori dari dataset historis Anda.")
    
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
    st.markdown("</div>", unsafe_allow_html=True)
