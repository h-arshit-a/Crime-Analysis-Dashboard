import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import all your existing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="Crime Analysis Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .section-header {
        color: #1E3A8A;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .data-loaded-badge {
        background-color: #10B981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🚨 Crime Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Analysis Section:",
    ["📊 Dataset Overview", "📈 Crime Trend Prediction", "🌗 Day vs Night Classification", 
     "👤 Victim Gender Prediction", "📍 Crime Hotspots", "📊 PCA Analysis"]
)

# ==============================
# DATA LOADING FUNCTION
# ==============================
@st.cache_data
def load_data():
    """Load the crime_dataset.csv file from local directory"""
    try:
        # Try to load the dataset from current directory
        file_path = 'crime_dataset.csv'
        
        if os.path.exists(file_path):
            st.sidebar.success(f"✅ Found: {file_path}")
            df = pd.read_csv(file_path)
            
            # Process the data (your original processing code)
            with st.spinner('Processing data...'):
                # DATA CLEANING & FEATURE ENGINEERING
                df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
                df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)
                df['Hour'] = df['TIME OCC'].str[:2].astype(int)
                df = df[(df['LAT'] != 0) & (df['LON'] != 0)]
                df = df[df['Vict Sex'].isin(['M', 'F'])]
                df = df[df['Vict Age'] > 0]
                df.dropna(subset=['AREA NAME', 'Crm Cd Desc', 'DATE OCC', 'Vict Descent'], inplace=True)
                
                df['Month'] = df['DATE OCC'].dt.month
                df['Year'] = df['DATE OCC'].dt.year
                df['Day_of_Week'] = df['DATE OCC'].dt.day_name()
                df['Day_Night'] = np.where((df['Hour'] >= 6) & (df['Hour'] < 18), 'Day', 'Night')
                
                crime_year = df.groupby('Year').size().reset_index(name='Crime_Count')
            
            return df, crime_year, True
        else:
            # If file not found, show upload option
            st.sidebar.warning(f"❌ File not found: {file_path}")
            st.sidebar.info("Please make sure 'crime_dataset.csv' is in the same directory as this app.")
            
            # Alternative: Manual upload
            uploaded_file = st.sidebar.file_uploader("Or upload your file:", type=['csv'], key="sidebar_uploader")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                
                # Process the data
                with st.spinner('Processing data...'):
                    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
                    df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)
                    df['Hour'] = df['TIME OCC'].str[:2].astype(int)
                    df = df[(df['LAT'] != 0) & (df['LON'] != 0)]
                    df = df[df['Vict Sex'].isin(['M', 'F'])]
                    df = df[df['Vict Age'] > 0]
                    df.dropna(subset=['AREA NAME', 'Crm Cd Desc', 'DATE OCC', 'Vict Descent'], inplace=True)
                    
                    df['Month'] = df['DATE OCC'].dt.month
                    df['Year'] = df['DATE OCC'].dt.year
                    df['Day_of_Week'] = df['DATE OCC'].dt.day_name()
                    df['Day_Night'] = np.where((df['Hour'] >= 6) & (df['Hour'] < 18), 'Day', 'Night')
                    
                    crime_year = df.groupby('Year').size().reset_index(name='Crime_Count')
                
                st.sidebar.success("✅ File uploaded and processed!")
                return df, crime_year, True
                
            return None, None, False
            
    except Exception as e:
        st.sidebar.error(f"Error loading data: {str(e)}")
        return None, None, False

# Load data at the start
df, crime_year, data_loaded = load_data()

# Show data status in sidebar
st.sidebar.markdown("---")
if data_loaded and df is not None:
    st.sidebar.markdown('<div class="data-loaded-badge">✅ DATA LOADED</div>', unsafe_allow_html=True)
    st.sidebar.metric("Records", f"{len(df):,}")
    st.sidebar.metric("Time Period", f"{df['Year'].min()} - {df['Year'].max()}")
else:
    st.sidebar.markdown('<div style="background-color: #EF4444; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold;">❌ NO DATA</div>', unsafe_allow_html=True)
    st.sidebar.info("Please ensure 'crime_dataset.csv' is in the same folder.")

# ==============================
# MAIN APP LOGIC
# ==============================

if section == "📊 Dataset Overview":
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    if data_loaded and df is not None:
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Crimes", f"{len(df):,}")
        with col2:
            st.metric("Unique Crime Types", df['Crm Cd Desc'].nunique())
        with col3:
            st.metric("Time Period", f"{df['Year'].min()} - {df['Year'].max()}")
        with col4:
            day_night_ratio = (df['Day_Night'] == 'Day').sum() / len(df) * 100
            st.metric("Daytime Crimes", f"{day_night_ratio:.1f}%")
        
        # Dataset preview
        with st.expander("📋 Dataset Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**Dataset shape:** {df.shape}")
            st.write(f"**Total rows:** {len(df):,}")
            st.write(f"**Total columns:** {len(df.columns)}")
            st.write(f"**Memory usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Basic visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Crime Distribution by Day/Night")
            fig, ax = plt.subplots(figsize=(8, 6))
            day_night_counts = df['Day_Night'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4']
            ax.pie(day_night_counts.values, labels=day_night_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Day vs Night Crime Distribution')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Top 10 Crime Types")
            top_crimes = df['Crm Cd Desc'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 6))
            top_crimes.plot(kind='barh', ax=ax, color='#3B82F6')
            ax.set_xlabel('Number of Incidents')
            ax.set_title('Top 10 Most Common Crimes')
            st.pyplot(fig)
        
        # Data summary
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Numeric Columns Summary:**")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.write("**Categorical Columns Summary:**")
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols[:5]:  # Show first 5 categorical columns
                st.write(f"**{col}**: {df[col].nunique()} unique values")
                top_values = df[col].value_counts().head(3)
                for val, count in top_values.items():
                    st.write(f"  - {val}: {count:,}")
        
        # Show columns info
        with st.expander("📄 Column Information"):
            for col in df.columns:
                st.write(f"**{col}**")
                st.write(f"  - Type: {df[col].dtype}")
                st.write(f"  - Non-null: {df[col].count():,}")
                st.write(f"  - Null: {df[col].isnull().sum():,}")
                if df[col].dtype == 'object':
                    st.write(f"  - Unique values: {df[col].nunique()}")
                    if df[col].nunique() < 10:
                        st.write(f"  - Values: {', '.join(map(str, df[col].unique()))}")
                st.write("---")
    
    else:
        st.error("⚠️ Dataset not loaded!")
        st.info("""
        **To use this dashboard:**
        1. Make sure `crime_dataset.csv` is in the same folder as this app
        2. Or upload it using the sidebar uploader
        3. Restart the app if you just added the file
        """)

# ==============================
# CRIME TREND PREDICTION
# ==============================
elif section == "📈 Crime Trend Prediction":
    st.markdown('<h2 class="section-header">Crime Trend Prediction</h2>', unsafe_allow_html=True)
    
    if not data_loaded or df is None:
        st.error("⚠️ Please load the dataset first! Go to 'Dataset Overview' section.")
        st.stop()
    
    # 4. OBJECTIVE 1: CRIME TREND PREDICTION
    X = crime_year[['Year']]
    y = crime_year['Crime_Count']
    
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    y_pred = lr_model.predict(X)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2_score(y, y_pred):.4f}")
    with col2:
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y, y_pred)):,.0f}")
    with col3:
        trend = "Increasing" if lr_model.coef_[0] > 0 else "Decreasing"
        st.metric("Yearly Trend", trend, f"{lr_model.coef_[0]:.0f} crimes/year")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, label='Actual Crime Count', color='#3B82F6', alpha=0.7, s=100)
    ax.plot(X, y_pred, color='red', linewidth=3, label='Regression Trend')
    ax.set_title("Yearly Crime Trend with Linear Regression", fontweight='bold', fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Crime Count", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_facecolor('#f8f9fa')
    
    st.pyplot(fig)
    
    # Additional insights
    st.subheader("📈 Trend Analysis")
    st.write(f"**Regression Equation:** Crime Count = {lr_model.intercept_:.0f} + {lr_model.coef_[0]:.2f} × Year")
    
    # Predict next year
    next_year = int(X['Year'].max()) + 1
    next_year_pred = lr_model.predict([[next_year]])[0]
    st.write(f"**Predicted crime count for {next_year}:** {next_year_pred:,.0f}")
    
    # Display yearly data
    with st.expander("📊 View Yearly Crime Data"):
        st.dataframe(crime_year, use_container_width=True)

# ==============================
# DAY vs NIGHT CLASSIFICATION
# ==============================
elif section == "🌗 Day vs Night Classification":
    st.markdown('<h2 class="section-header">Day vs Night Crime Classification</h2>', unsafe_allow_html=True)
    
    if not data_loaded or df is None:
        st.error("⚠️ Please load the dataset first! Go to 'Dataset Overview' section.")
        st.stop()
    
    # 5. OBJECTIVE 2: DAY vs NIGHT CLASSIFICATION
    df['Day_Night_Label'] = df['Day_Night'].map({'Day': 1, 'Night': 0})
    
    X_dn = df[['Hour', 'Vict Age']]
    y_dn = df['Day_Night_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_dn, y_dn, test_size=0.3, random_state=42, stratify=y_dn
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    pred = log_model.predict(X_test_scaled)
    
    # Display metrics
    accuracy = accuracy_score(y_test, pred)
    conf_matrix = confusion_matrix(y_test, pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Day Samples", f"{sum(y_test == 1):,}")
    with col3:
        st.metric("Night Samples", f"{sum(y_test == 0):,}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Night', 'Day'], yticklabels=['Night', 'Day'],
                    ax=ax)
        ax.set_title("Day vs Night Classification", fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': ['Hour', 'Victim Age'],
            'Coefficient': np.abs(log_model.coef_[0])
        })
        fig, ax = plt.subplots(figsize=(6, 5))
        feature_importance.sort_values('Coefficient').plot(
            x='Feature', y='Coefficient', kind='barh', ax=ax, color='#3B82F6'
        )
        ax.set_title("Feature Impact on Classification")
        st.pyplot(fig)
    
    # Classification report
    st.subheader("Detailed Performance Metrics")
    st.text(classification_report(y_test, pred, target_names=['Night', 'Day']))

# ==============================
# VICTIM GENDER PREDICTION
# ==============================
elif section == "👤 Victim Gender Prediction":
    st.markdown('<h2 class="section-header">Victim Gender Prediction</h2>', unsafe_allow_html=True)
    
    if not data_loaded or df is None:
        st.error("⚠️ Please load the dataset first! Go to 'Dataset Overview' section.")
        st.stop()
    
    # 6. OBJECTIVE 3: VICTIM GENDER PREDICTION
    le = LabelEncoder()
    df['Vict Sex'] = le.fit_transform(df['Vict Sex'])
    
    X_gender = df[['Vict Age', 'Hour']]
    y_gender = df['Vict Sex']
    
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        X_gender, y_gender, test_size=0.3, random_state=42, stratify=y_gender
    )
    
    scaler = StandardScaler()
    Xg_train_scaled = scaler.fit_transform(Xg_train)
    Xg_test_scaled = scaler.transform(Xg_test)
    
    # Train models
    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(Xg_train_scaled, yg_train)
        pred = model.predict(Xg_test_scaled)
        results[name] = accuracy_score(yg_test, pred)
    
    # Display results
    st.subheader("Model Performance Comparison")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart of model accuracies
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(results.keys(), results.values(), color=colors)
        ax.set_ylabel('Accuracy')
        ax.set_title('Gender Prediction Model Accuracies', fontweight='bold')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        st.write("**Accuracy Scores:**")
        for model, acc in results.items():
            st.metric(model, f"{acc:.2%}")
    
    # Dataset distribution
    st.subheader("Gender Distribution in Dataset")
    gender_counts = df['Vict Sex'].value_counts()
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Pie chart
    labels = ['Male', 'Female'] if len(gender_counts) == 2 else gender_counts.index
    ax[0].pie(gender_counts.values, labels=labels, 
              autopct='%1.1f%%', colors=['#3B82F6', '#FF6B6B'])
    ax[0].set_title('Gender Distribution')
    
    # Box plot by gender
    box_data = [df[df['Vict Sex'] == gender]['Vict Age'] for gender in gender_counts.index]
    ax[1].boxplot(box_data, labels=labels)
    ax[1].set_title('Age Distribution by Gender')
    ax[1].set_ylabel('Age')
    
    st.pyplot(fig)
    
    # Best model details
    best_model_name = max(results, key=results.get)
    st.info(f"🏆 **Best Performing Model:** {best_model_name} ({results[best_model_name]:.2%} accuracy)")

# ==============================
# CRIME HOTSPOTS
# ==============================
elif section == "📍 Crime Hotspots":
    st.markdown('<h2 class="section-header">Crime Hotspot Analysis</h2>', unsafe_allow_html=True)
    
    if not data_loaded or df is None:
        st.error("⚠️ Please load the dataset first! Go to 'Dataset Overview' section.")
        st.stop()
    
    # 7. OBJECTIVE 4: CRIME HOTSPOTS (K-MEANS)
    st.write("Using K-Means clustering to identify crime hotspots")
    
    # Let user choose sample size
    sample_size = st.slider("Select sample size for clustering:", 
                           min_value=1000, 
                           max_value=min(100000, len(df)), 
                           value=50000, 
                           step=1000)
    
    coords = df[['LAT', 'LON']].sample(sample_size, random_state=42)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(coords)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords['LON'], coords['LAT'],
                         c=clusters, cmap='tab10', s=10, alpha=0.6)
    
    # Plot cluster centers
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 1], centers[:, 0], 
               c='red', s=300, alpha=0.8, marker='X', label='Cluster Centers')
    
    ax.set_title(f"Crime Hotspots (K-Means Clustering) - {sample_size:,} samples", 
                 fontweight='bold', fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster ID', fontsize=12)
    
    st.pyplot(fig)
    
    # Cluster statistics
    st.subheader("Cluster Analysis")
    cluster_df = pd.DataFrame({
        'Cluster': clusters,
        'Latitude': coords['LAT'].values,
        'Longitude': coords['LON'].values
    })
    
    cluster_stats = cluster_df.groupby('Cluster').agg({
        'Latitude': ['count', 'mean'],
        'Longitude': 'mean'
    }).round(4)
    
    cluster_stats.columns = ['Number_of_Crimes', 'Avg_Latitude', 'Avg_Longitude']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Cluster Statistics:**")
        st.dataframe(cluster_stats, use_container_width=True)
    
    with col2:
        st.write("**Cluster Interpretation:**")
        st.write("""
        - **Each cluster** represents a geographical hotspot
        - **Red X marks** show cluster centers (centroids)
        - **Larger clusters** indicate higher crime density areas
        - **Use case:** Police resource allocation and patrol planning
        """)

# ==============================
# PCA ANALYSIS
# ==============================
elif section == "📊 PCA Analysis":
    st.markdown('<h2 class="section-header">Principal Component Analysis</h2>', unsafe_allow_html=True)
    
    if not data_loaded or df is None:
        st.error("⚠️ Please load the dataset first! Go to 'Dataset Overview' section.")
        st.stop()
    
    # 8. OBJECTIVE 5: PCA + FEATURE CONTRIBUTIONS
    features = df[['Vict Age', 'Hour', 'LAT', 'LON']]
    scaled_features = StandardScaler().fit_transform(features)
    
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_features)
    
    # Display metrics
    st.subheader("PCA Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PC1 Variance Explained", f"{pca.explained_variance_ratio_[0]*100:.2f}%")
    with col2:
        st.metric("PC2 Variance Explained", f"{pca.explained_variance_ratio_[1]*100:.2f}%")
    with col3:
        total_var = pca.explained_variance_ratio_.sum() * 100
        st.metric("Total Variance Explained", f"{total_var:.2f}%")
    
    # Create loadings dataframe
    loadings = pd.DataFrame(
        pca.components_,
        columns=['Vict Age', 'Hour', 'Latitude', 'Longitude'],
        index=['PC1', 'PC2']
    )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PCA Projection")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use Day_Night label if available, otherwise use Hour
        if 'Day_Night_Label' in df.columns:
            color_data = df['Day_Night_Label']
            color_label = 'Day (1) vs Night (0)'
        else:
            color_data = df['Hour']
            color_label = 'Hour of Day'
        
        scatter = ax.scatter(
            pca_data[:, 0],
            pca_data[:, 1],
            c=color_data,
            cmap='coolwarm',
            alpha=0.4,
            s=10
        )
        
        ax.set_xlabel(f"PC1 – Spatial Component ({pca.explained_variance_ratio_[0]*100:.2f}%)")
        ax.set_ylabel(f"PC2 – Time & Age Component ({pca.explained_variance_ratio_[1]*100:.2f}%)")
        ax.set_title("PCA Projection of Crime Data", fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_label)
        
        # Feature arrows
        for i, feature in enumerate(['Vict Age', 'Hour', 'Latitude', 'Longitude']):
            ax.arrow(0, 0,
                    loadings.iloc[0, i] * 3,
                    loadings.iloc[1, i] * 3,
                    color='black', width=0.01)
            ax.text(loadings.iloc[0, i] * 3.2,
                   loadings.iloc[1, i] * 3.2,
                   feature, fontsize=10, fontweight='bold')
        
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Feature Contributions (Loadings)")
        st.dataframe(loadings.style.background_gradient(cmap='RdBu', axis=None), 
                    use_container_width=True)
        
        st.write("**Interpretation:**")
        st.write("""
        - **Positive loadings**: Strong positive correlation with the component
        - **Negative loadings**: Strong negative correlation
        - **Arrow length**: Strength of contribution
        - **Arrow direction**: Relationship with components
        
        **Key Insights:**
        1. PC1 captures spatial patterns (latitude/longitude)
        2. PC2 captures temporal patterns (hour, victim age)
        """)
    
    # Explained variance plot
    st.subheader("Cumulative Explained Variance")
    pca_full = PCA().fit(scaled_features)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o', linewidth=2)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA Cumulative Variance Explained', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax.legend()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("### 📊 Crime Analysis Dashboard")
st.markdown("This dashboard analyzes crime patterns, predicts trends, and identifies hotspots using machine learning algorithms.")

# Refresh button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()