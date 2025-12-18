# ==============================
# 1. IMPORT REQUIRED LIBRARIES
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

sns.set_style('whitegrid')

# ==============================
# 2. LOAD DATASET
# ==============================

df = pd.read_csv('crime_dataset.csv')
print("Initial Dataset Shape:", df.shape)

# ==============================
# 3. DATA CLEANING & FEATURE ENGINEERING
# ==============================

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

print("Cleaned Dataset Shape:", df.shape)

# ==============================
# 4. OBJECTIVE 1: CRIME TREND PREDICTION (REGRESSION)
# ==============================

crime_year = df.groupby('Year').size().reset_index(name='Crime_Count')

X = crime_year[['Year']]
y = crime_year['Crime_Count']

lr_model = LinearRegression()
lr_model.fit(X, y)
y_pred = lr_model.predict(X)

print("\nOBJECTIVE 1: CRIME TREND PREDICTION")
print("R² Score:", r2_score(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

plt.figure(figsize=(8,5))
plt.scatter(X, y, label='Actual Crime Count')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Trend')
plt.title("Yearly Crime Trend with Linear Regression", fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Crime Count")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==============================
# 5. OBJECTIVE 2: DAY vs NIGHT CLASSIFICATION
# ==============================

df['Day_Night_Label'] = df['Day_Night'].map({'Day': 1, 'Night': 0})

X_dn = df[['Hour', 'Vict Age']]
y_dn = df['Day_Night_Label']

X_train, X_test, y_train, y_test = train_test_split(
    X_dn, y_dn, test_size=0.3, random_state=42, stratify=y_dn
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
pred = log_model.predict(X_test)

print("\nOBJECTIVE 2: DAY vs NIGHT CLASSIFICATION")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, pred),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Night','Day'],
            yticklabels=['Night','Day'])
plt.title("Day vs Night Confusion Matrix", fontweight='bold')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 6. OBJECTIVE 3: VICTIM GENDER PREDICTION
# ==============================

le = LabelEncoder()
df['Vict Sex'] = le.fit_transform(df['Vict Sex'])

X_gender = df[['Vict Age', 'Hour']]
y_gender = df['Vict Sex']

Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_gender, y_gender, test_size=0.3, random_state=42, stratify=y_gender
)

scaler = StandardScaler()
Xg_train = scaler.fit_transform(Xg_train)
Xg_test = scaler.transform(Xg_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(Xg_train, yg_train)
print("\nKNN Accuracy:", accuracy_score(yg_test, knn.predict(Xg_test)))

nb = GaussianNB()
nb.fit(Xg_train, yg_train)
print("Naive Bayes Accuracy:", accuracy_score(yg_test, nb.predict(Xg_test)))

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(Xg_train, yg_train)
print("Random Forest Accuracy:", accuracy_score(yg_test, rf.predict(Xg_test)))

# ==============================
# 7. OBJECTIVE 4: CRIME HOTSPOTS (K-MEANS)
# ==============================

coords = df[['LAT', 'LON']].sample(50000, random_state=42)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(coords)

plt.figure(figsize=(8,6))
plt.scatter(coords['LON'], coords['LAT'],
            c=clusters, cmap='tab10', s=10, alpha=0.6)
plt.colorbar(label='Cluster ID')
plt.title("Crime Hotspots using K-Means Clustering", fontweight='bold')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# ==============================
# 8. OBJECTIVE 5: PCA + FEATURE CONTRIBUTIONS
# ==============================

features = df[['Vict Age', 'Hour', 'LAT', 'LON']]
scaled_features = StandardScaler().fit_transform(features)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_features)

print("\nOBJECTIVE 5: PCA ANALYSIS")
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

loadings = pd.DataFrame(
    pca.components_,
    columns=['Vict Age','Hour','Latitude','Longitude'],
    index=['PC1','PC2']
)
print("\nPCA Feature Contributions (Loadings):")
print(loadings)

plt.figure(figsize=(10,6))
scatter = plt.scatter(
    pca_data[:,0],
    pca_data[:,1],
    c=df['Day_Night_Label'],
    cmap='coolwarm',
    alpha=0.4,
    s=10
)

plt.xlabel(f"PC1 – Spatial Component ({pca.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 – Time & Age Component ({pca.explained_variance_ratio_[1]*100:.2f}%)")
plt.title("PCA Projection of Crime Data", fontweight='bold')
plt.colorbar(scatter, label='Day (1) vs Night (0)')

# Feature arrows
for i, feature in enumerate(['Vict Age','Hour','Latitude','Longitude']):
    plt.arrow(0, 0,
              loadings.iloc[0,i]*3,
              loadings.iloc[1,i]*3,
              color='black', width=0.01)
    plt.text(loadings.iloc[0,i]*3.2,
             loadings.iloc[1,i]*3.2,
             feature, fontsize=10)

plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print("\n✅ PROJECT EXECUTED SUCCESSFULLY")
