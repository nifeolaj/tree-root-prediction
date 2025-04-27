import pandas as pd
import numpy as np
import networkx as nx
import ast
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Function to extract centralities
def compute_centralities(edge_list):
    T = nx.from_edgelist(edge_list)
    dc = nx.degree_centrality(T)
    hc = nx.harmonic_centrality(T)
    bc = nx.betweenness_centrality(T)
    pr = nx.pagerank(T)
    try:
        ec = nx.eigenvector_centrality(T)
    except:
        ec = {v: 0 for v in T.nodes()}
    try:
        kc = nx.katz_centrality(T, alpha=0.1)
    except:
        kc = {v: 0 for v in T.nodes()}
    lc = nx.load_centrality(T)
    andc = nx.average_neighbor_degree(T)
    return {v: (dc[v], hc[v], bc[v], pr[v], ec[v], kc[v], lc[v], andc[v]) for v in T}

# 3. Build expanded training data
train_rows = []
for idx, row in train_df.iterrows():
    edges = ast.literal_eval(row['edgelist'])
    T = nx.from_edgelist(edges)
    centralities = compute_centralities(edges)
    root = row['root']
    for node, feats in centralities.items():
        train_rows.append({
            'sentence': row['sentence'],
            'node': node,
            'deg_cent': feats[0],
            'harm_cent': feats[1],
            'betw_cent': feats[2],
            'pagerank': feats[3],
            'eigen_cent': feats[4],
            'katz_cent': feats[5],
            'load_cent': feats[6],
            'avg_neigh_deg': feats[7],
            'n_nodes': row['n'],
            'target': 1 if node == root else 0
        })

expanded_train = pd.DataFrame(train_rows)

# 4. Normalize centralities **per sentence**
features = ['deg_cent', 'harm_cent', 'betw_cent', 'pagerank', 'eigen_cent', 'katz_cent', 'load_cent', 'avg_neigh_deg']
expanded_train[features] = expanded_train.groupby('sentence')[features].transform(lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1,1)).flatten())

# 5. Modeling setup
X = expanded_train[features + ['n_nodes']]
y = expanded_train['target']
groups = expanded_train['sentence']

# 6. Cross-validation evaluation
gkf = GroupKFold(n_splits=5)

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=30,
        learning_rate=0.05,
        n_estimators=300,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
}

print("Cross-validation scores:")
for name, model in models.items():
    fold_accuracies = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        fold_accuracies.append(accuracy_score(y_val, y_pred))
    print(f"{name}: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")

# 7. Train final models
for model in models.values():
    model.fit(X, y)

# 8. Predicting for the test set
submission_rows = []
for idx, row in test_df.iterrows():
    edges = ast.literal_eval(row['edgelist'])
    T = nx.from_edgelist(edges)
    centralities = compute_centralities(edges)

    test_nodes = []
    test_feats = []
    for node, feats in centralities.items():
        test_nodes.append(node)
        test_feats.append(feats)

    test_feats_df = pd.DataFrame(test_feats, columns=features)

    # Normalize per sentence
    test_feats_df = pd.DataFrame(MinMaxScaler().fit_transform(test_feats_df), columns=features)

    # Add n_nodes feature
    test_feats_df['n_nodes'] = row['n']

    # Predict probabilities and average them
    probs = np.zeros(len(test_feats_df))
    for model in models.values():
        probs += model.predict_proba(test_feats_df)[:, 1]
    probs /= len(models)

    # Choose node with highest average probability
    predicted_root = test_nodes[np.argmax(probs)]

    submission_rows.append({
        'id': row['id'],
        'root': predicted_root
    })

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
