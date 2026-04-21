from app import load_fred_data, PLOTLY_LAYOUT
import plotly.graph_objects as go
import json

t1, t2 = "UNRATE", "FEDFUNDS"
df1 = load_fred_data(t1)
df2 = load_fred_data(t2)
df_combined = df1.join(df2, how="inner").dropna()

fig = go.Figure(data=go.Scatter(
    x=df_combined[t1], y=df_combined[t2], mode='markers',
    marker=dict(size=6, color="#3b82f6", opacity=0.6),
    name="Relationship"
))
res = fig.to_json()

parsed = json.loads(res)
print("Keys:", parsed.keys())
print("Data X type:", type(parsed['data'][0]['x']))
print("Data X length:", len(parsed['data'][0]['x']))
