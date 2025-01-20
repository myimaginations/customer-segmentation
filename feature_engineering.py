import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_features(data):
    """
    Create features like Recency, Frequency, Monetary, and additional metrics.
    """
    # Ensure 'InvoiceDate' is a datetime type
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

    # Calculate RFM metrics
    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (data['InvoiceDate'].max() - x.max()).days,  # Recency
        'InvoiceNo': 'count',  # Frequency
        'Quantity': 'sum',  # Monetary (based on quantity sold)
        'UnitPrice': 'mean'  # Average Order Value
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'Quantity': 'Monetary', 'UnitPrice': 'AOV'})

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm)

    print("Features created and scaled.")
    return scaled_features
