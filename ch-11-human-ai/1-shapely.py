# snippet from the book - for reference only

def explain_prediction(features, weights):
    contributions = {f: round(w * 100, 2) for f, w in zip(features, weights)}
    total = sum(contributions.values())
    print('Feature Contributions (%):', contributions)
    print('Total Influence:', total)

explain_prediction(['age', 'income', 'loan_amount'], [0.2, 0.5, 0.3])
