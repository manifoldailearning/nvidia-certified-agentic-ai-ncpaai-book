def check_bias(sample_a, sample_b):
    ratio = (sum(sample_a)/len(sample_a)) / (sum(sample_b)/len(sample_b))
    if ratio < 0.8 or ratio > 1.25:
        print('Potential bias detected.')
    else:
        print('Fair distribution.')

check_bias([0.6, 0.7, 0.65], [0.9, 0.88, 0.87])