import py_entitymatching as em

# Read the CSV files
A = em.read_csv_metadata('./data/csv_example_messy_input.csv', key='Id')
print(len(A))

# Downsample the datasets
sample_A, sample_B = em.down_sample(A, A, size=500, y_param=1, show_progress=False)
print(len(sample_A), len(sample_B))
