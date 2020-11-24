import py_entitymatching as em

# Read the CSV files
A = em.read_csv_metadata('./data/csv_example_input_with_true_ids.csv', key='Id')
# Downsample the datasets
sample_A, sample_B = em.down_sample(A, A, size=500, y_param=1, show_progress=False)


def match_func(ltuple, rtuple):
    l_phone, l_zip = ltuple['Phone'], ltuple['Zip']
    r_phone, r_zip = rtuple['Phone'], rtuple['Zip']
    if l_phone != r_phone and l_zip != r_zip:
        return True
    else:
        return False


bb = em.BlackBoxBlocker()
bb.set_black_box_function(match_func)
C = bb.block_tables(sample_A, sample_B, l_output_attrs=['Site name', 'Address', 'Phone', 'Zip', 'True Id'], r_output_attrs=['Site name', 'Address', 'Phone', 'Zip', 'True Id'])
ob = em.OverlapBlocker()
C1 = ob.block_candset(C, 'Address', 'Address', word_level=True, overlap_size=2)

# Sample Candidate Set
S = em.sample_table(C1, 450)

# Label the sampled set
# Specify the name for the label column
G = em.label_table(S, 'label')

save_file = False
if save_file:
    em.to_csv_metadata(sample_A, './data/a.csv')
    em.to_csv_metadata(sample_B, './data/b.csv')
    em.to_csv_metadata(G, './data/labelled_data.csv')

