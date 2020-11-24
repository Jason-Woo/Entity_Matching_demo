import py_entitymatching as em

# Read the CSV files
A = em.read_csv_metadata('./data/csv_example_messy_input.csv', key='Id')
# Downsample the datasets
sample_A, sample_B = em.down_sample(A, A, size=500, y_param=1, show_progress=False)


# # Attribute Equivalence Blocker
# ab = em.AttrEquivalenceBlocker()
# C0 = ab.block_tables(sample_A, sample_B, 'Phone', 'Phone', l_output_attrs=['Site name', 'Address', 'Phone', 'Zip'], r_output_attrs=['Site name', 'Address', 'Phone', 'Zip'])
# print(C0)
#
# # Overlap Blocker
# ob = em.OverlapBlocker()
# C1 = ob.block_tables(sample_A, sample_B, 'Address', 'Address', overlap_size=1, l_output_attrs=['Site name', 'Address', 'Phone', 'Zip'], r_output_attrs=['Site name', 'Address', 'Phone', 'Zip'])
# print(C1)
#
#
# # Blackbox Blockers
# def match_phone(ltuple, rtuple):
#     l_phone = ltuple['Phone']
#     r_phone = rtuple['Phone']
#     if l_phone != r_phone:
#         return True
#     else:
#         return False
#
#
# bb = em.BlackBoxBlocker()
# bb.set_black_box_function(match_phone)
# C2 = bb.block_tables(sample_A, sample_B, l_output_attrs=['Site name', 'Address', 'Phone', 'Zip'], r_output_attrs=['Site name', 'Address', 'Phone', 'Zip'])
# print(C2.head())


# Rule-Based Blocker
block_f = em.get_features_for_blocking(sample_A, sample_B, validate_inferred_attr_types=False)
print(block_f)
rule1 = ['Address_Address_lev_dist(ltuple, rtuple) > 5']

rb = em.RuleBasedBlocker()
rb.add_rule(rule1, block_f)

C3 = rb.block_tables(sample_A, sample_B, l_output_attrs=['Site name', 'Address', 'Phone', 'Zip'], r_output_attrs=['Site name', 'Address', 'Phone', 'Zip'])
print(len(C3))
print(C3.head())

# # Combining Multiple Blockers
# ab = em.AttrEquivalenceBlocker()
# C0 = ab.block_tables(sample_A, sample_B, 'Phone', 'Phone', l_output_attrs=['Site name', 'Address', 'Phone', 'Zip'], r_output_attrs=['Site name', 'Address', 'Phone', 'Zip'])
# ob = em.OverlapBlocker()
# C4 = ob.block_candset(C0, 'Address', 'Address', overlap_size=1)
# print(C4.head())
