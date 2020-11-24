import py_entitymatching as em

# Read the CSV files
sample_A = em.read_csv_metadata('./data/a.csv', key='Id')
sample_B = em.read_csv_metadata('./data/b.csv', key='Id')

G = em.read_csv_metadata('./data/labelled_data.csv', key='_id', fk_ltable='ltable_Id', fk_rtable='rtable_Id',
                         ltable=sample_A, rtable=sample_B)

# Create a set of ML-matchers
dt = em.DTMatcher(name='DecisionTree')
svm = em.SVMMatcher(name='SVM')
rf = em.RFMatcher(name='RF')
lg = em.LogRegMatcher(name='LogReg')
ln = em.LinRegMatcher(name='LinReg')

# Generate a set of features
F = em.get_features_for_matching(sample_A, sample_B, validate_inferred_attr_types=False)
print(F.feature_name)
em.to_csv_metadata(F, "./1.csv")
# Convert the I into a set of feature vectors using F
H = em.extract_feature_vecs(G,
                            feature_table=F,
                            attrs_after='label',
                            show_progress=False)
em.to_csv_metadata(H, './2.csv')

# Impute feature vectors with the mean of the column values.
H = em.impute_table(H,
                    exclude_attrs=['_id', 'ltable_Id', 'rtable_Id', 'label'],
                    strategy='mean')

# Select the best ML matcher using CV
result = em.select_matcher([dt, rf, svm, ln, lg], table=H,
                           exclude_attrs=['_id', 'ltable_Id', 'rtable_Id', 'label'],
                           k=5,
                           target_attr='label', metric_to_select_matcher='f1', random_state=0)
print(result['cv_stats'])
