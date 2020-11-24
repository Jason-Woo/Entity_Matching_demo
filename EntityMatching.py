import py_entitymatching as em

# Read the CSV files
A = em.read_csv_metadata('./data/csv_example_input_with_true_ids.csv', key='Id')
A_0 = em.read_csv_metadata('./data/csv_example_messy_input.csv', key='Id')


def match_func(ltuple, rtuple):
    l_phone, l_zip = ltuple['Phone'], ltuple['Zip']
    r_phone, r_zip = rtuple['Phone'], rtuple['Zip']
    if l_phone != r_phone and l_zip != r_zip:
        return True
    else:
        return False


has_label = True
if not has_label:
    # Combining Multiple Blockers
    bb = em.BlackBoxBlocker()
    bb.set_black_box_function(match_func)
    C = bb.block_tables(A, A,
                        l_output_attrs=['Site name', 'Address', 'Zip', 'Phone', 'True Id'],
                        r_output_attrs=['Site name', 'Address', 'Zip', 'Phone', 'True Id'])

    ob = em.OverlapBlocker()
    C1 = ob.block_candset(C,
                          'Address', 'Address',
                          word_level=True,
                          overlap_size=2)

    em.to_csv_metadata(C1, './data/block_tables.csv')

    # Sample Candidate Set
    S = em.sample_table(C1, 400)

    # Label the sampled set
    # Specify the name for the label column
    G = em.label_table(S, 'label')
    em.to_csv_metadata(G, './data/labelled_data_full.csv')

else:
    G = em.read_csv_metadata('./data/labelled_data_full.csv',
                             key='_id',
                             fk_ltable='ltable_Id', fk_rtable='rtable_Id',
                             ltable=A_0, rtable=A_0)
    C = em.read_csv_metadata('./data/block_tables.csv',
                             key='_id',
                             fk_ltable='ltable_Id', fk_rtable='rtable_Id',
                             ltable=A_0, rtable=A_0)

    train_test = em.split_train_test(G, train_proportion=0.7)
    devel_set = train_test['train']
    eval_set = train_test['test']

    # Generate a set of features
    F = em.get_features_for_matching(A_0, A_0, validate_inferred_attr_types=False)
    H_train = em.extract_feature_vecs(devel_set, feature_table=F, attrs_after='label')
    H_eval = em.extract_feature_vecs(eval_set, feature_table=F, attrs_after='label')
    C_test = em.extract_feature_vecs(C, feature_table=F)

    # Impute feature vectors with the mean of the column values.
    H_train = em.impute_table(H_train,
                              exclude_attrs=['_id', 'ltable_Id', 'rtable_Id', 'label'],
                              strategy='mean')
    H_eval = em.impute_table(H_eval,
                             exclude_attrs=['_id', 'ltable_Id', 'rtable_Id', 'label'],
                             strategy='mean')
    C_test = em.impute_table(C_test,
                             exclude_attrs=['_id', 'ltable_Id', 'rtable_Id'],
                             strategy='mean')

    # Train Matcher
    rf = em.RFMatcher(name='RF')
    rf.fit(table=H_train,
           exclude_attrs=['_id', 'ltable_Id', 'rtable_Id', 'label'],
           target_attr='label')

    # Test1
    pred_table = rf.predict(table=H_eval,
                            exclude_attrs=['_id', 'ltable_Id', 'rtable_Id', 'label'],
                            append=True,
                            target_attr='predicted_labels')

    eval_summary = em.eval_matches(pred_table, 'label', 'predicted_labels')
    print(eval_summary)

    # Test2
    pred_table = rf.predict(table=C_test,
                            exclude_attrs=['_id', 'ltable_Id', 'rtable_Id'],
                            append=True, target_attr='predicted_labels')
    em.to_csv_metadata(C_test, './data/result.csv')