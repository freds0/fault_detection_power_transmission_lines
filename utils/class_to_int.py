
def class_label_to_int(row_label):
    if row_label == 'baliser_ok':
        return 1
    elif row_label == 'baliser_aok':
        return 2
    elif row_label == 'baliser_nok':
        return 3
    elif row_label == 'insulator_ok':
        return 4
    elif row_label == 'insulator_nok':
        return 5
    elif row_label == 'bird_nest':
        return 6
    elif row_label == 'stockbridge_ok':
        return 7
    elif row_label == 'stockbridge_nok':
        return 8
    elif row_label == 'spacer_ok':
        return 9
    elif row_label == 'spacer_nok':
        return 10
    elif row_label == 'insulator_unk':
        return 11
    else:
        print("Error: " + row_label)
        None
