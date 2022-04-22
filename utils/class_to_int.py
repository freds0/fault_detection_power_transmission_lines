
def class_label_to_int(row_label, labels_list):

    if row_label not in labels_list:
        print("Error: " + row_label)
        return None

    for index, class_label in enumerate(labels_list):
        if row_label == class_label:
            return index + 1 # Index must start with 1
