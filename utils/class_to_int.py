
def class_text_to_int(row_label):
    if row_label == 'esfera_boa':
        return 1
    elif row_label == 'esfera_razoavel':
        return 2
    elif row_label == 'esfera_ruim':
        return 3
    elif row_label == 'isolador_ok':
        return 4
    elif row_label == 'isolador_falha':
        return 5
    elif row_label == 'ninho':
        return 6
    else:
        print("Erro: " + row_label)
        None
