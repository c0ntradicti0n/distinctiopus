import list_and_dict_type as L

Predicate_Tuple = L.L(["pred_tuple"])

def print_predicate(p, short = True):
    shorts = ["full_ex", "wff_comp", "wff_dict"]
    if isinstance(p, dict):
        fields = list(p.keys())
        print('\n      ' + ',\n      '.join(
            ['{0:16}: {1}'.format(k, str(p[k]))          for k in fields  if short and k in shorts]))
    else:
        fields = p._fields
        print('\n      ' + ',\n      '.join(
            ['{0:16}: {1}'.format(k, str(getattr(p, k))) for k in fields  if short and k in shorts]))


def print_predicate_list(pl):
    for p_namedtuple in pl:
        print_predicate(p_namedtuple)
    print()
    return None
